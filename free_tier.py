"""
Free-tier models and news search for Tom's bot (Fall 2026, version 2).

Metaculus declined the LLM-credit request, so the bot runs on free tiers only.
Status checked on 27 Sep 2026:

  Mistral, free "Experiment" plan (MISTRAL_API_KEY)
      Mistral counts "requests per second" as requests running at the same
      time, and the free plan allows about one, so every Mistral call here
      waits for the previous one to finish. About 1 billion tokens a month.
  Groq, free plan (GROQ_API_KEY)
      gpt-oss-120b, gpt-oss-20b and qwen3.8-27b: each 30 requests/minute,
      1,000 requests/day, 8,000 tokens/minute and 200,000 tokens/day.
      8,000 tokens/minute means about one forecast per minute per model.
      The gpt-oss models can search the web (Groq's "browser_search" tool).
  Google Gemini, free tier (GEMINI_API_KEY, optional)
      Flash about 20 requests/day, Flash-Lite about 500 requests/day
      (measured by developers in Sep 2026; Google no longer publishes them).
  GDELT (no key): recent headlines, one request every 5 seconds.

No longer available: GitHub Models (retired 30 Jul 2026) and Groq Compound
(retired 21 Sep 2026); version 1 of this file used both.

Every provider is optional; the bot uses whatever keys are present. When a
model fails, the call moves on to the next model. A model that keeps failing
or has used up its daily quota is skipped for the rest of the run, and a
report at the end of the run shows what happened to each model.

Keys are read from environment variables by LiteLLM and are never put into
the model settings, because forecasting-tools prints those settings in the
explanations it publishes.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

_PLACEHOLDERS = {"", "REPLACE_ME", "1234567890", "your-api-key-here"}


def has_key(name: str) -> bool:
    return (os.getenv(name) or "").strip() not in _PLACEHOLDERS


# ---------------------------------------------------------------- pacing
class ProviderGate:
    """Spaces out request starts for one provider (or one model), caps how many
    requests run at once, and remembers when it can't be used any more in this
    run (bad key)."""

    def __init__(self, name: str, min_interval: float, max_concurrent: int) -> None:
        self.name = name
        self.min_interval = min_interval
        self.max_concurrent = max_concurrent
        self.disabled_reason: str | None = None
        self._loop: Any = None
        self._sem: asyncio.Semaphore | None = None
        self._lock: asyncio.Lock | None = None
        self._last_start = 0.0

    def _bind(self) -> None:
        # asyncio primitives belong to one event loop; make fresh ones per loop
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def slot(self):
        self._bind()
        assert self._sem is not None and self._lock is not None
        async with self._sem:
            async with self._lock:
                wait = self._last_start + self.min_interval - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_start = time.monotonic()
            yield


# ---------------------------------------------------------------- errors and the end-of-run report
# Errors after which a model is skipped for the rest of the run.
_MODEL_ENDING_ERRORS = (
    "per day", "perday", "86400", "daily", "(rpd)", "(tpd)", "per month", "monthly",
    "not found", "does not exist", "unknown model", "invalid model", "model_not_found",
    "unavailable model", "unavailable_model", "not available", "no access", "not allowed",
    "decommissioned", "retired", "deprecated", "no longer supported",
)
# Errors after which the whole provider is skipped (bad or missing key, no permission).
_PROVIDER_ENDING_ERRORS = (
    "authenticationerror", "permissiondeniederror", "unauthorized", "invalid api key",
    "invalid_api_key", "forbidden",
)
MAX_CONSECUTIVE_FAILURES = 4


def _matches(error: BaseException, markers: tuple[str, ...]) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in markers)


@dataclass
class ModelStats:
    ok: int = 0
    failed: int = 0
    consecutive_failures: int = 0
    skipped_because: str | None = None
    last_error: str = ""


STATS: dict[str, ModelStats] = {}
NOTES: dict[str, int] = {}  # counters for the report, e.g. keyless searches used


def note(key: str) -> None:
    NOTES[key] = NOTES.get(key, 0) + 1


def free_model_report() -> str:
    lines = ["=" * 30 + " Free-model report " + "=" * 30]
    if not STATS:
        lines.append("No model was called in this run.")
    for model, st in sorted(STATS.items()):
        line = f"{model:<36} ok {st.ok:>3}   failed {st.failed:>3}"
        if st.skipped_because:
            line += f"   SKIPPED: {st.skipped_because}"
        elif st.last_error:
            line += f"   last error: {st.last_error}"
        lines.append(line)
    for key, n in sorted(NOTES.items()):
        lines.append(f"{key}: {n}")
    lines.append("=" * 79)
    return "\n".join(lines)


def estimate_tokens(prompt: Any) -> int:
    # conservative: about 3 characters per token, plus message overhead
    return int(len(str(prompt)) / 3) + 100


class FreeTierLlm(GeneralLlm):
    """A GeneralLlm that waits for its gate, refuses prompts that are too long
    for a free tier, and is skipped for the rest of the run after errors that
    won't go away (daily quota used up, model retired, repeated failures)."""

    def __init__(self, model: str, gate: ProviderGate, max_input_tokens: int | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._gate = gate
        self._max_input_tokens = max_input_tokens

    @property
    def stats(self) -> ModelStats:
        return STATS.setdefault(self.model, ModelStats())

    def unavailable_reason(self) -> str | None:
        return self._gate.disabled_reason or self.stats.skipped_because

    def can_take(self, prompt: Any) -> bool:
        if self.unavailable_reason():
            return False
        return self._max_input_tokens is None or estimate_tokens(prompt) <= self._max_input_tokens

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"{self.model} skipped for this run: {reason}")
        st = self.stats
        async with self._gate.slot():
            reason = self.unavailable_reason()  # may have changed while waiting for the slot
            if reason:
                raise RuntimeError(f"{self.model} skipped for this run: {reason}")
            try:
                result = await super()._mockable_direct_call_to_model(prompt)
            except Exception as e:
                st.failed += 1
                st.consecutive_failures += 1
                note_text = f"{type(e).__name__}: {' '.join(str(e).split())[:220]}"
                st.last_error = note_text
                if _matches(e, _MODEL_ENDING_ERRORS):  # checked first: "no access to this model" is about one model
                    st.skipped_because = note_text
                    logger.warning(f"{self.model}: {note_text} - skipping this model for the rest of the run")
                elif _matches(e, _PROVIDER_ENDING_ERRORS):
                    self._gate.disabled_reason = note_text
                    logger.warning(f"{self._gate.name}: {note_text} - skipping this provider for the rest of the run")
                elif st.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    st.skipped_because = f"{st.consecutive_failures} failures in a row; last: {note_text}"
                    logger.warning(f"{self.model}: {st.skipped_because} - skipping this model for the rest of the run")
                else:
                    logger.warning(f"{self.model}: {note_text}")
                raise
        st.ok += 1
        st.consecutive_failures = 0
        return result


class FreeModelMix(GeneralLlm):
    """Tries several free models. With rotate=True each call starts at the next
    model, so the predictions for a question come from different models (an
    ensemble); otherwise the list is an ordered fallback chain. When every
    model fails with a passing error (rate limit, overload), it waits and tries
    again, up to len(waits) more rounds."""

    def __init__(self, name: str, members: list[FreeTierLlm], rotate: bool, waits: tuple[float, ...] = (30.0, 90.0)) -> None:
        if not members:
            raise ValueError("FreeModelMix needs at least one model")
        super().__init__(model=name, allowed_tries=1)
        self.members = members
        self.rotate = rotate
        self.waits = waits
        self._next = 0

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        n = len(self.members)
        start = self._next % n if self.rotate else 0
        if self.rotate:
            self._next += 1
        errors: list[str] = []
        for round_no in range(len(self.waits) + 1):
            tried = 0
            for k in range(n):
                member = self.members[(start + k) % n]
                if not member.can_take(prompt):
                    continue
                tried += 1
                try:
                    answer = await member.invoke(prompt, system_prompt)
                    logger.info(f"{self.model}: answered by {member.model}")
                    return answer
                except Exception as e:
                    errors.append(f"{member.model}: {type(e).__name__}")
            if tried == 0 or round_no == len(self.waits):
                break
            logger.info(f"{self.model}: every model failed this round; waiting {self.waits[round_no]:.0f}s before trying again")
            await asyncio.sleep(self.waits[round_no])
        skipped = [m.model for m in self.members if not m.can_take(prompt)]
        detail = "; ".join(errors[-8:]) or "no model could take this prompt"
        if skipped:
            detail += f" (skipped: {', '.join(skipped)})"
        raise RuntimeError(f"All free models failed for {self.model}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_model": self.model,
            "mode": "rotation" if self.rotate else "fallback",
            "models": [m.model for m in self.members],
        }


# ---------------------------------------------------------------- the lineup
MISTRAL = ProviderGate("mistral", min_interval=1.2, max_concurrent=1)
# Groq limits are per model; 8,000 tokens/minute ~ one big request per minute each
GROQ_120B = ProviderGate("groq-gpt-oss-120b", min_interval=62.0, max_concurrent=1)
GROQ_20B = ProviderGate("groq-gpt-oss-20b", min_interval=62.0, max_concurrent=1)
GROQ_QWEN = ProviderGate("groq-qwen3.8-27b", min_interval=62.0, max_concurrent=1)
GEMINI_FLASH = ProviderGate("gemini-flash", min_interval=13.0, max_concurrent=1)
GEMINI_LITE = ProviderGate("gemini-flash-lite", min_interval=4.5, max_concurrent=1)
GATES = [MISTRAL, GROQ_120B, GROQ_20B, GROQ_QWEN, GEMINI_FLASH, GEMINI_LITE]


def build_free_llms() -> tuple[dict[str, Any] | None, list[str]]:
    """Returns (llms for the bot, providers found). llms is None when there is
    no usable key at all."""
    mistral = has_key("MISTRAL_API_KEY")
    groq = has_key("GROQ_API_KEY")
    gemini = has_key("GEMINI_API_KEY")
    providers = [p for p, ok in (("Mistral", mistral), ("Groq", groq), ("Gemini", gemini)) if ok]
    if not providers:
        return None, []

    def m(model: str, temperature: float = 0.3, timeout: int = 240) -> FreeTierLlm:
        return FreeTierLlm(f"mistral/{model}", MISTRAL, temperature=temperature, timeout=timeout, allowed_tries=3)

    def gq(model: str, gate: ProviderGate, **kw: Any) -> FreeTierLlm:
        # prompt + max_tokens must stay under 8,000 tokens/minute
        return FreeTierLlm(f"groq/{model}", gate, max_input_tokens=4500, temperature=kw.pop("temperature", 0.3),
                           timeout=180, allowed_tries=1, max_tokens=kw.pop("max_tokens", 3000), **kw)

    def gm(model: str, gate: ProviderGate, temperature: float = 0.3) -> FreeTierLlm:
        return FreeTierLlm(f"gemini/{model}", gate, temperature=temperature, timeout=180, allowed_tries=1)

    forecasters: list[FreeTierLlm] = []
    if mistral:
        forecasters.append(m("mistral-medium-latest"))
    if groq:
        forecasters.append(gq("openai/gpt-oss-120b", GROQ_120B, reasoning_effort="low"))
    if gemini:
        forecasters.append(gm("gemini-flash-latest", GEMINI_FLASH))
    if mistral:
        forecasters.append(m("magistral-medium-latest", timeout=300))
    if groq:
        forecasters.append(gq("qwen/qwen3.8-27b", GROQ_QWEN))
    if mistral:
        forecasters.append(m("mistral-large-latest"))
    if gemini:
        forecasters.append(gm("gemini-flash-lite-latest", GEMINI_LITE))
    if groq and not mistral:
        forecasters.append(gq("openai/gpt-oss-20b", GROQ_20B, reasoning_effort="low"))

    writers: list[FreeTierLlm] = []  # research plans and fact sheets: long prompts
    if mistral:
        writers += [m("mistral-medium-latest"), m("mistral-large-latest"), m("mistral-small-latest")]
    if gemini:
        writers.append(gm("gemini-flash-lite-latest", GEMINI_LITE))
    if groq:
        writers.append(gq("openai/gpt-oss-120b", GROQ_120B, reasoning_effort="low"))

    parsers: list[FreeTierLlm] = []  # only used when the answer can't be read directly
    if mistral:
        parsers.append(m("mistral-small-latest", temperature=0))
    if gemini:
        parsers.append(gm("gemini-flash-lite-latest", GEMINI_LITE, temperature=0))
    if groq:
        parsers.append(gq("openai/gpt-oss-20b", GROQ_20B, temperature=0, reasoning_effort="low", max_tokens=2000))

    llms: dict[str, Any] = {
        "default": FreeModelMix("free-forecasters", forecasters, rotate=True),
        "writer": FreeModelMix("free-writer", writers, rotate=False),
        "summarizer": FreeModelMix("free-summarizer", parsers, rotate=False),
        "parser": FreeModelMix("free-parser", parsers, rotate=False, waits=(20.0,)),
        "researcher": "free-news",  # keyless GDELT headlines per claim (main.py)
        "web": None,
    }
    if groq:  # one web search per question with Groq's browser_search tool
        search = dict(tools=[{"type": "browser_search"}], tool_choice="required", reasoning_effort="low",
                      temperature=0.2, max_tokens=1800)
        llms["web"] = FreeModelMix(
            "free-web-search",
            [gq("openai/gpt-oss-20b", GROQ_20B, **dict(search)), gq("openai/gpt-oss-120b", GROQ_120B, **dict(search))],
            rotate=False, waits=(),
        )
    return llms, providers


# ---------------------------------------------------------------- reading answers without a model
_PCT = r"([0-9]+(?:[.,][0-9]+)?)\s*%"


def read_binary(text: str) -> float | None:
    """The last 'Probability: NN%' near the end of the text, as a fraction."""
    hits = re.findall(r"probability\W{0,6}" + _PCT, text[-400:], flags=re.IGNORECASE)
    if not hits:
        return None
    value = float(hits[-1].replace(",", "."))
    return value / 100 if 0 <= value <= 100 else None


def read_options(text: str, options: list[str]) -> dict[str, float] | None:
    """The last 'Option: NN%' line for every option, normalised to sum to 1.
    None if any option is missing or the numbers don't look like a final answer."""
    text = text[-2500:]  # the final answer is the last thing written
    found: dict[str, float] = {}
    for opt in options:
        pattern = (r"(?im)^[\s\-\*•#>\"'“]*(?:option[\s_]*[a-z0-9]?[\s:)\-–.]*)?[\"'“*]*"
                   + re.escape(opt.strip()) + r"[\"'”*]*\s*[:=\-–]\s*\**\s*" + _PCT)
        hits = re.findall(pattern, text)
        if not hits:
            return None
        found[opt] = float(hits[-1].replace(",", "."))
    total = sum(found.values())
    if not 80 <= total <= 120:
        return None
    return {k: v / total for k, v in found.items()}


def read_percentiles(text: str) -> dict[int, float] | None:
    """The last 'Percentile NN: value' line for 10/20/40/60/80/90, if all six
    are present and increasing."""
    text = text[-1500:]  # the final answer is the last thing written
    found: dict[int, float] = {}
    pattern = r"(?im)^[\s\-\*•\"']*percentile\s*(10|20|40|60|80|90)\s*[:=]\s*\**\s*\$?\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
    for pct, number in re.findall(pattern, text):
        try:
            found[int(pct)] = float(number.replace(",", ""))
        except ValueError:
            return None
    keys = [10, 20, 40, 60, 80, 90]
    if any(k not in found for k in keys):
        return None
    values = [found[k] for k in keys]
    if any(b < a for a, b in zip(values, values[1:])):
        return None
    return found


# ---------------------------------------------------------------- keyless news
# GDELT's DOC API is an open news index (headlines, dates, sources in ~65
# languages) meant for programmatic use; it asks for at most one request every
# 5 seconds.
_UA = {"User-Agent": "Mozilla/5.0 (compatible; metaculus-forecasting-bot/1.0)"}
GDELT = ProviderGate("gdelt", min_interval=5.5, max_concurrent=1)

_STOPWORDS = set("""
a an the and or of to in on at by for from with without into over under about after before between during
is are was were be been being will would shall should can could may might must do does did has have had
what which who whom whose when where why how whether than then this that these those it its as not no
yes any all each every more most less least than per via vs versus next last new before after until
""".split())


def _keywords(query: str, limit: int = 5) -> list[str]:
    words = re.findall(r"[^\W_]+", str(query), flags=re.UNICODE)
    keep = [w for w in words if len(w) > 2 and w.lower() not in _STOPWORDS]
    return keep[:limit]


def gdelt_news(query: str, language: str | None = None, limit: int = 8) -> list[dict]:
    words = _keywords(query)
    if not words:
        return []
    q = " ".join(words)
    if language and language.strip().lower() != "english":
        q += f" sourcelang:{language.strip().lower()}"
    r = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={"query": q, "mode": "ArtList", "maxrecords": limit, "format": "json", "sort": "DateDesc"},
        headers=_UA,
        timeout=25,
    )
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError:  # GDELT reports problems as plain text with HTTP 200
        raise RuntimeError(f"GDELT: {r.text[:120]}")
    items = []
    for a in (data or {}).get("articles", []):
        date = None
        try:
            date = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        items.append({"title": a.get("title", ""), "source": a.get("domain", ""), "date": date, "link": a.get("url", "")})
    items.sort(key=lambda x: x["date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items[:limit]


async def free_news_search(query: str, language: str | None = None, limit: int = 8) -> str:
    """Recent headlines from GDELT (in the requested language when given,
    otherwise any language). Needs no key. Returns plain text for the fact sheet."""
    query = " ".join(str(query).split())[:200]
    lang = language if language and language.strip().lower() != "english" else None
    attempts = [lang, None] if lang else [None]
    items: list[dict] = []
    used_lang = None
    for attempt in attempts:
        try:
            async with GDELT.slot():
                items = await asyncio.to_thread(gdelt_news, query, attempt, limit)
        except Exception as e:
            logger.info(f"GDELT search failed for '{query}' ({attempt or 'any language'}): {type(e).__name__}: {str(e)[:120]}")
            items = []
        if items:
            used_lang = attempt
            break
    note("GDELT searches with results" if items else "GDELT searches without results")
    label = f"{used_lang} sources" if used_lang else "any language"
    if not items:
        return f"[keyless news search] No recent headlines found for: {query}"
    lines = [f"[keyless news search via GDELT, {label}; headlines only] Query: {query}"]
    for it in items:
        day = it["date"].strftime("%Y-%m-%d") if it["date"] else "date unknown"
        source = f" ({it['source']})" if it["source"] else ""
        lines.append(f"- {day}{source}: {it['title']} <{it['link']}>")
    return "\n".join(lines)
