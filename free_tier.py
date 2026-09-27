"""
Free-tier models and news search for Tom's bot (Fall 2026).

Metaculus declined the LLM-credit request, so the bot runs on free tiers only:

  Mistral (free "Experiment" plan)   1 request/second, 1 billion tokens/month -> most calls
  GitHub Models (free, no new key)   GPT-4.1 through the workflow's GITHUB_TOKEN;
                                     a few dozen requests/day, 8k tokens in, 4k out
  Groq (free)                        gpt-oss-120b (1k requests/day, 8k tokens/minute)
                                     and Compound, a model with built-in web search

Every provider is optional; the bot uses whatever keys are present. When a
provider hits a limit, the call moves on to the next model, and a provider
whose daily quota is used up is skipped for the rest of the run.

Keys are read from environment variables by LiteLLM and are never put into the
model settings, because forecasting-tools prints those settings in the
explanations it publishes.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

_PLACEHOLDERS = {"", "REPLACE_ME", "1234567890", "your-api-key-here"}

# GitHub Models' current endpoint (LiteLLM still defaults to the retired Azure one)
os.environ.setdefault("GITHUB_API_BASE", "https://models.github.ai/inference")


def has_key(name: str) -> bool:
    return (os.getenv(name) or "").strip() not in _PLACEHOLDERS


# ---------------------------------------------------------------- rate limits
class ProviderGate:
    """Spaces out request starts for one provider, caps parallel requests, and
    remembers when a provider can't be used any more in this run (daily quota
    used up, model not available)."""

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


# Errors after which a model is skipped for the rest of the run (free limits
# are mostly per model: GitHub counts per model per day, Groq per model per day).
# Per-minute limits are not in this list: those are retried or passed on.
_MODEL_ENDING_ERRORS = (
    "per day", "perday", "byday", "86400", "daily", "(rpd)", "(tpd)", "per month",
    "quota", "insufficient", "not found", "does not exist", "unknown model",
    "invalid model", "model_not_found", "unavailable_model", "not available",
    "no access", "not allowed", "decommissioned", "deprecated",
)
# Errors after which the whole provider is skipped (bad or missing key, no permission).
_PROVIDER_ENDING_ERRORS = (
    "authenticationerror", "permissiondeniederror", "unauthorized", "invalid api key",
    "invalid_api_key", "forbidden", "permission",
)
_DISABLED_MODELS: dict[str, str] = {}


def _matches(error: BaseException, markers: tuple[str, ...]) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in markers)


def estimate_tokens(prompt: Any) -> int:
    return int(len(str(prompt)) / 3.5) + 50


class FreeTierLlm(GeneralLlm):
    """A GeneralLlm that waits for its provider's gate and can refuse prompts
    that are too long for a free tier."""

    def __init__(
        self,
        model: str,
        gate: ProviderGate,
        max_input_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._gate = gate
        self._max_input_tokens = max_input_tokens

    @property
    def provider(self) -> str:
        return self._gate.name

    def unavailable_reason(self) -> str | None:
        return self._gate.disabled_reason or _DISABLED_MODELS.get(self.model)

    def can_take(self, prompt: Any) -> bool:
        if self.unavailable_reason():
            return False
        return self._max_input_tokens is None or estimate_tokens(prompt) <= self._max_input_tokens

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"{self.model} skipped for this run: {reason}")
        async with self._gate.slot():
            try:
                return await super()._mockable_direct_call_to_model(prompt)
            except Exception as e:
                note = f"{type(e).__name__}: {str(e)[:160]}"
                if _matches(e, _MODEL_ENDING_ERRORS):  # checked first: "no access to this model" is about one model
                    _DISABLED_MODELS[self.model] = note
                    logger.warning(f"{self.model}: {note} - skipping this model for the rest of the run")
                elif _matches(e, _PROVIDER_ENDING_ERRORS):
                    self._gate.disabled_reason = note
                    logger.warning(f"{self._gate.name}: {note} - skipping this provider for the rest of the run")
                raise


class FreeModelMix(GeneralLlm):
    """Tries several free models. With rotate=True each call starts at the next
    model, so the five predictions for a question come from different models
    (an ensemble); otherwise the list is an ordered fallback chain."""

    def __init__(self, name: str, members: list[FreeTierLlm], rotate: bool) -> None:
        if not members:
            raise ValueError("FreeModelMix needs at least one model")
        super().__init__(model=name, allowed_tries=1)
        self.members = members
        self.rotate = rotate
        self._next = 0

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        n = len(self.members)
        start = self._next % n if self.rotate else 0
        if self.rotate:
            self._next += 1
        errors: list[str] = []
        for k in range(n):
            member = self.members[(start + k) % n]
            if not member.can_take(prompt):
                errors.append(f"{member.model}: skipped")
                continue
            try:
                answer = await member.invoke(prompt, system_prompt)
                logger.info(f"{self.model}: answered by {member.model}")
                return answer
            except Exception as e:
                errors.append(f"{member.model}: {type(e).__name__}")
                logger.warning(f"{self.model}: {member.model} failed ({type(e).__name__}: {str(e)[:200]}); trying the next model")
        raise RuntimeError(f"All free models failed for {self.model}: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_model": self.model,
            "mode": "rotation" if self.rotate else "fallback",
            "models": [m.model for m in self.members],
        }


# ---------------------------------------------------------------- the lineup
MISTRAL = ProviderGate("mistral", min_interval=1.15, max_concurrent=3)
GITHUB = ProviderGate("github-models", min_interval=6.5, max_concurrent=2)
GROQ = ProviderGate("groq", min_interval=8.0, max_concurrent=1)
GROQ_SEARCH = ProviderGate("groq-compound", min_interval=2.5, max_concurrent=2)
GROQ_SMALL = ProviderGate("groq-small", min_interval=2.5, max_concurrent=2)


def build_free_llms() -> tuple[dict[str, Any] | None, list[str]]:
    """Returns (llms for the bot, providers found). llms is None when there is
    no usable key at all."""
    mistral = has_key("MISTRAL_API_KEY")
    github = has_key("GITHUB_API_KEY")  # the workflow passes GITHUB_TOKEN here
    groq = has_key("GROQ_API_KEY")
    providers = [p for p, ok in (("Mistral", mistral), ("GitHub Models", github), ("Groq", groq)) if ok]
    if not (mistral or github or groq):
        return None, []

    def m(model: str, **kw: Any) -> FreeTierLlm:
        return FreeTierLlm(f"mistral/{model}", MISTRAL, temperature=kw.pop("temperature", 0.3), timeout=kw.pop("timeout", 180), allowed_tries=3, **kw)

    def gh(model: str, **kw: Any) -> FreeTierLlm:
        return FreeTierLlm(f"github/{model}", GITHUB, max_input_tokens=7000, temperature=kw.pop("temperature", 0.3), timeout=120, allowed_tries=1, max_tokens=3500, **kw)

    def gq(model: str, gate: ProviderGate, max_in: int | None, **kw: Any) -> FreeTierLlm:
        return FreeTierLlm(f"groq/{model}", gate, max_input_tokens=max_in, temperature=kw.pop("temperature", 0.3), timeout=kw.pop("timeout", 120), allowed_tries=1, **kw)

    # Forecasters: one prediction from each, in turn (5 predictions per question).
    forecasters: list[FreeTierLlm] = []
    if mistral:
        forecasters.append(m("mistral-large-latest"))
    if github:
        forecasters.append(gh("openai/gpt-4.1"))
    if mistral:
        forecasters.append(m("magistral-medium-latest", timeout=300))
    if groq:
        forecasters.append(gq("openai/gpt-oss-120b", GROQ, 5000, max_tokens=2500, reasoning_effort="low"))
    if mistral:
        forecasters.append(m("mistral-medium-latest"))
    if not mistral and github:
        forecasters.append(gh("openai/gpt-4o"))

    # Writer (research plans and fact sheets): long prompts, so Mistral first.
    writers: list[FreeTierLlm] = []
    if mistral:
        writers.append(m("mistral-large-latest"))
    if github:
        writers.append(gh("openai/gpt-4.1"))
    if groq:
        writers.append(gq("openai/gpt-oss-120b", GROQ, 5000, max_tokens=2500))

    # Parser and summarizer: small, fast models.
    parsers: list[FreeTierLlm] = []
    if mistral:
        parsers.append(m("mistral-small-latest", temperature=0, timeout=90))
    if github:
        parsers.append(gh("openai/gpt-4.1-mini", temperature=0))
    if groq:
        parsers.append(gq("llama-3.1-8b-instant", GROQ_SMALL, 5000, temperature=0, max_tokens=2000))

    # Researcher: Groq Compound searches the web. Without Groq (or if it fails),
    # main.py uses the keyless news search below.
    researcher: Any = "free-news"
    if groq:
        researcher = FreeModelMix(
            "free-web-search",
            [
                gq("groq/compound", GROQ_SEARCH, None, temperature=0.1, timeout=180, max_tokens=3000),
                gq("groq/compound-mini", GROQ_SEARCH, None, temperature=0.1, timeout=180, max_tokens=3000),
            ],
            rotate=False,
        )

    llms = {
        "default": FreeModelMix("free-forecasters", forecasters, rotate=True),
        "writer": FreeModelMix("free-writer", writers, rotate=False),
        "summarizer": FreeModelMix("free-summarizer", parsers, rotate=False),
        "parser": FreeModelMix("free-parser", parsers, rotate=False),
        "researcher": researcher,
    }
    return llms, providers


# ---------------------------------------------------------------- keyless news
# GDELT's DOC API is an open news index (headlines, dates, sources in ~65
# languages) meant for programmatic use; it asks for at most one request every
# 5 seconds. Used only when the web-search model is unavailable.
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
    label = f"{used_lang} sources" if used_lang else "any language"
    if not items:
        return f"[keyless news search] No recent headlines found for: {query}"
    lines = [f"[keyless news search via GDELT, {label}; headlines only] Query: {query}"]
    for it in items:
        day = it["date"].strftime("%Y-%m-%d") if it["date"] else "date unknown"
        source = f" ({it['source']})" if it["source"] else ""
        lines.append(f"- {day}{source}: {it['title']} <{it['link']}>")
    return "\n".join(lines)
