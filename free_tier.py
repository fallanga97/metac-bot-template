"""
Free-tier models and news search for Tom's bot (Fall 2026, version 3).

Metaculus declined the LLM-credit request, so the bot runs on free tiers only.
Status checked on 27 Sep 2026, after two live test runs:

  Google Gemini, free tier (GEMINI_API_KEY) - strongly recommended
      gemini-3.8-flash: about 20 requests/day; gemini-3.5-flash-lite and
      gemini-3.1-flash-lite: several hundred requests/day each (developer
      measurements; Google shows the exact numbers only in AI Studio).
      Temperature stays at Gemini 3's default of 1.0, as Google recommends.
      Google Search grounding is not part of the free tier.
  Groq, free plan (GROQ_API_KEY)
      gpt-oss-120b and gpt-oss-20b: each 30 requests/minute, 1,000/day,
      8,000 tokens/minute and 200,000 tokens/day. A forecast uses ~7,000
      tokens and a web search (Groq's browser_search tool) ~12,000, so each
      model manages roughly 20-25 calls a day. Groq is kept for forecasts,
      the web search and, as a last resort, reading answers; it no longer
      writes research plans or fact sheets. qwen3.8-27b is not used: the
      free plan allows it only 1,000 output tokens per minute.
  Mistral, free "Experiment" plan (MISTRAL_API_KEY)
      One request at a time. Mistral Large is not included. In the second
      test run every Mistral call was refused with "Rate limit exceeded";
      the limits are shown in Mistral's console (Admin > API > Limits).
  GDELT (no key): recent headlines, one request every 5 seconds.

No longer available: GitHub Models (retired 30 Jul 2026) and Groq Compound
(retired 21 Sep 2026); version 1 of this file used both.

Every provider is optional; the bot uses whatever keys are present. Each of
the three forecasts per question starts with a different provider and falls
back to the others. A model that has used up its daily quota, keeps failing,
or whose provider answers "rate limit" six times in a row is skipped for the
rest of the run (each rate-limit answer also spaces that provider's requests
further apart), and a report at the end of the run shows what happened to
each model.

Keys are read from environment variables by LiteLLM and are never put into
the model settings, because forecasting-tools prints those settings in the
explanations it publishes. Error texts are scrubbed of keys before they are
logged or reported.
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


# ---------------------------------------------------------------- keeping keys out of logs
_SECRET_ENV_NAMES = (
    "MISTRAL_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "METACULUS_TOKEN",
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY",
    "EXA_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET",
)
_KEY_SHAPES = re.compile(
    r"(gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9_\-]{16,}|(?<=key=)[^&\s\"']+)"
)


def redact(text: str) -> str:
    """Removes anything that looks like an API key or token from a message."""
    for name in _SECRET_ENV_NAMES:
        value = (os.getenv(name) or "").strip()
        if len(value) >= 8:
            text = text.replace(value, "***")
    return _KEY_SHAPES.sub("***", text)


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
        self.rate_limits_in_a_row = 0
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

    def slow_down(self, factor: float = 1.6, cap: float = 60.0) -> None:
        """After a rate-limit error: space requests further apart for the rest
        of the run (the free tiers don't publish all their limits)."""
        new = min(cap, self.min_interval * factor)
        if new > self.min_interval:
            logger.info(f"{self.name}: rate limited - now {new:.1f}s between requests")
            self.min_interval = new


# ---------------------------------------------------------------- errors and the end-of-run report
# Errors after which a model is skipped for the rest of the run:
# its daily (or monthly) quota is used up ...
_QUOTA_ERRORS = (
    "per day", "perday", "86400", "daily", "(rpd)", "(tpd)", "per month", "permonth", "monthly",
)
# ... or it isn't available at all.
_GONE_ERRORS = (
    "not found", "does not exist", "unknown model", "invalid model", "model_not_found",
    "unavailable model", "unavailable_model", "not available", "no access", "not allowed",
    "decommissioned", "retired", "deprecated", "no longer supported",
)
# Errors after which the whole provider is skipped (bad or missing key, no permission).
_PROVIDER_ENDING_ERRORS = (
    "authenticationerror", "permissiondeniederror", "unauthorized", "invalid api key",
    "invalid_api_key", "forbidden",
)
MAX_CONSECUTIVE_FAILURES = 4  # per model, for errors other than rate limits
# Rate limits are counted per provider (Mistral's apply to the whole account;
# Groq's and Gemini's gates are per model anyway). Each one also spaces that
# provider's requests further apart.
MAX_CONSECUTIVE_RATE_LIMITS = 6
RETRY_WAIT = 8.0  # seconds before the second try of a model (times the try number)


def _matches(error: BaseException, markers: tuple[str, ...]) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in markers)


def _is_rate_limit(error: BaseException) -> bool:
    if _matches(error, ("request too large", "reduce your message size")):
        return False  # Groq reports an oversized prompt as a rate limit; waiting won't help
    return type(error).__name__ == "RateLimitError" or _matches(error, ("rate limit", "rate_limit", "too many requests"))


@dataclass
class ModelStats:
    ok: int = 0
    failed: int = 0
    consecutive_failures: int = 0
    skipped_because: str | None = None
    quota_used_up: bool = False
    last_error: str = ""


STATS: dict[str, ModelStats] = {}
NOTES: dict[str, int] = {}  # counters for the report, e.g. keyless searches used
MODEL_GATES: dict[str, ProviderGate] = {}  # model -> its gate, for the report


def note(key: str) -> None:
    NOTES[key] = NOTES.get(key, 0) + 1


def free_model_report() -> str:
    lines = ["=" * 30 + " Free-model report " + "=" * 30]
    if not STATS:
        lines.append("No model was called in this run.")
    for model, st in sorted(STATS.items()):
        line = f"{model:<36} ok {st.ok:>3}   failed {st.failed:>3}"
        gate = MODEL_GATES.get(model)
        if st.skipped_because:
            label = "DAILY QUOTA USED UP" if st.quota_used_up else "SKIPPED"
            line += f"   {label}: {st.skipped_because}"
        elif gate is not None and gate.disabled_reason:
            line += f"   SKIPPED ({gate.name}): {gate.disabled_reason}"
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


class NoFreeModelLeft(RuntimeError):
    """No model of a mix can take a prompt any more in this run (daily quotas
    used up, models skipped after repeated failures, or prompt too long)."""


class _Skipped(RuntimeError):
    pass


class FreeTierLlm(GeneralLlm):
    """A GeneralLlm that waits for its gate, refuses prompts that are too long
    for a free tier, retries passing errors itself (tries), and is skipped for
    the rest of the run after errors that won't go away (daily quota used up,
    model retired, bad key, repeated failures)."""

    def __init__(self, model: str, gate: ProviderGate, max_input_tokens: int | None = None, tries: int = 1,
                 **kwargs: Any) -> None:
        kwargs.pop("allowed_tries", None)
        # forecasting-tools' own retry would also retry a model that was just
        # skipped, so retries happen in _mockable_direct_call_to_model instead
        super().__init__(model=model, allowed_tries=1, **kwargs)
        self._gate = gate
        self._max_input_tokens = max_input_tokens
        self._tries = max(1, tries)
        MODEL_GATES[model] = gate

    @property
    def stats(self) -> ModelStats:
        return STATS.setdefault(self.model, ModelStats())

    def unavailable_reason(self) -> str | None:
        return self._gate.disabled_reason or self.stats.skipped_because

    def can_take(self, prompt: Any) -> bool:
        if self.unavailable_reason():
            return False
        return self._max_input_tokens is None or estimate_tokens(prompt) <= self._max_input_tokens

    def _record_failure(self, e: Exception) -> None:
        st = self.stats
        st.failed += 1
        st.consecutive_failures += 1
        note_text = redact(f"{type(e).__name__}: {' '.join(str(e).split())[:400]}")
        st.last_error = note_text
        gate = self._gate
        # model-level checks first: "no access to this model" is about one model
        if _matches(e, _QUOTA_ERRORS):
            st.skipped_because = note_text
            st.quota_used_up = True
            logger.warning(f"{self.model}: {note_text} - daily quota used up, skipping this model for the rest of the run")
        elif _matches(e, _GONE_ERRORS):
            st.skipped_because = note_text
            logger.warning(f"{self.model}: {note_text} - skipping this model for the rest of the run")
        elif _matches(e, _PROVIDER_ENDING_ERRORS):
            gate.disabled_reason = note_text
            logger.warning(f"{gate.name}: {note_text} - skipping this provider for the rest of the run")
        elif _is_rate_limit(e):
            gate.rate_limits_in_a_row += 1
            if gate.rate_limits_in_a_row >= MAX_CONSECUTIVE_RATE_LIMITS:
                gate.disabled_reason = f"{gate.rate_limits_in_a_row} rate-limit errors in a row; last: {note_text}"
                logger.warning(f"{gate.name}: {gate.disabled_reason} - skipping this provider for the rest of the run")
            else:
                logger.warning(f"{self.model}: {note_text}")
                gate.slow_down()
        elif st.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            st.skipped_because = f"{st.consecutive_failures} failures in a row; last: {note_text}"
            logger.warning(f"{self.model}: {st.skipped_because} - skipping this model for the rest of the run")
        else:
            logger.warning(f"{self.model}: {note_text}")

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        for attempt in range(1, self._tries + 1):
            reason = self.unavailable_reason()
            if reason:
                raise _Skipped(f"{self.model} skipped for this run: {reason}")
            try:
                async with self._gate.slot():
                    reason = self.unavailable_reason()  # may have changed while waiting for the slot
                    if reason:
                        raise _Skipped(f"{self.model} skipped for this run: {reason}")
                    result = await super()._mockable_direct_call_to_model(prompt)
            except _Skipped:
                raise
            except Exception as e:
                self._record_failure(e)
                if attempt == self._tries or self.unavailable_reason():
                    raise
                await asyncio.sleep(RETRY_WAIT * attempt)
                continue
            self.stats.ok += 1
            self.stats.consecutive_failures = 0
            self._gate.rate_limits_in_a_row = 0
            return result
        raise AssertionError("unreachable")


class FreeModelMix(GeneralLlm):
    """Tries several free models in order. With several orders ("slots"),
    successive calls use successive orders, so the forecasts for a question
    start with different models (an ensemble) and each falls back to the
    others. When every model fails with a passing error (rate limit,
    overload), it waits and tries again, up to len(waits) more rounds."""

    def __init__(self, name: str, orders: list[list[FreeTierLlm]], waits: tuple[float, ...] = (30.0, 90.0)) -> None:
        orders = [order for order in orders if order]
        if not orders:
            raise ValueError("FreeModelMix needs at least one model")
        super().__init__(model=name, allowed_tries=1)
        self.orders = orders
        self.waits = waits
        self._next = 0
        seen: dict[str, FreeTierLlm] = {}
        for order in orders:
            for m in order:
                seen.setdefault(m.model, m)
        self.members = list(seen.values())

    def restart_slots(self) -> None:
        """Called at the start of each question, so its first forecast uses slot 1."""
        self._next = 0

    def available(self) -> bool:
        return any(m.can_take("") for m in self.members)

    def limits_reached(self) -> bool:
        """True when the free tiers, not a bug, stopped the forecasts: every
        model is out for this run or was refused on its last call for a rate
        limit, and at least one used up its daily quota or hit a rate limit."""
        def rate_limited(m: FreeTierLlm) -> bool:
            return m.stats.consecutive_failures > 0 and m.stats.last_error.startswith("RateLimitError")

        return all(not m.can_take("") or rate_limited(m) for m in self.members) and any(
            m.stats.quota_used_up or rate_limited(m) for m in self.members
        )

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        order = self.orders[self._next % len(self.orders)]
        self._next += 1
        errors: list[str] = []
        for round_no in range(len(self.waits) + 1):
            tried = 0
            for member in order:
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
        skipped = [m.model for m in order if not m.can_take(prompt)]
        detail = "; ".join(errors[-8:]) or "no model could take this prompt"
        if skipped:
            detail += f" (skipped: {', '.join(skipped)})"
        error_type = RuntimeError if errors else NoFreeModelLeft
        raise error_type(f"All free models failed for {self.model}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_model": self.model,
            "mode": f"{len(self.orders)} starting points" if len(self.orders) > 1 else "fallback",
            "models": [m.model for m in self.members],
        }


# ---------------------------------------------------------------- the lineup
MISTRAL = ProviderGate("mistral", min_interval=1.2, max_concurrent=1)
# Groq limits are per model; 8,000 tokens/minute ~ one big request per minute each
GROQ_120B = ProviderGate("groq-gpt-oss-120b", min_interval=62.0, max_concurrent=1)
GROQ_20B = ProviderGate("groq-gpt-oss-20b", min_interval=62.0, max_concurrent=1)
# Gemini limits are per model as well
GEMINI_FLASH = ProviderGate("gemini-3.8-flash", min_interval=13.0, max_concurrent=1)
GEMINI_LITE_35 = ProviderGate("gemini-3.5-flash-lite", min_interval=6.5, max_concurrent=1)
GEMINI_LITE_31 = ProviderGate("gemini-3.1-flash-lite", min_interval=6.5, max_concurrent=1)
GATES = [MISTRAL, GROQ_120B, GROQ_20B, GEMINI_FLASH, GEMINI_LITE_35, GEMINI_LITE_31]

# Where each of the three forecasts for a question starts, and where it goes
# next when a model is unavailable: Gemini first, Groq first, Mistral first.
FORECAST_ORDERS = [
    ["flash", "lite35", "120b", "medium", "lite31", "20b", "magistral"],
    ["120b", "20b", "medium", "flash", "lite35", "magistral", "lite31"],
    ["medium", "magistral", "lite31", "20b", "lite35", "120b", "flash"],
]


def build_free_llms(predictions_per_question: int = 3) -> tuple[dict[str, Any] | None, list[str]]:
    """Returns (llms for the bot, providers found). llms is None when there is
    no usable key at all."""
    mistral = has_key("MISTRAL_API_KEY")
    groq = has_key("GROQ_API_KEY")
    gemini = has_key("GEMINI_API_KEY")
    providers = [p for p, ok in (("Gemini", gemini), ("Groq", groq), ("Mistral", mistral)) if ok]
    if not providers:
        return None, []

    def m(model: str, temperature: float = 0.3, timeout: int = 240) -> FreeTierLlm:
        return FreeTierLlm(f"mistral/{model}", MISTRAL, temperature=temperature, timeout=timeout, tries=2)

    def gq(model: str, gate: ProviderGate, **kw: Any) -> FreeTierLlm:
        # prompt + max_tokens must stay under 8,000 tokens/minute
        return FreeTierLlm(f"groq/{model}", gate, max_input_tokens=5000, temperature=kw.pop("temperature", 0.3),
                           timeout=180, tries=1, max_tokens=kw.pop("max_tokens", 2500), **kw)

    def gm(model: str, gate: ProviderGate, timeout: int = 180) -> FreeTierLlm:
        # temperature left at Gemini 3's default (1.0): Google warns that lower
        # values can cause looping or worse reasoning
        return FreeTierLlm(f"gemini/{model}", gate, timeout=timeout, tries=2)

    makers: dict[str, Any] = {}
    if gemini:
        makers.update(
            flash=lambda: gm("gemini-3.8-flash", GEMINI_FLASH, timeout=300),
            lite35=lambda: gm("gemini-3.5-flash-lite", GEMINI_LITE_35),
            lite31=lambda: gm("gemini-3.1-flash-lite", GEMINI_LITE_31),
        )
    if groq:
        makers.update({
            "120b": lambda: gq("openai/gpt-oss-120b", GROQ_120B, reasoning_effort="low"),
            "20b": lambda: gq("openai/gpt-oss-20b", GROQ_20B, reasoning_effort="low"),
        })
    if mistral:
        makers.update(
            medium=lambda: m("mistral-medium-latest"),
            magistral=lambda: m("magistral-medium-latest", timeout=300),
            small=lambda: m("mistral-small-latest"),
        )
    made: dict[str, FreeTierLlm] = {}

    def pick(names: list[str]) -> list[FreeTierLlm]:
        # one object per model, shared by every order it appears in
        out = []
        for name in names:
            if name in makers:
                if name not in made:
                    made[name] = makers[name]()
                out.append(made[name])
        return out

    slots = max(1, predictions_per_question)
    forecast_orders = [pick(FORECAST_ORDERS[i % len(FORECAST_ORDERS)]) for i in range(min(slots, len(FORECAST_ORDERS)))]

    # research plans and fact sheets: long prompts, so no Groq (its token
    # budget is kept for forecasts and the web search); without Gemini or
    # Mistral the bot uses the simpler research instead
    writers = pick(["lite35", "lite31", "medium", "small"])

    # only used when an answer can't be read directly from the forecast text
    parsers = pick(["lite35", "lite31"])
    if mistral:
        parsers.append(m("mistral-small-latest", temperature=0))
    if groq:
        parsers.append(gq("openai/gpt-oss-20b", GROQ_20B, temperature=0, reasoning_effort="low", max_tokens=2000))

    llms: dict[str, Any] = {
        "default": FreeModelMix("free-forecasters", forecast_orders),
        "writer": FreeModelMix("free-writer", [writers]) if writers else None,
        "summarizer": FreeModelMix("free-summarizer", [parsers]),
        "parser": FreeModelMix("free-parser", [parsers], waits=(20.0,)),
        "researcher": "free-news",  # keyless GDELT headlines per claim (main.py)
        "web": None,
    }
    if groq:  # one web search per question with Groq's browser_search tool (gpt-oss-20b only)
        llms["web"] = FreeModelMix(
            "free-web-search",
            [[gq("openai/gpt-oss-20b", GROQ_20B, tools=[{"type": "browser_search"}], tool_choice="required",
                 reasoning_effort="low", temperature=0.2, max_tokens=1800)]],
            waits=(),
        )
    return llms, providers


# ---------------------------------------------------------------- reading answers without a model
# A wrong number read here would be published as the forecast, so the readers
# are strict: they only use the final answer block at the end of the text and
# return None when in doubt; the parser model then reads the answer instead.
_PCT = r"([0-9]+(?:[.,][0-9]+)?)\s*%"
_NEGATION = re.compile(r"\b(no|not|non|never|neither|against|fail\w*|won't|wont|doesn't|doesnt|isn't|isnt)\b", re.IGNORECASE)


def read_binary(text: str) -> float | None:
    """The last 'Probability: NN%' near the end of the text, as a fraction.
    None if that last one is about the opposite outcome ('probability that it
    does not pass: 70%')."""
    matches = list(re.finditer(r"probability(\s+(?:of|that)\s+[a-z' ]{1,25})?\W{0,6}" + _PCT, text[-800:],
                               flags=re.IGNORECASE))
    if not matches:
        return None
    last = matches[-1]
    if _NEGATION.search(last.group(1) or ""):
        return None
    value = float(last.group(2).replace(",", "."))
    return value / 100 if 0 <= value <= 100 else None


def _final_block(lines: list[str], read_line, keys: list, max_gap: int = 2) -> tuple[dict, int] | None:
    """Values from the last block of answer lines (other lines in between: at
    most max_gap in a row) that covers every key. Returns (values, index of the
    block's last line), or None if the final block misses a key."""
    hits = []
    for i, line in enumerate(lines):
        found = read_line(line)
        if found is not None:
            hits.append((i, found[0], found[1]))
    if not hits:
        return None
    values: dict = {}
    previous = hits[-1][0]
    for i, key, value in reversed(hits):
        if previous - i - 1 > max_gap:
            break  # an earlier block (e.g. in the reasoning) doesn't count
        values.setdefault(key, value)  # going backwards: the later line wins
        previous = i
        if all(k in values for k in keys):
            break
    if not all(k in values for k in keys):
        return None
    return values, hits[-1][0]


def _normalised(found: dict[str, float]) -> dict[str, float] | None:
    total = sum(found.values())
    if not 80 <= total <= 120:
        return None  # doesn't look like a final answer
    return {k: v / total for k, v in found.items()}


def _option_blocks(text: str, options: list[str]) -> list[tuple[dict, int]]:
    lines = text[-2500:].splitlines()
    blocks = []
    # "Red: 20%" lines; longer names first, so "Other" can't claim "Other party" lines
    by_length = sorted(options, key=lambda o: -len(o.strip()))
    named = [(opt, re.compile(r"^[\s\-\*•#>\"'“]*(?:option[\s_]*[a-z0-9]?[\s:)\-–.]*)?[\"'“*]*"
                              + re.escape(opt.strip()) + r"[\"'”*]*\s*[:=\-–]\s*\**\s*" + _PCT, re.IGNORECASE))
             for opt in by_length]

    def read_named(line: str):
        for opt, pattern in named:
            m = pattern.match(line)
            if m:
                return opt, float(m.group(1).replace(",", "."))
        return None

    block = _final_block(lines, read_named, list(options))
    if block:
        blocks.append(block)
    if 0 < len(options) <= 26:
        # the prompt's own pattern taken literally: "Option_A: 20%" in the order given
        lettered = re.compile(r"^[\s\-\*•#>]*\**option[\s_]*([a-z])\b[^:\n]{0,40}[:=]\s*\**\s*" + _PCT, re.IGNORECASE)

        def read_lettered(line: str):
            m = lettered.match(line)
            if not m:
                return None
            index = ord(m.group(1).upper()) - ord("A")
            if index >= len(options):
                return None
            return options[index], float(m.group(2).replace(",", "."))

        block = _final_block(lines, read_lettered, list(options))
        if block:
            blocks.append(block)
    return blocks


def read_options(text: str, options: list[str]) -> dict[str, float] | None:
    """The final 'Red: 20%' (or 'Option_A: 20%') block for all options,
    normalised to sum to 1. If both styles appear, the later block counts.
    None if any option is missing or the numbers don't look like a final answer."""
    blocks = _option_blocks(text, options)
    if not blocks:
        return None
    values, _ = max(blocks, key=lambda b: b[1])
    return _normalised({opt: values[opt] for opt in options})


def read_lettered_options(text: str, options: list[str]) -> dict[str, float] | None:
    """Kept for compatibility: read_options handles both styles."""
    return read_options(text, options)


_PERCENTILE_KEYS = [10, 20, 40, 60, 80, 90]
_PERCENTILE_LINE = r"^[\s\-\*•\"']*percentile\s*(10|20|40|60|80|90)\s*[:=]\s*\**\s*"


def read_percentiles(text: str) -> dict[int, float] | None:
    """The final 'Percentile NN: value' block for 10/20/40/60/80/90, if all
    six are there, increasing, and without words like 'million' (the parser
    model converts those)."""
    lines = text[-1500:].splitlines()
    pattern = re.compile(_PERCENTILE_LINE + r"\$?\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)(.{0,14})",
                         re.IGNORECASE)
    magnitude = re.compile(r"\s*(thousand|million|billion|trillion|bn\b|mn\b|k\b|m\b|b\b)", re.IGNORECASE)
    trouble: list[bool] = []

    def read_line(line: str):
        m = pattern.match(line)
        if not m:
            return None
        if magnitude.match(m.group(3)):
            trouble.append(True)
        try:
            return int(m.group(1)), float(m.group(2).replace(",", ""))
        except ValueError:
            trouble.append(True)
            return None

    block = _final_block(lines, read_line, _PERCENTILE_KEYS)
    if block is None or trouble:
        return None
    found = block[0]
    values = [found[k] for k in _PERCENTILE_KEYS]
    if any(b < a for a, b in zip(values, values[1:])):
        return None
    return found


def read_date_percentiles(text: str) -> dict[int, datetime] | None:
    """The final 'Percentile NN: YYYY-MM-DD' (optionally with THH:MM[:SS]Z)
    block for 10/20/40/60/80/90, if all six are there and in order. UTC."""
    lines = text[-1500:].splitlines()
    pattern = re.compile(_PERCENTILE_LINE + r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?)\s*(?:Z|UTC)?)?",
                         re.IGNORECASE)

    def read_line(line: str):
        m = pattern.match(line)
        if not m:
            return None
        try:
            moment = datetime.fromisoformat(f"{m.group(2)}T{m.group(3) or '00:00:00'}")
        except ValueError:
            return None
        return int(m.group(1)), moment.replace(tzinfo=timezone.utc)

    block = _final_block(lines, read_line, _PERCENTILE_KEYS)
    if block is None:
        return None
    found = block[0]
    values = [found[k] for k in _PERCENTILE_KEYS]
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


def gdelt_news(query: str, language: str | None = None, limit: int = 8, keywords: int = 4) -> list[dict]:
    words = _keywords(query, keywords)
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
    """Recent headlines from GDELT (from sources in the requested language when
    given, otherwise any language). Needs no key. If four keywords find
    nothing, three are tried. Returns plain text for the fact sheet."""
    query = " ".join(str(query).split())[:200]
    lang = language if language and language.strip().lower() != "english" else None
    items: list[dict] = []
    for keywords in (4, 3):
        try:
            async with GDELT.slot():
                items = await asyncio.to_thread(gdelt_news, query, lang, limit, keywords)
        except Exception as e:
            logger.info(f"GDELT search failed for '{query}' ({lang or 'any language'}): {type(e).__name__}: {str(e)[:120]}")
            items = []
        if items or len(_keywords(query, 5)) <= keywords - 1:
            break  # found something, or fewer keywords wouldn't change the query
    note("GDELT searches with results" if items else "GDELT searches without results")
    label = f"{lang} sources" if lang else "any language"
    if not items:
        return f"[keyless news search] No recent headlines found for: {query}"
    lines = [f"[keyless news search via GDELT, {label}; headlines only] Query: {query}"]
    for it in items:
        day = it["date"].strftime("%Y-%m-%d") if it["date"] else "date unknown"
        source = f" ({it['source']})" if it["source"] else ""
        lines.append(f"- {day}{source}: {it['title']} <{it['link']}>")
    return "\n".join(lines)
