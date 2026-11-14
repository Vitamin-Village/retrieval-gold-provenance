"""Thin LLM backend adapters for the 4-model cross-judge consensus.

Inputs : system + user strings, max_tokens (caller supplied).
Outputs: plain-text completion string (never raises on non-empty response).
Decisions: D32 (4-model cross-family), D38 (W2 auto-label), D38b (Fleiss kappa gate).

DO NOT EDIT without reading paper/MASTER_PLAN.md D33-D41.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

# ASCII-only error / log strings (Windows cp949 console safety).

# ---------------------------------------------------------------------------
# Load API keys from project .env BEFORE any backend reads os.environ.
# ANTHROPIC_API_KEY / GEMINI_API_KEY live in the project .env file rather
# than as system env vars on this host. python-dotenv is optional; if it is
# not installed we fall back to whatever is in the system env (and the
# backend constructors will raise MissingApiKey if nothing is set).
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # fall back to system env; MissingApiKey raised on demand

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingApiKey(RuntimeError):
    """Raised when a backend cannot find its required API key in env."""


class BackendFailure(RuntimeError):
    """Raised after a backend fails twice in a row (network / SDK error)."""


# ---------------------------------------------------------------------------
# Retry helper: one retry on transient error with 2s delay.
# ---------------------------------------------------------------------------


def _call_with_one_retry(fn, backend_name: str, delay_s: float = 2.0) -> str:
    """Run fn(); on Exception sleep delay_s and try once more.

    Returns a non-empty string on success. Raises BackendFailure otherwise.
    """
    try:
        out = fn()
    except Exception as first_err:  # noqa: BLE001 - we re-raise after retry
        time.sleep(delay_s)
        try:
            out = fn()
        except Exception as second_err:  # noqa: BLE001
            raise BackendFailure(
                "backend {0} failed twice: first={1} second={2}".format(
                    backend_name,
                    type(first_err).__name__,
                    type(second_err).__name__,
                )
            ) from second_err
    if out is None:
        raise BackendFailure(
            "backend {0} returned None on success path".format(backend_name)
        )
    return out


# ---------------------------------------------------------------------------
# Claude Haiku backend
# ---------------------------------------------------------------------------


class ClaudeHaikuBackend:
    """Anthropic Claude Haiku 4.5 adapter (deterministic, temperature=0)."""

    name = "claude-haiku"
    model_id = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: Optional[str] = None, client=None) -> None:
        # client param is for tests: lets callers inject a stub.
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if client is None:
            if not key:
                raise MissingApiKey("ANTHROPIC_API_KEY not set for ClaudeHaikuBackend")
            try:
                import anthropic  # local import: optional dep at call time
            except ImportError as e:
                raise BackendFailure(
                    "anthropic package missing for ClaudeHaikuBackend"
                ) from e
            self._client = anthropic.Anthropic(api_key=key)
        else:
            self._client = client

    def chat(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        def _call() -> str:
            resp = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            # anthropic SDK: resp.content is list of content blocks
            parts = []
            for block in getattr(resp, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            text_out = "".join(parts).strip()
            if not text_out:
                raise BackendFailure("claude-haiku empty response")
            return text_out

        return _call_with_one_retry(_call, self.name)


# ---------------------------------------------------------------------------
# Claude Sonnet backend (independent-benchmark GENERATOR, D-A 2026-05-29).
# Distinct from ClaudeHaikuBackend so the gold-build pipeline can inject a
# stronger generator while the verifier panel stays cross-family (OpenAI).
# ---------------------------------------------------------------------------

# Model id pinned as a module constant for easy swap. As of 2026-05 the
# current Sonnet snapshot is claude-sonnet-4-5; bump this string to retarget.
CLAUDE_SONNET_MODEL_ID = "claude-sonnet-4-5-20250929"


class ClaudeSonnetBackend:
    """Anthropic Claude Sonnet adapter (deterministic, temperature=0).

    Used as the QA GENERATOR for the independent benchmark (design D-A:
    generation = Sonnet, verification = OpenAI). Same interface and retry
    semantics as ClaudeHaikuBackend; only model_id differs.
    """

    name = "claude-sonnet"
    model_id = CLAUDE_SONNET_MODEL_ID

    def __init__(self, api_key: Optional[str] = None, client=None) -> None:
        # client param is for tests: lets callers inject a stub.
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if client is None:
            if not key:
                raise MissingApiKey(
                    "ANTHROPIC_API_KEY not set for ClaudeSonnetBackend"
                )
            try:
                import anthropic  # local import: optional dep at call time
            except ImportError as e:
                raise BackendFailure(
                    "anthropic package missing for ClaudeSonnetBackend"
                ) from e
            self._client = anthropic.Anthropic(api_key=key)
        else:
            self._client = client

    def chat(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        def _call() -> str:
            resp = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = []
            for block in getattr(resp, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            text_out = "".join(parts).strip()
            if not text_out:
                raise BackendFailure("claude-sonnet empty response")
            return text_out

        return _call_with_one_retry(_call, self.name)


# ---------------------------------------------------------------------------
# OpenAI GPT backend (independent-benchmark VERIFIER, D-A 2026-05-29).
# Cross-family with the Sonnet generator so the verification panel cannot
# rubber-stamp its own generations.
# ---------------------------------------------------------------------------

# Model id pinned as a module constant for easy swap. As of 2026-05 a
# reasonable current GPT chat model is gpt-4o; bump this string to retarget
# (e.g. a newer GPT snapshot) without touching call sites.
OPENAI_MODEL_ID = "gpt-4o"


class OpenAIBackend:
    """OpenAI GPT chat-completions adapter (deterministic, temperature=0).

    Used as the answer-paper VERIFIER for the independent benchmark. Same
    interface and one-retry semantics as the Claude backends.
    """

    name = "openai-gpt"
    model_id = OPENAI_MODEL_ID

    def __init__(self, api_key: Optional[str] = None, client=None) -> None:
        # client param is for tests: lets callers inject a stub.
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if client is None:
            if not key:
                raise MissingApiKey("OPENAI_API_KEY not set for OpenAIBackend")
            try:
                import openai  # local import: optional dep at call time
            except ImportError as e:
                raise BackendFailure(
                    "openai package missing for OpenAIBackend"
                ) from e
            self._client = openai.OpenAI(api_key=key)
        else:
            self._client = client

    def chat(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            # openai SDK: resp.choices[0].message.content is the text.
            text_out = ""
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                text_out = (getattr(msg, "content", "") or "").strip()
            if not text_out:
                raise BackendFailure("openai-gpt empty response")
            return text_out

        return _call_with_one_retry(_call, self.name)


# ---------------------------------------------------------------------------
# Gemini 2.0 Flash backend
# ---------------------------------------------------------------------------


class GeminiFlashBackend:
    """Google Gemini Flash adapter (deterministic, temperature=0).

    Note: gemini-2.0-flash is deprecated for new keys; gemini-2.5-flash is
    the working substitute (D32 technical amendment 2026-05-23).
    """

    name = "gemini-flash"
    model_id = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, client=None) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if client is None:
            if not key:
                raise MissingApiKey("GEMINI_API_KEY not set for GeminiFlashBackend")
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise BackendFailure(
                    "google.generativeai missing for GeminiFlashBackend"
                ) from e
            genai.configure(api_key=key)
            self._client = genai.GenerativeModel(self.model_id)
            self._genai = genai
        else:
            self._client = client
            self._genai = None

    def chat(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        def _call() -> str:
            # google-generativeai supports system_instruction kwarg via the
            # model constructor in newer versions; for portability we prepend.
            combined = system.rstrip() + "\n\n" + user
            generation_config = {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": max_tokens,
            }
            resp = self._client.generate_content(
                combined,
                generation_config=generation_config,
            )
            text_out = (getattr(resp, "text", "") or "").strip()
            if not text_out:
                raise BackendFailure("gemini-flash empty response")
            return text_out

        return _call_with_one_retry(_call, self.name)


# ---------------------------------------------------------------------------
# Ollama backend (local Llama 3.1:8b / Mistral:7b)
# ---------------------------------------------------------------------------


class OllamaBackend:
    """Ollama local-model adapter (deterministic, temperature=0, seed=0)."""

    def __init__(self, model_tag: str, client=None) -> None:
        self.model_tag = model_tag
        # Public ASCII name = the sanitized model tag, so provenance recorded in
        # the dataset card is always the ACTUAL model. (The old hard-coded
        # "mistral-7b" branch mislabelled mistral-small:24b as 7b.) Backward
        # compatible: "mistral:7b"->"mistral-7b", "llama3.1:8b"->"llama3.1-8b".
        self.name = model_tag.replace(":", "-")
        if client is None:
            try:
                import ollama  # local import: optional dep at call time
            except ImportError as e:
                raise BackendFailure(
                    "ollama package missing for OllamaBackend"
                ) from e
            self._ollama = ollama
            self._client = None
        else:
            self._ollama = None
            self._client = client

    def chat(self, system: str, user: str, *, max_tokens: int = 256) -> str:
        def _call() -> str:
            options = {
                "temperature": 0,
                "seed": 0,
                "num_predict": max_tokens,
            }
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if self._client is not None:
                resp = self._client.chat(
                    model=self.model_tag, messages=messages, options=options
                )
            else:
                resp = self._ollama.chat(
                    model=self.model_tag, messages=messages, options=options
                )
            # ollama returns {"message": {"content": "..."}}
            try:
                text_out = resp["message"]["content"].strip()
            except (KeyError, TypeError, AttributeError) as e:
                raise BackendFailure(
                    "ollama bad response shape: {0}".format(type(e).__name__)
                ) from e
            if not text_out:
                raise BackendFailure("ollama empty response")
            return text_out

        return _call_with_one_retry(_call, self.name)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_default_backends() -> list:
    """Build the 2-model cross-cloud default panel: Claude + Gemini.

    Ollama local models removed per W2 evidence (D42, 2026-05-23): on the W2
    step6 v3 run Llama-3.1-8B abstained on 91.1% of relation tasks and
    Mistral-7B on 78.6%, while Claude Haiku and Gemini Flash both abstained
    near 20%. Under the n_agree>=3 rule that asymmetric abstention forced
    ~50% of tasks below threshold despite genuine agreement among the
    remaining votes, so the 4-model lock was empirically harmful and is not
    used.

    D42 (2-model cross-cloud) replaces D32 (4-model cross-family).
    consensus_min in auto_label_pipeline is correspondingly lowered to 2
    (unanimous-of-2) and the IAA metric switches from Fleiss kappa to
    Cohen's kappa for the 2-rater case (Fleiss degenerates with N=2).

    Caller must have ANTHROPIC_API_KEY and GEMINI_API_KEY set. Used by
    runtime drivers, not by unit tests (tests inject stubs).
    """
    return [
        ClaudeHaikuBackend(),
        GeminiFlashBackend(),
    ]
