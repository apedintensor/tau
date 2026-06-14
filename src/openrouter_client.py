from __future__ import annotations

import logging
import os
from typing import Any

import httpx

import tau.utils
from tau.io.chat_completion import assistant_text_from_payload, empty_content_error
from tau.io.openrouter import CachedLLMClient, LLMClient, LLMRequest, normalize_base_url

log = logging.getLogger("swe-eval.openrouter_client")

_DEFAULT_MODEL = "google/gemini-3.1-flash-lite"


class HttpxLLMClient(LLMClient):
    """Calls OpenRouter directly via httpx."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete_text(self, request: LLMRequest, *, timeout: int) -> str:
        payload: dict[str, Any] = {
            "model": _resolve_model(request.model),
            "messages": _build_messages(system_prompt=request.system_prompt, prompt=request.prompt),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.reasoning is not None:
            payload["reasoning"] = request.reasoning
        if request.cache_control is not None:
            payload["cache_control"] = request.cache_control
        provider = _provider_preferences_from_env()
        if provider is not None:
            payload["provider"] = provider
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "swe-eval",
        }
        log.debug("Calling OpenRouter model=%s timeout=%ss", payload["model"], timeout)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(_openrouter_url(), headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(_no_choices_error(data))
        text = assistant_text_from_payload(data)
        if not text.strip():
            raise RuntimeError(empty_content_error(data))
        return text


def _build_client(api_key: str) -> LLMClient:
    """Build an LLMClient based on environment variables.

    LLM_REPLAY_DIR  — replay-only mode: serve from cache, raise CacheMissError on miss.
    LLM_CACHE_DIR   — record+replay mode: delegate to OpenRouter on miss and save the result.
    (unset)         — call OpenRouter directly with no caching.
    """
    from pathlib import Path

    replay_dir = os.environ.get("LLM_REPLAY_DIR")
    if replay_dir:
        return CachedLLMClient(Path(replay_dir), inner=None)
    cache_dir = os.environ.get("LLM_CACHE_DIR")
    if cache_dir:
        return CachedLLMClient(Path(cache_dir), inner=HttpxLLMClient(api_key))
    return HttpxLLMClient(api_key)


def complete_text(
    *,
    prompt: str | list[dict[str, Any]],
    model: str | None,
    timeout: int,
    openrouter_api_key: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    reasoning: dict[str, Any] | None = None,
    cache_control: dict[str, Any] | None = None,
) -> str:
    return _build_client(openrouter_api_key).complete_text(
        LLMRequest(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning=reasoning,
            cache_control=cache_control,
        ),
        timeout=timeout,
    )


def _openrouter_url() -> str:
    return (
        normalize_base_url(
            os.environ.get("OPENROUTER_UPSTREAM_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL"),
        )
        + "/v1/chat/completions"
    )


def _resolve_model(model: str | None) -> str:
    if not model:
        return _DEFAULT_MODEL
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model


def _provider_preferences_from_env() -> dict[str, Any] | None:
    only_raw = os.environ.get("OPENROUTER_PROVIDER_ONLY") or os.environ.get("SOLVER_PROVIDER_ONLY")
    only = [part.strip() for part in (only_raw or "").split(",") if part.strip()]
    allow_fallbacks_raw = os.environ.get("OPENROUTER_PROVIDER_ALLOW_FALLBACKS")
    if allow_fallbacks_raw is None:
        allow_fallbacks_raw = os.environ.get("SOLVER_PROVIDER_ALLOW_FALLBACKS")
    provider: dict[str, Any] = {}
    if only:
        provider["only"] = only
    if allow_fallbacks_raw is not None:
        provider["allow_fallbacks"] = allow_fallbacks_raw.strip().lower() in {"1", "true", "yes", "on"}
    return provider or None


def _build_messages(
    *,
    system_prompt: str | None,
    prompt: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _no_choices_error(data: dict[str, Any]) -> str:
    error = tau.utils.get_dict(data, "error")
    return (
        "OpenRouter returned no choices "
        f"(error_code={error.get('code')!r}, "
        f"error_message={_truncate_error_text(error.get('message'))!r}, "
        f"response_keys={sorted(data.keys())})"
    )


def _truncate_error_text(raw: Any, limit: int = 240) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
