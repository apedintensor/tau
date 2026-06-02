from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

log = logging.getLogger(__name__)

class GitHubClient(ABC):
    @abstractmethod
    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def put(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def patch(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def delete(self, url: str, **kwargs: Any) -> httpx.Response: ...


class GitHubAuthRotatingClient(GitHubClient):
    """Small GitHub client wrapper with token rotation and 401 blacklisting."""

    def __init__(
        self,
        *,
        base_headers: dict[str, str],
        timeout: float,
        tokens: Sequence[str],
        rotate: bool,
        user_agent: str,
    ) -> None:
        self._client = httpx.Client(
            base_url="https://api.github.com",
            headers=base_headers,
            follow_redirects=True,
            timeout=timeout,
        )
        self._tokens = _dedupe_preserve_order([token for token in tokens if token])
        self._rotate = rotate
        self._user_agent = user_agent
        self._lock = threading.Lock()
        self._next_index = 0
        self._disabled_indexes: set[int] = set()
        self._all_tokens_disabled_logged = False

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        attempts = self._token_attempts()
        last_response: httpx.Response | None = None
        for token_index, token in attempts:
            response = self._client.request(
                method,
                url,
                **self._request_kwargs_with_token(kwargs, token),
            )
            last_response = response
            if response.status_code == 401 and token_index is not None:
                self._disable_token(token_index)
                continue
            if token_index is not None and self._rotate:
                self._mark_success(token_index)
            return response
        if last_response is None:
            raise RuntimeError("GitHub client made no request")
        return last_response

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def github_cache_namespace(self) -> str:
        attempts = self._token_attempts()
        token_index, token = attempts[0]
        if token_index is None or not token:
            return f"{self._user_agent}:unauthenticated"
        return f"{self._user_agent}:token:{_token_fingerprint(token)}"

    def _token_attempts(self) -> list[tuple[int | None, str | None]]:
        with self._lock:
            active_indexes = [idx for idx in range(len(self._tokens)) if idx not in self._disabled_indexes]
            if not active_indexes:
                if self._tokens and not self._all_tokens_disabled_logged:
                    self._all_tokens_disabled_logged = True
                    log.error(
                        "GitHub client %s exhausted all configured auth tokens after HTTP 401 responses; "
                        "falling back to unauthenticated requests",
                        self._user_agent,
                    )
                return [(None, None)]
            self._all_tokens_disabled_logged = False
            if not self._rotate:
                return [(idx, self._tokens[idx]) for idx in active_indexes]
            attempts: list[tuple[int | None, str | None]] = []
            for offset in range(len(self._tokens)):
                idx = (self._next_index + offset) % len(self._tokens)
                if idx in self._disabled_indexes:
                    continue
                attempts.append((idx, self._tokens[idx]))
            return attempts

    def _request_kwargs_with_token(self, kwargs: dict[str, Any], token: str | None) -> dict[str, Any]:
        request_kwargs = dict(kwargs)
        headers = dict(request_kwargs.get("headers") or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            headers.pop("Authorization", None)
        request_kwargs["headers"] = headers
        return request_kwargs

    def _disable_token(self, token_index: int) -> None:
        with self._lock:
            if token_index in self._disabled_indexes:
                return
            self._disabled_indexes.add(token_index)
            remaining = len(self._tokens) - len(self._disabled_indexes)
            fingerprint = _token_fingerprint(self._tokens[token_index])
        log.warning(
            "GitHub client %s permanently blacklisted token #%d (%s) after HTTP 401; %d token(s) remain",
            self._user_agent,
            token_index + 1,
            fingerprint,
            remaining,
        )

    def _mark_success(self, token_index: int) -> None:
        with self._lock:
            self._next_index = (token_index + 1) % max(1, len(self._tokens))


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]
