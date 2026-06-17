import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from openrouter_proxy import (
    REQUEST_LIMIT_EXIT_REASON,
    TOKEN_LIMIT_EXIT_REASON,
    CachedUpstreamClient,
    HttpxUpstreamClient,
    OpenRouterProxy,
    ProxyRequestRecord,
    SolveBudget,
    UpstreamResponse,
)
from sampling_seed import VALIDATOR_TOP_P
from tau.io.openrouter import CacheMissError
from tau.utils import DiskCache, json_sha256

_SAMPLE_PAYLOAD = {
    "model": "test/model",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.0,
    "top_p": VALIDATOR_TOP_P,
}
_SAMPLE_RESPONSE_PAYLOAD = {
    "id": "chatcmpl-abc",
    "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
_SAMPLE_UPSTREAM_RESPONSE = UpstreamResponse(
    body=json.dumps(_SAMPLE_RESPONSE_PAYLOAD).encode("utf-8"),
    payload=_SAMPLE_RESPONSE_PAYLOAD,
    status=200,
    headers=httpx.Headers({"Content-Type": "application/json"}),
    first_token_latency_ms=None,
)


class _MockUpstreamClient(HttpxUpstreamClient):
    """Test double that returns a fixed response without making HTTP calls."""

    def __init__(self, response: UpstreamResponse) -> None:
        self._mock_response = response
        self.call_count = 0

    def fetch(self, **kwargs) -> UpstreamResponse:
        self.call_count += 1
        return self._mock_response


class CachedUpstreamClientTest(unittest.TestCase):
    def _client(self, tmp: str, inner=None) -> CachedUpstreamClient:
        return CachedUpstreamClient(Path(tmp), inner=inner)

    def _fetch(self, client: CachedUpstreamClient, payload=_SAMPLE_PAYLOAD) -> UpstreamResponse:
        return client.fetch(
            command="POST",
            url="http://unused",
            headers={},
            body=None,
            prepared_payload=payload,
            request_path="/v1/chat/completions",
            start=0.0,
        )

    def test_cache_miss_raises_without_inner(self):
        with TemporaryDirectory() as tmp:
            client = self._client(tmp)
            with self.assertRaises(CacheMissError):
                self._fetch(client)

    def test_cache_miss_delegates_to_inner(self):
        with TemporaryDirectory() as tmp:
            inner = _MockUpstreamClient(_SAMPLE_UPSTREAM_RESPONSE)
            client = self._client(tmp, inner=inner)
            result = self._fetch(client)
            self.assertEqual(result.payload, _SAMPLE_RESPONSE_PAYLOAD)
            self.assertEqual(inner.call_count, 1)

    def test_cache_hit_returns_stored_response(self):
        with TemporaryDirectory() as tmp:
            inner = _MockUpstreamClient(_SAMPLE_UPSTREAM_RESPONSE)
            client = self._client(tmp, inner=inner)
            self._fetch(client)  # miss — writes cache
            result = self._fetch(client)  # hit — no inner call
            self.assertEqual(result.payload, _SAMPLE_RESPONSE_PAYLOAD)
            self.assertEqual(inner.call_count, 1)

    def test_successful_response_written_to_disk(self):
        with TemporaryDirectory() as tmp:
            inner = _MockUpstreamClient(_SAMPLE_UPSTREAM_RESPONSE)
            client = self._client(tmp, inner=inner)
            self._fetch(client)
            key = json_sha256({"path": "/v1/chat/completions", "body": _SAMPLE_PAYLOAD})
            entry = DiskCache(Path(tmp)).read(key)
            assert entry is not None, "expected cache entry to be written"
            self.assertEqual(entry["status_code"], 200)
            self.assertEqual(entry["body"], _SAMPLE_RESPONSE_PAYLOAD)

    def test_error_response_not_written_to_disk(self):
        error_response = UpstreamResponse(
            body=b'{"error": "rate limited"}',
            payload={"error": "rate limited"},
            status=429,
            headers=httpx.Headers({}),
            first_token_latency_ms=None,
        )
        with TemporaryDirectory() as tmp:
            inner = _MockUpstreamClient(error_response)
            client = self._client(tmp, inner=inner)
            self._fetch(client)
            key = json_sha256({"path": "/v1/chat/completions", "body": _SAMPLE_PAYLOAD})
            entry = DiskCache(Path(tmp)).read(key)
            self.assertIsNone(entry)

    def test_replay_only_miss_raises(self):
        with TemporaryDirectory() as tmp:
            client = self._client(tmp, inner=None)
            with self.assertRaises(CacheMissError):
                self._fetch(client)

    def test_record_then_replay_roundtrip(self):
        with TemporaryDirectory() as tmp:
            inner = _MockUpstreamClient(_SAMPLE_UPSTREAM_RESPONSE)
            record_client = self._client(tmp, inner=inner)
            self._fetch(record_client)

            replay_client = self._client(tmp, inner=None)
            result = self._fetch(replay_client)
            self.assertEqual(result.payload, _SAMPLE_RESPONSE_PAYLOAD)
            self.assertEqual(inner.call_count, 1)

    def test_different_payloads_cached_independently(self):
        payload_a = {**_SAMPLE_PAYLOAD, "messages": [{"role": "user", "content": "A"}]}
        payload_b = {**_SAMPLE_PAYLOAD, "messages": [{"role": "user", "content": "B"}]}
        response_a = UpstreamResponse(
            body=b'{"id": "a"}',
            payload={"id": "a"},
            status=200,
            headers=httpx.Headers({}),
            first_token_latency_ms=None,
        )
        response_b = UpstreamResponse(
            body=b'{"id": "b"}',
            payload={"id": "b"},
            status=200,
            headers=httpx.Headers({}),
            first_token_latency_ms=None,
        )
        with TemporaryDirectory() as tmp:
            call_count = [0]

            class _SelectiveMock(HttpxUpstreamClient):
                def fetch(self, *, prepared_payload, **kwargs):
                    call_count[0] += 1
                    return response_a if prepared_payload == payload_a else response_b

            client = CachedUpstreamClient(Path(tmp), inner=_SelectiveMock())
            r_a = self._fetch(client, payload_a)
            r_b = self._fetch(client, payload_b)
            self.assertEqual(r_a.payload["id"], "a")
            self.assertEqual(r_b.payload["id"], "b")
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)


class CacheDeterminismTest(unittest.TestCase):
    """Cache entries are immutable: first write wins, later calls never overwrite."""

    def _fetch(
        self, client: CachedUpstreamClient, payload: dict = _SAMPLE_PAYLOAD
    ) -> UpstreamResponse:
        return client.fetch(
            command="POST",
            url="http://unused",
            headers={},
            body=None,
            prepared_payload=payload,
            request_path="/v1/chat/completions",
            start=0.0,
        )

    def _response(self, tag: str) -> UpstreamResponse:
        payload = {"id": tag}
        return UpstreamResponse(
            body=json.dumps(payload).encode("utf-8"),
            payload=payload,
            status=200,
            headers=httpx.Headers({}),
            first_token_latency_ms=None,
        )

    def test_existing_entry_not_replaced_by_new_inner(self):
        """A cached entry recorded by one client is returned as-is by a later client
        with a different inner — the second inner is never called."""
        with TemporaryDirectory() as tmp:
            inner_v1 = _MockUpstreamClient(self._response("v1"))
            inner_v2 = _MockUpstreamClient(self._response("v2"))

            self._fetch(CachedUpstreamClient(Path(tmp), inner=inner_v1))
            result = self._fetch(CachedUpstreamClient(Path(tmp), inner=inner_v2))

            self.assertEqual(result.payload["id"], "v1")
            self.assertEqual(inner_v2.call_count, 0)

    def test_repeated_calls_always_return_same_response(self):
        """Multiple calls with the same payload always return the identical cached response."""
        with TemporaryDirectory() as tmp:
            inner = _MockUpstreamClient(_SAMPLE_UPSTREAM_RESPONSE)
            client = CachedUpstreamClient(Path(tmp), inner=inner)

            results = [self._fetch(client) for _ in range(3)]

            self.assertTrue(all(r.payload == _SAMPLE_RESPONSE_PAYLOAD for r in results))
            self.assertEqual(inner.call_count, 1)

    def test_replay_only_client_returns_same_response_as_record_client(self):
        """A replay-only client reading the same cache_dir returns the same response
        that was recorded by the record client."""
        with TemporaryDirectory() as tmp:
            record = CachedUpstreamClient(
                Path(tmp), inner=_MockUpstreamClient(_SAMPLE_UPSTREAM_RESPONSE)
            )
            replay = CachedUpstreamClient(Path(tmp), inner=None)

            recorded = self._fetch(record)
            replayed = self._fetch(replay)

            self.assertEqual(recorded.payload, replayed.payload)
            self.assertEqual(recorded.status, replayed.status)


class ProxyCacheBudgetTest(unittest.TestCase):
    """Budget limits are enforced against every request, including cache hits."""

    def _body_and_payload(self) -> tuple:
        p = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
        return json.dumps(p).encode("utf-8"), p

    def _proxy(self, tmp: str, budget: SolveBudget) -> OpenRouterProxy:
        return OpenRouterProxy(
            openrouter_api_key="key",
            cache_dir=Path(tmp),
            solve_budget=budget,
        )

    def test_request_limit_counts_every_request(self):
        """max_requests applies to all requests; caching does not bypass it."""
        body, payload = self._body_and_payload()
        with TemporaryDirectory() as tmp:
            proxy = self._proxy(tmp, SolveBudget(max_requests=1))

            _, r1 = proxy._prepare_request_body(body=body, request_payload=dict(payload))
            self.assertIsNone(r1)

            _, r2 = proxy._prepare_request_body(body=body, request_payload=dict(payload))
            self.assertEqual(r2, REQUEST_LIMIT_EXIT_REASON)

    def test_token_budget_accumulates_from_cached_response_payload(self):
        """Tokens reported in a cached response payload count toward max_total_tokens."""
        body, payload = self._body_and_payload()
        with TemporaryDirectory() as tmp:
            proxy = self._proxy(tmp, SolveBudget(max_total_tokens=8))

            # Simulate a completed request (cache hit or upstream) that consumed 8 tokens.
            proxy._record_request(
                ProxyRequestRecord(
                    method="POST",
                    path="/v1/chat/completions",
                    status_code=200,
                    latency_ms=0,
                    request_model="m",
                    total_tokens=8,
                )
            )
            self.assertEqual(proxy.usage_snapshot().total_tokens, 8)

            # The next request should be blocked: tokens already at the limit.
            _, rejection = proxy._prepare_request_body(body=body, request_payload=dict(payload))
            self.assertEqual(rejection, TOKEN_LIMIT_EXIT_REASON)

    def test_budget_exceeded_reason_not_set_by_cache_miss(self):
        """A replay-only cache miss does not set budget_exceeded_reason,
        so subsequent requests are not blocked."""
        body, payload = self._body_and_payload()
        with TemporaryDirectory() as tmp:
            proxy = self._proxy(tmp, SolveBudget(max_requests=2))
            proxy.cache_replay_only = True  # type: ignore[misc]

            # Simulate the CacheMissError path manually (budget check still ran).
            proxy._prepare_request_body(body=body, request_payload=dict(payload))

            # budget_exceeded_reason must remain None — cache miss is not a budget event.
            self.assertIsNone(proxy.usage_snapshot().budget_exceeded_reason)


class ProxyCacheMissCountersTest(unittest.TestCase):
    """Usage counters are correct when a replay-only cache miss occurs.

    These tests encode the expected behaviour and currently FAIL due to two bugs:

    Bug #1 — double request_count: _prepare_request_body increments request_count
    before the upstream fetch; the CacheMissError handler then also increments
    rejected_request_count, so a single miss inflates both counters.

    Bug #3 — missing error_count: the CacheMissError handler bypasses
    _record_request, so error_count is never incremented for cache misses.
    """

    _REQUEST_BODY = json.dumps({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
    }).encode("utf-8")

    def _make_proxy(self, tmp: str, **extra) -> OpenRouterProxy:
        return OpenRouterProxy(
            openrouter_api_key="key",
            cache_dir=Path(tmp),
            cache_replay_only=True,
            require_auth=False,
            enforced_sampling_params=None,
            **extra,
        )

    def _post(self, proxy: OpenRouterProxy) -> httpx.Response:
        return httpx.post(
            f"http://{proxy.host}:{proxy.port}/v1/chat/completions",
            content=self._REQUEST_BODY,
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )

    def test_cache_miss_returns_503_with_correct_error_type(self):
        """Replay-only miss returns 503 with error.type == proxy_cache_miss."""
        with TemporaryDirectory() as tmp:
            with self._make_proxy(tmp) as proxy:
                resp = self._post(proxy)
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error"]["type"], "proxy_cache_miss")

    def test_cache_miss_does_not_increment_request_count(self):
        """Bug #1: a cache miss must not count as an upstream request (request_count stays 0)."""
        with TemporaryDirectory() as tmp:
            with self._make_proxy(tmp) as proxy:
                self._post(proxy)
                usage = proxy.usage_snapshot()
        self.assertEqual(usage.request_count, 0)
        self.assertEqual(usage.rejected_request_count, 1)

    def test_cache_miss_increments_error_count(self):
        """Bug #3: a cache miss is a failed request and must appear in error_count."""
        with TemporaryDirectory() as tmp:
            with self._make_proxy(tmp) as proxy:
                self._post(proxy)
                usage = proxy.usage_snapshot()
        self.assertEqual(usage.error_count, 1)

    def test_cache_misses_do_not_deplete_request_budget(self):
        """Bug #1 consequence: cache misses must not consume max_requests slots.

        With max_requests=1, two consecutive misses should both return 503 (cache
        miss), not have the second one return 429 (budget exceeded) because the
        first miss incorrectly consumed the one allowed slot.
        """
        with TemporaryDirectory() as tmp:
            with self._make_proxy(tmp, solve_budget=SolveBudget(max_requests=1)) as proxy:
                r1 = self._post(proxy)
                r2 = self._post(proxy)
                usage = proxy.usage_snapshot()
        self.assertEqual(r1.status_code, 503)
        self.assertEqual(r2.status_code, 503)
        self.assertIsNone(usage.budget_exceeded_reason)
        self.assertEqual(usage.rejected_request_count, 2)
        self.assertEqual(usage.request_count, 0)


if __name__ == "__main__":
    unittest.main()
