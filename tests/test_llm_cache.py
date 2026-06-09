import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from openrouter_client import complete_text
from tau.io.openrouter import CachedLLMClient, CacheMissError, LLMRequest, MockLLMClient

_OK_PAYLOAD = {
    "choices": [{"message": {"content": "cached answer"}, "finish_reason": "stop"}],
}


class _FakeResponse:
    def raise_for_status(self): ...
    def json(self):
        return _OK_PAYLOAD


class _FakeHttpxClient:
    def __init__(self):
        self.call_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_): ...
    def post(self, url, headers, json):
        self.call_count += 1
        return _FakeResponse()


class CachedLLMClientTest(unittest.TestCase):
    def _request(self, prompt="hello"):
        return LLMRequest(prompt=prompt, model="test-model", temperature=0.0)

    def test_cache_miss_raises_without_inner(self):
        with TemporaryDirectory() as tmp:
            client = CachedLLMClient(Path(tmp))
            with self.assertRaises(CacheMissError):
                client.complete_text(self._request(), timeout=10)

    def test_inner_called_on_miss_and_result_returned(self):
        with TemporaryDirectory() as tmp:
            inner = MockLLMClient("hello from inner")
            client = CachedLLMClient(Path(tmp), inner=inner)
            result = client.complete_text(self._request(), timeout=10)
            self.assertEqual(result, "hello from inner")

    def test_response_written_to_disk_on_miss(self):
        with TemporaryDirectory() as tmp:
            inner = MockLLMClient("saved response")
            client = CachedLLMClient(Path(tmp), inner=inner)
            client.complete_text(self._request(), timeout=10)
            cache_files = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            data = json.loads(cache_files[0].read_text())
            self.assertEqual(data["response"], "saved response")

    def test_cache_hit_returns_stored_response(self):
        with TemporaryDirectory() as tmp:
            inner = MockLLMClient("first response")
            client = CachedLLMClient(Path(tmp), inner=inner)
            req = self._request()
            client.complete_text(req, timeout=10)

            # replace inner with one that would return something different
            client._inner = MockLLMClient("second response")
            result = client.complete_text(req, timeout=10)

            self.assertEqual(result, "first response")

    def test_inner_not_called_on_cache_hit(self):
        with TemporaryDirectory() as tmp:
            calls = []
            inner = MockLLMClient(lambda r: calls.append(r) or "response")
            client = CachedLLMClient(Path(tmp), inner=inner)
            req = self._request()
            client.complete_text(req, timeout=10)  # miss — writes cache
            client.complete_text(req, timeout=10)  # hit — should not call inner
            self.assertEqual(len(calls), 1)

    def test_different_prompts_cached_independently(self):
        with TemporaryDirectory() as tmp:
            call_count = [0]

            def counter(r):
                call_count[0] += 1
                return f"response for {r.prompt}"

            client = CachedLLMClient(Path(tmp), inner=MockLLMClient(counter))
            r1 = client.complete_text(self._request("prompt A"), timeout=10)
            r2 = client.complete_text(self._request("prompt B"), timeout=10)
            self.assertEqual(r1, "response for prompt A")
            self.assertEqual(r2, "response for prompt B")
            self.assertEqual(call_count[0], 2)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)

    def test_same_prompt_different_model_cached_independently(self):
        with TemporaryDirectory() as tmp:
            inner = MockLLMClient(lambda r: f"response from {r.model}")
            client = CachedLLMClient(Path(tmp), inner=inner)
            r1 = client.complete_text(LLMRequest(prompt="q", model="model-a"), timeout=10)
            r2 = client.complete_text(LLMRequest(prompt="q", model="model-b"), timeout=10)
            self.assertEqual(r1, "response from model-a")
            self.assertEqual(r2, "response from model-b")
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 2)

    def test_timeout_does_not_affect_cache_key(self):
        with TemporaryDirectory() as tmp:
            inner = MockLLMClient("response")
            client = CachedLLMClient(Path(tmp), inner=inner)
            req = self._request()
            client.complete_text(req, timeout=10)
            # same request, different timeout — should hit cache
            result = client.complete_text(req, timeout=999)
            self.assertEqual(result, "response")
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)


class CompleteTextEnvSwitchTest(unittest.TestCase):
    """Tests for the LLM_CACHE_DIR / LLM_REPLAY_DIR env var switching in complete_text."""

    _FAKE_HTTPX = "openrouter_client.httpx.Client"

    def test_llm_cache_dir_records_response(self):
        with TemporaryDirectory() as tmp:
            fake = _FakeHttpxClient()
            with patch(self._FAKE_HTTPX, return_value=fake):
                with patch.dict("openrouter_client.os.environ", {"LLM_CACHE_DIR": tmp}):
                    result = complete_text(
                        prompt="hello", model="m", timeout=10, openrouter_api_key="key"
                    )
            self.assertEqual(result, "cached answer")
            self.assertEqual(fake.call_count, 1)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 1)

    def test_llm_cache_dir_replays_on_second_call(self):
        with TemporaryDirectory() as tmp:
            fake = _FakeHttpxClient()
            with patch(self._FAKE_HTTPX, return_value=fake):
                with patch.dict("openrouter_client.os.environ", {"LLM_CACHE_DIR": tmp}):
                    complete_text(prompt="hello", model="m", timeout=10, openrouter_api_key="key")
                    complete_text(prompt="hello", model="m", timeout=10, openrouter_api_key="key")
            self.assertEqual(fake.call_count, 1)

    def test_llm_replay_dir_raises_on_miss(self):
        with TemporaryDirectory() as tmp:
            with patch.dict("openrouter_client.os.environ", {"LLM_REPLAY_DIR": tmp}):
                with self.assertRaises(CacheMissError):
                    complete_text(prompt="hello", model="m", timeout=10, openrouter_api_key="key")

    def test_llm_replay_dir_serves_pre_written_cache(self):
        with TemporaryDirectory() as tmp:
            # pre-populate cache using record mode
            fake = _FakeHttpxClient()
            with patch(self._FAKE_HTTPX, return_value=fake):
                with patch.dict("openrouter_client.os.environ", {"LLM_CACHE_DIR": tmp}):
                    complete_text(prompt="hello", model="m", timeout=10, openrouter_api_key="key")

            # now replay — no network calls
            with patch.dict("openrouter_client.os.environ", {"LLM_REPLAY_DIR": tmp}):
                result = complete_text(
                    prompt="hello", model="m", timeout=10, openrouter_api_key="key"
                )
            self.assertEqual(result, "cached answer")
            self.assertEqual(fake.call_count, 1)

    def test_llm_replay_dir_takes_precedence_over_cache_dir(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(
                "openrouter_client.os.environ",
                {"LLM_REPLAY_DIR": tmp, "LLM_CACHE_DIR": tmp},
            ):
                with self.assertRaises(CacheMissError):
                    complete_text(prompt="hello", model="m", timeout=10, openrouter_api_key="key")


if __name__ == "__main__":
    unittest.main()
