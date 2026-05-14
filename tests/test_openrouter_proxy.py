import json
import unittest
from unittest.mock import patch

from openrouter_proxy import OpenRouterProxy, _upstream_base_url


class OpenRouterProxyModelEnforcementTest(unittest.TestCase):
    def test_upstream_base_url_reads_env_at_request_time(self):
        with patch.dict(
            "openrouter_proxy.os.environ",
            {"OPENROUTER_BASE_URL": "https://example.test/custom/v1"},
            clear=False,
        ):
            self.assertEqual(_upstream_base_url(), "https://example.test/custom")

    def test_rewrites_requested_model_to_validator_model(self):
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", enforced_model="validator/model")
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 12,
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["model"], "validator/model")

    def test_adds_validator_model_when_request_omits_model(self):
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", enforced_model="validator/model")
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["model"], "validator/model")

    def test_rewrites_sampling_params_to_validator_policy(self):
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", enforced_model="validator/model")
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.0,
                "top_p": 0.2,
                "top_k": 7,
                "seed": 123,
                "presence_penalty": 1.5,
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["temperature"], 0.0)
        self.assertEqual(prepared["top_p"], 1.0)
        self.assertNotIn("top_k", prepared)
        self.assertNotIn("seed", prepared)
        self.assertNotIn("presence_penalty", prepared)

    def test_rewrites_provider_to_validator_policy(self):
        proxy = OpenRouterProxy(
            openrouter_api_key="upstream-key",
            enforced_model="validator/model",
            enforced_provider={
                "sort": "throughput",
                "only": ["validator/highspeed"],
                "allow_fallbacks": False,
                "preferred_min_throughput": {"p90": 50},
            },
        )
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "provider": {"only": ["slow-provider"]},
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(
            prepared["provider"],
            {
                "sort": "throughput",
                "only": ["validator/highspeed"],
                "allow_fallbacks": False,
                "preferred_min_throughput": {"p90": 50},
            },
        )


if __name__ == "__main__":
    unittest.main()
