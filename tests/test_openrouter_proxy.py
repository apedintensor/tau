import json
import unittest

from openrouter_proxy import OpenRouterProxy


class OpenRouterProxyModelEnforcementTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
