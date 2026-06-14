import unittest

from tau.io.upstream_request_policy import (
    SOLVER_SHELL_TOOL,
    UpstreamRequestPolicy,
    apply_upstream_request_policy,
    build_upstream_request_policy,
    drop_empty_assistant_messages,
)


class UpstreamRequestPolicyTest(unittest.TestCase):
    def test_build_returns_none_when_unconfigured(self):
        self.assertIsNone(build_upstream_request_policy())

    def test_build_shell_tools_policy(self):
        policy = build_upstream_request_policy(shell_tools=True, empty_response_retries=5)
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertTrue(policy.shell_tools)
        self.assertEqual(policy.empty_response_retries, 5)

    def test_text_only_and_shell_tools_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            UpstreamRequestPolicy(text_only=True, shell_tools=True)

    def test_apply_shell_tools_policy(self):
        payload = {
            "model": "provider/model",
            "tools": [{"type": "function", "function": {"name": "miner_tool"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
            ],
        }
        apply_upstream_request_policy(payload, UpstreamRequestPolicy(shell_tools=True))
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"], [SOLVER_SHELL_TOOL])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(len(payload["messages"]), 1)

    def test_apply_text_only_policy(self):
        payload = {
            "model": "provider/model",
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
            ],
        }
        apply_upstream_request_policy(payload, UpstreamRequestPolicy(text_only=True))
        self.assertEqual(payload["tool_choice"], "none")
        self.assertNotIn("tools", payload)
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(len(payload["messages"]), 1)

    def test_drop_empty_assistant_messages_keeps_nonempty(self):
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "ok"},
            {"role": "assistant", "content": ""},
        ]
        cleaned = drop_empty_assistant_messages(messages)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[1]["content"], "ok")


if __name__ == "__main__":
    unittest.main()
