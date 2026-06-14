import unittest

from tau.io.chat_completion import (
    assistant_text_from_payload,
    is_retryable_empty_response,
    normalize_chat_completion_payload,
    payload_has_retryable_empty_content,
    tool_calls_to_bash_blocks,
)


class ChatCompletionNormalizationTest(unittest.TestCase):
    def test_tool_calls_become_bash_blocks(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "ls -la"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
        }
        normalized = normalize_chat_completion_payload(payload)
        text = assistant_text_from_payload(normalized)
        self.assertIn("```bash", text)
        self.assertIn("ls -la", text)

    def test_reasoning_fallback_is_merged_into_content(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "Need to inspect files.\n\n```bash\necho hi\n```",
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        text = assistant_text_from_payload(normalize_chat_completion_payload(payload))
        self.assertIn("echo hi", text)

    def test_malformed_function_call_is_retryable_when_empty(self):
        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "error",
                    "native_finish_reason": "MALFORMED_FUNCTION_CALL",
                },
            ],
        }
        self.assertTrue(payload_has_retryable_empty_content(payload))
        self.assertTrue(
            is_retryable_empty_response("error", "MALFORMED_FUNCTION_CALL"),
        )

    def test_tool_calls_to_bash_blocks_helper(self):
        text = tool_calls_to_bash_blocks(
            [{"function": {"name": "shell", "arguments": '{"cmd":"pwd"}'}}],
        )
        self.assertEqual(text, "```bash\npwd\n```")


if __name__ == "__main__":
    unittest.main()
