import unittest

from tau.io.chat_completion import (
    append_stream_text_part,
    assistant_text_from_payload,
    is_retryable_empty_response,
    normalize_chat_completion_payload,
    normalize_message_text,
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

    def test_tool_calls_use_first_command_only(self):
        text = tool_calls_to_bash_blocks(
            [
                {"function": {"name": "bash", "arguments": '{"command": "first"}'}},
                {"function": {"name": "bash", "arguments": '{"command": "second"}'}},
            ],
        )
        self.assertIn("first", text)
        self.assertNotIn("second", text)

    def test_top_level_tool_call_fields(self):
        text = tool_calls_to_bash_blocks(
            [{"name": "run_terminal_cmd", "input": {"command": "pwd"}}],
        )
        self.assertEqual(text, "```bash\npwd\n```")

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

    def test_content_list_with_tool_use_part(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Running a probe."},
                            {
                                "type": "tool_use",
                                "name": "bash",
                                "input": {"command": "grep -rn TODO ."},
                            },
                        ],
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        text = assistant_text_from_payload(normalize_chat_completion_payload(payload))
        self.assertIn("Running a probe.", text)
        self.assertIn("grep -rn TODO .", text)
        self.assertIn("```bash", text)

    def test_kimi_native_tool_tokens_in_text(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I'll inspect the repo.\n"
                            "<|tool_call_begin|>bash<|tool_call_argument_begin|>"
                            '{"command":"find . -maxdepth 2"}'
                            "<|tool_call_end|>"
                        ),
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        text = assistant_text_from_payload(normalize_chat_completion_payload(payload))
        self.assertIn("```bash", text)
        self.assertIn("find . -maxdepth 2", text)
        self.assertNotIn("tool_call_begin", text)

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

    def test_tool_calls_finish_reason_is_retryable_when_unparsed(self):
        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "", "tool_calls": []},
                    "finish_reason": "tool_calls",
                },
            ],
        }
        self.assertTrue(payload_has_retryable_empty_content(payload))

    def test_stream_text_part_keeps_partial_strings_raw(self):
        parts: list[str] = []
        self.assertTrue(append_stream_text_part(parts, "hel"))
        self.assertTrue(append_stream_text_part(parts, "lo"))
        self.assertEqual("".join(parts), "hello")

    def test_default_bash_session_tool_call_becomes_bash_block(self):
        text = tool_calls_to_bash_blocks(
            [{"function": {"name": "default_bash_session", "arguments": '{"command": "pwd"}'}}],
        )
        self.assertEqual(text, "```bash\npwd\n```")

    def test_normalize_message_text_handles_input_text_parts(self):
        text = normalize_message_text(
            [{"type": "input_text", "text": "alpha"}, {"type": "input_text", "text": "beta"}],
        )
        self.assertEqual(text, "alphabeta")


if __name__ == "__main__":
    unittest.main()
