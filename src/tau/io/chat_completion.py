from __future__ import annotations

import copy
import json
import re
from typing import Any

import tau.utils

_EMPTY_FINISH_REASONS = {"error", "length", "content_filter", "tool_calls"}
_EMPTY_NATIVE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",
    "MAX_TOKENS",
    "SAFETY",
    "RECITATION",
}
_SHELL_TOOL_NAMES = {
    "bash",
    "shell",
    "execute",
    "run_command",
    "run_terminal_cmd",
    "run_shell_command",
    "run_terminal_command",
    "execute_bash",
    "execute_command",
    "terminal",
    "bash_command",
    "default_bash_session",
    "default_bash_command",
    "default_api_bash",
}
_COMMAND_ARG_KEYS = (
    "command",
    "cmd",
    "code",
    "code_string",
    "script",
    "shell_command",
    "terminal_command",
    "input",
    "parameters",
    "args",
)
_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_TOOL_PART_TYPES = {"tool_use", "tool_call", "function_call"}
_NATIVE_TOOL_CALL_RES = (
    re.compile(
        r"<\|tool_call_begin\|>(?P<name>.*?)<\|tool_call_argument_begin\|>(?P<args>.*?)<\|tool_call_end\|>",
        re.DOTALL,
    ),
    re.compile(
        r"<\|redacted_tool_call_begin\|>(?P<name>.*?)<\|redacted_tool_call_argument_begin\|>(?P<args>.*?)<\|redacted_tool_call_end\|>",
        re.DOTALL,
    ),
)


def normalize_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _append_native_tool_blocks(value)
    if isinstance(value, list):
        text_parts: list[str] = []
        tool_blocks: list[str] = []
        for part in value:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").lower()
            if part_type in _TEXT_PART_TYPES:
                text_parts.append(str(part.get("text") or ""))
                continue
            if part_type in _TOOL_PART_TYPES:
                block = _command_to_bash_block(
                    name=str(part.get("name") or ""),
                    arguments=part.get("input") or part.get("arguments") or part.get("function"),
                )
                if block:
                    tool_blocks.append(block)
                continue
            text_parts.append(str(part.get("text") or part.get("content") or ""))
        text = _append_native_tool_blocks("".join(text_parts), existing_blocks=tool_blocks)
        return text
    return str(value)


def tool_calls_to_bash_blocks(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments") or tool_call.get("input"),
            }
        block = _command_to_bash_block(
            name=str(function.get("name") or tool_call.get("name") or ""),
            arguments=function.get("arguments") or function.get("input") or tool_call.get("input"),
        )
        if block:
            return block
    return ""


def command_from_tool_call(*, name: str, arguments: Any) -> str:
    parsed_args = parse_tool_arguments(arguments)
    if isinstance(parsed_args, dict):
        for key in _COMMAND_ARG_KEYS:
            value = parsed_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested = command_from_tool_call(name=name, arguments=value)
                if nested:
                    return nested
    if isinstance(parsed_args, str) and parsed_args.strip():
        lowered = name.strip().lower()
        if not lowered or lowered in _SHELL_TOOL_NAMES:
            return parsed_args.strip()
    if isinstance(arguments, str) and arguments.strip():
        lowered = name.strip().lower()
        if not lowered or lowered in _SHELL_TOOL_NAMES:
            return arguments.strip()
    return ""


def parse_tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    text = arguments.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def normalize_assistant_message(*, message: dict[str, Any], choice: dict[str, Any] | None = None) -> str:
    del choice  # reserved for future finish-reason-aware normalization
    content = normalize_message_text(message.get("content"))
    if not content.strip():
        content = normalize_message_text(message.get("reasoning") or message.get("reasoning_content"))
    if not content.strip():
        details = message.get("reasoning_details")
        if isinstance(details, list):
            content = normalize_message_text(details)
    tool_text = tool_calls_to_bash_blocks(message.get("tool_calls"))
    if tool_text:
        content = _merge_primary_bash_block(content, tool_text)
    return content.strip()


def is_retryable_empty_response(finish_reason: Any, native_finish_reason: Any) -> bool:
    native = str(native_finish_reason or "").upper()
    if native in _EMPTY_NATIVE_FINISH_REASONS:
        return True
    finish = str(finish_reason or "").lower()
    return finish in _EMPTY_FINISH_REASONS


def normalize_chat_completion_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of payload with assistant message content normalized for miners."""
    out = copy.deepcopy(payload)
    choices = out.get("choices")
    if not isinstance(choices, list):
        return out
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        message = dict(message)
        message["content"] = normalize_assistant_message(message=message, choice=choice)
        choice["message"] = message
    return out


def assistant_text_from_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return normalize_assistant_message(message=message, choice=choice)


def payload_has_retryable_empty_content(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    text = normalize_assistant_message(message=message, choice=choice)
    if text.strip():
        return False
    return is_retryable_empty_response(choice.get("finish_reason"), choice.get("native_finish_reason"))


def empty_content_error(payload: dict[str, Any]) -> str:
    choice = (payload.get("choices") or [{}])[0]
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = tau.utils.get_dict(payload, "usage")
    completion_details = tau.utils.get_dict(usage, "completion_tokens_details")
    return (
        "OpenRouter returned empty content "
        f"(finish_reason={choice.get('finish_reason')!r}, "
        f"native_finish_reason={choice.get('native_finish_reason')!r}, "
        f"message_keys={sorted(message.keys())}, "
        f"completion_tokens={usage.get('completion_tokens')!r}, "
        f"reasoning_tokens={completion_details.get('reasoning_tokens')!r})"
    )


def merge_stream_tool_call_delta(
    tool_calls_by_index: dict[int, dict[str, Any]],
    delta_tool_calls: Any,
) -> None:
    if not isinstance(delta_tool_calls, list):
        return
    for tool_call in delta_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        try:
            index = int(tool_call.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        entry = tool_calls_by_index.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if tool_call.get("id"):
            entry["id"] = str(tool_call["id"])
        if tool_call.get("type"):
            entry["type"] = str(tool_call["type"])
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments") or tool_call.get("input"),
            }
        if isinstance(function, dict):
            fn = entry.setdefault("function", {"name": "", "arguments": ""})
            if function.get("name"):
                fn["name"] = str(fn.get("name") or "") + str(function["name"])
            arguments = function.get("arguments")
            if arguments is None:
                arguments = function.get("input")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, sort_keys=True)
            if arguments is not None and str(arguments):
                fn["arguments"] = str(fn.get("arguments") or "") + str(arguments)


def append_stream_text_part(parts: list[str], value: Any) -> bool:
    """Append one streamed content/reasoning fragment without over-normalizing partial text."""
    if value is None or value == "":
        return False
    if isinstance(value, str):
        parts.append(value)
        return True
    text = normalize_message_text(value)
    if not text:
        return False
    parts.append(text)
    return True


def _command_to_bash_block(*, name: str, arguments: Any) -> str:
    command = command_from_tool_call(name=name, arguments=arguments)
    if not command:
        return ""
    return f"```bash\n{command}\n```"


def _merge_primary_bash_block(content: str, tool_text: str) -> str:
    if not content.strip():
        return tool_text
    if "```bash" in content or "```sh" in content:
        return content
    return f"{content}\n\n{tool_text}".strip()


def _append_native_tool_blocks(text: str, *, existing_blocks: list[str] | None = None) -> str:
    blocks = list(existing_blocks or [])
    cleaned = text or ""
    for pattern in _NATIVE_TOOL_CALL_RES:
        matches = list(pattern.finditer(cleaned))
        for match in matches:
            try:
                args = json.loads(match.group("args").strip())
            except (ValueError, TypeError):
                continue
            block = _command_to_bash_block(name=match.group("name") or "", arguments=args)
            if block:
                blocks.append(block)
        if matches:
            cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip()
    if not blocks:
        return text
    primary = blocks[0]
    if not cleaned:
        return primary
    if "```bash" in cleaned or "```sh" in cleaned:
        return cleaned
    return f"{cleaned}\n\n{primary}".strip()
