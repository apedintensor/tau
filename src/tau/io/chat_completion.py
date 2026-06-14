from __future__ import annotations

import copy
import json
from typing import Any

import tau.utils

_EMPTY_FINISH_REASONS = {"error", "length", "content_filter"}
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
    "terminal",
}


def normalize_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return str(value)


def tool_calls_to_bash_blocks(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    blocks: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        command = command_from_tool_call(
            name=str(function.get("name") or ""),
            arguments=function.get("arguments"),
        )
        if command:
            blocks.append(f"```bash\n{command}\n```")
    return "\n\n".join(blocks)


def command_from_tool_call(*, name: str, arguments: Any) -> str:
    parsed_args = parse_tool_arguments(arguments)
    if isinstance(parsed_args, dict):
        for key in ("command", "cmd", "code", "code_string", "script"):
            value = parsed_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
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
    content = normalize_message_text(message.get("content"))
    if not content.strip():
        content = normalize_message_text(message.get("reasoning") or message.get("reasoning_content"))
    tool_text = tool_calls_to_bash_blocks(message.get("tool_calls"))
    if tool_text:
        content = f"{content}\n\n{tool_text}".strip() if content.strip() else tool_text
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
        if isinstance(function, dict):
            fn = entry.setdefault("function", {"name": "", "arguments": ""})
            if function.get("name"):
                fn["name"] = str(fn.get("name") or "") + str(function["name"])
            if function.get("arguments"):
                fn["arguments"] = str(fn.get("arguments") or "") + str(function["arguments"])
