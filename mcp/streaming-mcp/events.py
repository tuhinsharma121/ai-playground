"""Structured events shared by the agent adapter, MCP relay, and Streamlit UI."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

TOOL_LIFECYCLE_EVENT = "tool_lifecycle"
TOOL_STREAM_EVENT = "tool_stream"
RELAY_PREFIX = "streaming-mcp:"

EventType = Literal[
    "on_tool_start",
    "on_tool_stream",
    "on_tool_end",
    "on_tool_error",
    "on_agent_thinking",
    "on_chat_model_stream",
    "error",
]


class AgentEvent(TypedDict, total=False):
    type: EventType
    call_id: str
    tool: str
    chunk: str
    text: str
    source: str
    stream_kind: str
    progress: float
    total: float | None
    message: str


def encode_relay_event(*, kind: str, text: str, tool: str | None = None, call_id: str | None = None) -> str:
    """Encode child-agent activity in the string-only MCP progress message field."""
    payload: dict[str, str] = {"kind": kind, "text": text}
    if tool:
        payload["tool"] = tool
    if call_id:
        payload["call_id"] = call_id
    return RELAY_PREFIX + json.dumps(payload, separators=(",", ":"))


def decode_relay_event(message: str | None) -> dict[str, Any] | None:
    """Decode a relay event, leaving ordinary MCP progress messages untouched."""
    if not message or not message.startswith(RELAY_PREFIX):
        return None
    try:
        payload = json.loads(message.removeprefix(RELAY_PREFIX))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
