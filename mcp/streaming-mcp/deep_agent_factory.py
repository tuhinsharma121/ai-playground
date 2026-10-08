"""Reusable Deep Agents implementation for streamed MCP tools."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient

from events import (
    AgentEvent,
    TOOL_LIFECYCLE_EVENT,
    TOOL_STREAM_EVENT,
    decode_relay_event,
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOOL_TIMEOUT_SECONDS = 600.0
_DEEP_AGENT_BUILTIN_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)
_active_run_config: ContextVar[RunnableConfig | None] = ContextVar("active_run_config", default=None)
_active_call_id: ContextVar[str | None] = ContextVar("active_call_id", default=None)


def _configure_gemini_deep_agent() -> None:
    """Keep the Deep Agents extension points without changing today's MCP-only tool surface."""
    register_harness_profile(
        f"google_genai:{MODEL}",
        HarnessProfile(
            excluded_tools=_DEEP_AGENT_BUILTIN_TOOLS,
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )


def build_process_query(
        *,
        server_name: str,
        server_url: str,
        system_prompt: str,
) -> Callable[[str, list[dict[str, str]] | None], AsyncIterator[AgentEvent]]:
    """Build an async query stream for one Deep Agent/MCP-server pairing."""
    _configure_gemini_deep_agent()

    async def on_progress(progress: float, total: float | None, message: str | None, context: Any) -> None:
        relay = decode_relay_event(message)
        payload: dict[str, Any] = {
            "tool": getattr(context, "tool_name", None) or "mcp_tool",
            "call_id": _active_call_id.get() or "unknown",
            "chunk": relay.get("text", "") if relay else message or "",
            "progress": progress,
            "total": total,
            "source": "child_agent" if relay else "mcp",
            "stream_kind": relay.get("kind", "mcp_progress") if relay else "mcp_progress",
        }
        if relay and relay.get("tool"):
            payload["child_tool"] = relay["tool"]
        config = _active_run_config.get()
        if config is not None:
            await adispatch_custom_event(TOOL_STREAM_EVENT, payload, config=config)

    def wrap_tool(base_tool: BaseTool) -> StructuredTool:
        async def call(**kwargs: Any) -> Any:
            config = ensure_config()
            config_token = _active_run_config.set(config)
            call_id = str(uuid4())
            call_token = _active_call_id.set(call_id)
            completed = False
            try:
                await adispatch_custom_event(
                    TOOL_LIFECYCLE_EVENT,
                    {"phase": "start", "tool": base_tool.name, "call_id": call_id},
                    config=config,
                )
                result = await base_tool.ainvoke(kwargs)
                completed = True
                return result
            except Exception:
                await adispatch_custom_event(
                    TOOL_LIFECYCLE_EVENT,
                    {"phase": "error", "tool": base_tool.name, "call_id": call_id},
                    config=config,
                )
                raise
            finally:
                if completed:
                    await adispatch_custom_event(
                        TOOL_LIFECYCLE_EVENT,
                        {"phase": "end", "tool": base_tool.name, "call_id": call_id},
                        config=config,
                    )
                _active_call_id.reset(call_token)
                _active_run_config.reset(config_token)

        return StructuredTool(name=base_tool.name, description=base_tool.description, args_schema=base_tool.args_schema,
                              coroutine=call)

    async def process_query(query: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[AgentEvent]:
        messages = list(history or []) + [{"role": "user", "content": query}]
        client = MultiServerMCPClient({server_name: {"transport": "streamable_http", "url": server_url,
                                                     "timeout": TOOL_TIMEOUT_SECONDS,
                                                     "sse_read_timeout": TOOL_TIMEOUT_SECONDS}},
                                      callbacks=Callbacks(on_progress=on_progress))
        tools = [wrap_tool(tool) for tool in await client.get_tools()]
        model = ChatGoogleGenerativeAI(
            model=MODEL,
            max_tokens=4096,
            temperature=0,
            streaming=True,
            include_thoughts=True,
        )
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            subagents=[],
        )
        async for event in agent.astream_events({"messages": messages}, version="v2"):
            if event["event"] == "on_custom_event" and event["name"] == TOOL_STREAM_EVENT:
                yield {"type": "on_tool_stream", **event["data"]}
            elif event["event"] == "on_custom_event" and event["name"] == TOOL_LIFECYCLE_EVENT:
                phase = event["data"]["phase"]
                if phase == "start":
                    yield {"type": "on_tool_start", **event["data"]}
                elif phase == "error":
                    yield {"type": "on_tool_error", **event["data"]}
                elif phase == "end":
                    yield {"type": "on_tool_end", **event["data"]}
            elif event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                for block in getattr(chunk, "content_blocks", []):
                    thought = block.get("thinking") or block.get("reasoning")
                    if block.get("type") in {"thinking", "reasoning"} and thought:
                        yield {
                            "type": "on_agent_thinking",
                            "run_id": event["run_id"],
                            "text": str(thought),
                        }
                text = str(getattr(chunk, "text", "") or "")
                if text:
                    yield {"type": "on_chat_model_stream", "text": text}

    return process_query
