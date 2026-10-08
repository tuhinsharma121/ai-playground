"""Parent FastMCP server exposing one orchestration tool."""

from __future__ import annotations

import os

from fastmcp import Context, FastMCP

from child_react_agent import process_query as run_child_agent
from events import encode_relay_event

mcp = FastMCP("parent-mcp")
HTTP_HOST = os.getenv("PARENT_MCP_SERVER_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("PARENT_MCP_SERVER_PORT", "8000"))
HTTP_PATH = os.getenv("PARENT_MCP_SERVER_PATH", "/mcp")


@mcp.tool()
async def run_downstream_react_agent(ctx: Context, query: str) -> str:
    """Run the server-side child ReAct agent against the child MCP server."""
    step = 0
    final_response: list[str] = []

    async def emit(kind: str, text: str, *, tool: str | None = None, call_id: str | None = None) -> None:
        nonlocal step
        step += 1
        await ctx.report_progress(
            progress=step,
            total=None,
            message=encode_relay_event(kind=kind, text=text, tool=tool, call_id=call_id),
        )

    await emit("child_started", "Server-side child ReAct agent started")
    async for event in run_child_agent(query):
        kind = event["type"]
        if kind == "on_tool_start":
            await emit("child_tool_started", f"Calling {event['tool']}", tool=event["tool"], call_id=event["call_id"])
        elif kind == "on_tool_stream":
            await emit("child_mcp_progress", event["chunk"], tool=event["tool"], call_id=event["call_id"])
        elif kind == "on_tool_end":
            await emit("child_tool_finished", f"Finished {event['tool']}", tool=event["tool"], call_id=event["call_id"])
        elif kind == "on_tool_error":
            await emit("child_tool_failed", f"{event['tool']} failed", tool=event["tool"], call_id=event["call_id"])
        elif kind == "on_agent_thinking":
            await emit("child_reasoning", event["text"])
        elif kind == "on_chat_model_stream":
            final_response.append(event["text"])
            await emit("child_response", event["text"])

    await emit("child_completed", "Server-side child ReAct agent completed")
    return "".join(final_response).strip() or "Child workflow completed."


def main() -> None:
    mcp.run(transport="streamable-http", host=HTTP_HOST, port=HTTP_PORT, path=HTTP_PATH)


if __name__ == "__main__":
    main()
