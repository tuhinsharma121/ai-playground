"""Parent FastMCP server exposing one orchestration tool."""

from __future__ import annotations

import os
from typing import NotRequired, TypedDict

from fastmcp import Context, FastMCP

from child_react_agent import process_query as run_child_agent
from events import encode_relay_event

mcp = FastMCP("parent-mcp")
HTTP_HOST = os.getenv("PARENT_MCP_SERVER_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("PARENT_MCP_SERVER_PORT", "8000"))
HTTP_PATH = os.getenv("PARENT_MCP_SERVER_PATH", "/mcp")


class ExecutionTraceEvent(TypedDict):
    """One observable step in the parent tool's downstream workflow."""

    sequence: int
    event: str
    message: str
    tool: NotRequired[str]
    call_id: NotRequired[str]


class DownstreamAgentResult(TypedDict):
    """Structured result returned by the parent MCP tool."""

    status: str
    execution_trace: list[ExecutionTraceEvent]
    final_result: str


def format_tool_result(
    execution_trace: list[ExecutionTraceEvent],
    final_response: str,
) -> DownstreamAgentResult:
    """Return the progress replay and final answer as JSON-serializable data."""
    return {
        "status": "completed",
        "execution_trace": execution_trace,
        "final_result": final_response,
    }


@mcp.tool()
async def run_downstream_react_agent(ctx: Context, query: str) -> DownstreamAgentResult:
    """Run the server-side child ReAct agent against the child MCP server."""
    step = 0
    execution_trace: list[ExecutionTraceEvent] = []
    final_response: list[str] = []
    reasoning_recorded = False

    async def emit(
        kind: str,
        text: str,
        *,
        tool: str | None = None,
        call_id: str | None = None,
        trace_event: str | None = None,
        trace_message: str | None = None,
    ) -> None:
        nonlocal step
        step += 1
        if trace_event and trace_message:
            trace: ExecutionTraceEvent = {
                "sequence": len(execution_trace) + 1,
                "event": trace_event,
                "message": trace_message,
            }
            if tool:
                trace["tool"] = tool
            if call_id:
                trace["call_id"] = call_id
            execution_trace.append(trace)
        await ctx.report_progress(
            progress=step,
            total=None,
            message=encode_relay_event(kind=kind, text=text, tool=tool, call_id=call_id),
        )

    await emit(
        "child_started",
        "Server-side child ReAct agent started",
        trace_event="child_agent_started",
        trace_message="Server-side child agent started.",
    )
    async for event in run_child_agent(query):
        kind = event["type"]
        if kind == "on_tool_start":
            tool_name = event["tool"]
            await emit(
                "child_tool_started",
                f"Calling {tool_name}",
                tool=tool_name,
                call_id=event["call_id"],
                trace_event="child_tool_started",
                trace_message=f"Child agent called {tool_name}.",
            )
        elif kind == "on_tool_stream":
            tool_name = event["tool"]
            chunk = event["chunk"]
            await emit(
                "child_mcp_progress",
                chunk,
                tool=tool_name,
                call_id=event["call_id"],
                trace_event="child_tool_progress",
                trace_message=chunk,
            )
        elif kind == "on_tool_end":
            tool_name = event["tool"]
            await emit(
                "child_tool_finished",
                f"Finished {tool_name}",
                tool=tool_name,
                call_id=event["call_id"],
                trace_event="child_tool_finished",
                trace_message=f"Child agent received the {tool_name} result.",
            )
        elif kind == "on_tool_error":
            tool_name = event["tool"]
            await emit(
                "child_tool_failed",
                f"{tool_name} failed",
                tool=tool_name,
                call_id=event["call_id"],
                trace_event="child_tool_failed",
                trace_message=f"Child agent's {tool_name} call failed.",
            )
        elif kind == "on_agent_thinking":
            await emit("child_reasoning", event["text"])
            if not reasoning_recorded:
                execution_trace.append(
                    {
                        "sequence": len(execution_trace) + 1,
                        "event": "child_agent_reasoning",
                        "message": "Child agent evaluated the available information and selected its next action.",
                    }
                )
                reasoning_recorded = True
        elif kind == "on_chat_model_stream":
            final_response.append(event["text"])
            await emit("child_response", event["text"])

    await emit(
        "child_completed",
        "Server-side child ReAct agent completed",
        trace_event="child_agent_completed",
        trace_message="Server-side child agent completed.",
    )
    return format_tool_result(
        execution_trace,
        "".join(final_response).strip() or "Child workflow completed.",
    )


def main() -> None:
    mcp.run(transport="streamable-http", host=HTTP_HOST, port=HTTP_PORT, path=HTTP_PATH)


if __name__ == "__main__":
    main()
