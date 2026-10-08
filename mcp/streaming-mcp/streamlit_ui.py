"""Shared synchronous Streamlit renderer for either ReAct-agent stack."""
from __future__ import annotations
import asyncio
import logging
import queue
import threading
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import streamlit as st

from events import AgentEvent

_DONE = object()
LOGGER = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 20
MAX_PENDING_EVENTS = 200


ProcessQuery = Callable[[str, list[dict[str, str]]], AsyncIterator[AgentEvent]]


def _sync_events(process_query: ProcessQuery, query: str, history: list[dict[str, str]]) -> Iterator[AgentEvent]:
    """Bridge the async agent event stream to Streamlit's synchronous script run."""
    events: queue.Queue[AgentEvent | object] = queue.Queue(maxsize=MAX_PENDING_EVENTS)

    def worker() -> None:
        async def consume() -> None:
            async for event in process_query(query, history):
                events.put(event)

        try:
            asyncio.run(consume())
        except BaseException:
            LOGGER.exception("Streaming agent failed")
            events.put({"type": "error", "message": "The agent request failed. Check the server logs for details."})
        finally:
            events.put(_DONE)

    threading.Thread(target=worker, daemon=True).start()
    while (event := events.get()) is not _DONE:
        yield event  # type: ignore[misc]


def _render_saved_tool_call(tool_call: dict[str, Any]) -> None:
    """Render a completed tool call as an expanded, collapsible card."""
    with st.expander(f"Tool: `{tool_call['tool']}`", expanded=True):
        for chunk in tool_call["chunks"]:
            st.markdown(chunk)
        st.caption("Completed")


def _render_saved_reasoning(reasoning: str, label: str = "Agent reasoning summary") -> None:
    with st.expander(label, expanded=True):
        st.markdown(reasoning)


def _format_tool_chunk(event: AgentEvent) -> str:
    labels = {
        "child_started": "Child agent",
        "child_tool_started": "Child agent",
        "child_tool_finished": "Child agent",
        "child_tool_failed": "Child agent",
        "child_mcp_progress": "Child MCP",
        "child_reasoning": "Child agent reasoning",
        "child_response": "Child agent response",
        "child_completed": "Child agent",
    }
    label = labels.get(event.get("stream_kind", ""))
    return f"**{label}:** {event['chunk']}" if label else event["chunk"]


def render_app(*, title: str, caption: str, process_query: ProcessQuery) -> None:
    st.set_page_config(page_title=title, page_icon="🔗", layout="centered")
    st.title(title); st.caption(caption)
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for activity in message.get("activity", []):
                if activity["type"] == "tool":
                    _render_saved_tool_call(activity)
                elif activity["type"] == "reasoning":
                    _render_saved_reasoning(activity["text"], activity.get("label", "Agent reasoning summary"))
            st.markdown(message["content"])
    prompt = st.chat_input("Ask for streamed messages")
    if not prompt: return
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        answer_box, answer = None, ""
        active_cards: dict[str, dict[str, Any]] = {}
        activity: list[dict[str, Any]] = []
        call_counts: defaultdict[str, int] = defaultdict(int)
        reasoning_count = 0

        def start_tool_card(tool: str, call_id: str) -> dict[str, Any]:
            if call_id in active_cards:
                return active_cards[call_id]
            call_counts[tool] += 1
            card = st.expander(f"Tool: `{tool}` · call {call_counts[tool]}", expanded=True)
            content = card.container()
            state = card.empty()
            state.caption("Streaming…")
            tool_call = {"type": "tool", "tool": tool, "chunks": [], "content": content, "state": state}
            active_cards[call_id] = tool_call
            activity.append(tool_call)
            return tool_call

        def append_reasoning_card(text: str, label: str = "Agent reasoning summary") -> None:
            nonlocal reasoning_count
            reasoning_count += 1
            title = f"{label} · step {reasoning_count}"
            with st.expander(title, expanded=True):
                st.markdown(text)
            activity.append({"type": "reasoning", "text": text, "label": title})

        for event in _sync_events(process_query, prompt, history):
            if event["type"] == "on_tool_start":
                start_tool_card(event["tool"], event["call_id"])
            elif event["type"] == "on_tool_stream":
                tool_call = start_tool_card(event["tool"], event["call_id"])
                chunk = _format_tool_chunk(event)
                tool_call["chunks"].append(chunk)
                tool_call["content"].markdown(chunk)
            elif event["type"] == "on_tool_end":
                tool_call = active_cards.pop(event["call_id"], None)
                if tool_call is not None:
                    tool_call["state"].caption("Completed")
            elif event["type"] == "on_tool_error":
                tool_call = active_cards.pop(event["call_id"], None)
                if tool_call is not None:
                    tool_call["state"].caption("Failed")
            elif event["type"] == "on_agent_thinking":
                append_reasoning_card(event["text"])
            elif event["type"] == "on_chat_model_stream":
                if answer_box is None:
                    answer_box = st.empty()
                answer += event["text"]
                answer_box.markdown(answer + "▌")
            elif event["type"] == "error": st.error(event["message"])
        for tool_call in active_cards.values():
            tool_call["state"].caption("Finished")
        if answer_box is not None:
            answer_box.markdown(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "activity": [
            {"type": item["type"], "tool": item["tool"], "chunks": item["chunks"]}
            if item["type"] == "tool"
            else item
            for item in activity
        ],
    })
