"""Parent ReAct agent: Streamlit parent -> parent MCP orchestration tool."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from deep_agent_factory import build_process_query

load_dotenv()

process_query = build_process_query(
    server_name="parent-mcp",
    server_url=os.getenv("PARENT_MCP_URL", "http://127.0.0.1:8000/mcp"),
    system_prompt=(
        "You are the parent ReAct agent. Always call "
        "run_downstream_react_agent for streaming requests, then give a concise "
        "summary without repeating the live updates."
    ),
)
