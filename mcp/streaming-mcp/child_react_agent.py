"""Child ReAct agent: Streamlit child -> child MCP streaming tool."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from deep_agent_factory import build_process_query

load_dotenv()
process_query = build_process_query(
    server_name="child-mcp",
    server_url=os.getenv("CHILD_MCP_URL", "http://127.0.0.1:8001/mcp"),
    system_prompt=(
        "You are a child ReAct agent for account support. Decide which tools best answer "
        "the user's request: use validate_email_address for verification questions and "
        "give_profile_information for profile questions. If the user asks to verify an "
        "email and retrieve its profile, use both tools, validating first. Use only the "
        "tools needed for other requests, then provide a concise summary without "
        "repeating the live updates."
    ),
)
