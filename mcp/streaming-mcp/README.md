# Parent/Child Streaming MCP

This project has two explicit streaming paths:

1. Child Streamlit → child ReAct agent → child MCP `validate_email_address` and `give_profile_information`.
2. Parent Streamlit → parent ReAct agent → parent MCP `run_downstream_react_agent` → server-side child ReAct agent → child account tools.

| Role | File |
| --- | --- |
| Parent Streamlit | `parent_streamlit_app.py` |
| Child Streamlit | `child_streamlit_app.py` |
| Parent ReAct agent | `parent_react_agent.py` |
| Child ReAct agent | `child_react_agent.py` |
| Parent MCP server | `parent_mcp_server.py` |
| Child MCP server | `child_mcp_server.py` |

## Run

```bash
uv sync
uv run streaming-mcp-child-server
uv run streaming-mcp-parent-server
uv run streamlit run child_streamlit_app.py --server.port 8501
uv run streamlit run parent_streamlit_app.py --server.port 8502
```

Set `GEMINI_API_KEY` and the `PARENT_MCP_*` / `CHILD_MCP_*` endpoint settings from `.env.example` as needed.

## Example queries

Open the child UI at `http://localhost:8501` for the direct child-agent route:

```text
Please validate tuhin@gmail.com and then show me the profile information for that email address. do the same for alice@gmail.com.
```

The child agent chooses both tools because the request needs email validation and profile information. Each tool streams three intermediate stages at five-second intervals.

Open the parent UI at `http://localhost:8502` for the nested orchestration route:

```text
Please validate tuhin@gmail.com and then show me the profile information for that email address. do the same for alice@gmail.com.
```

This displays the parent MCP tool's relayed progress: server-side child-agent lifecycle updates, both child tool streams, the child-agent summary, and then the parent-agent response.

## Cursor behavior

Cursor currently shows the completed result of an MCP tool call rather than its
MCP progress notifications. Every MCP tool therefore returns JSON with a temporally
ordered `execution_trace` array. The child tools include their structured account
data in `result`; the parent orchestration tool includes the child agent's summary
in `final_result`. Streamlit still renders the same events live as they arrive.
