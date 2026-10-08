"""Parent Streamlit entry point."""
from parent_react_agent import process_query
from streamlit_ui import render_app
render_app(
    title="Parent streaming orchestrator",
    caption="Flow: Streamlit → parent ReAct agent → parent MCP → server-side child ReAct agent → child email and profile tools.",
    process_query=process_query,
)
