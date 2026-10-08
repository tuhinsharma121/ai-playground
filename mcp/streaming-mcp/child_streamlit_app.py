"""Child Streamlit entry point."""
from child_react_agent import process_query
from streamlit_ui import render_app
render_app(
    title="Child streaming agent",
    caption="Flow: Streamlit → child ReAct agent → child MCP email validation and profile lookup streams.",
    process_query=process_query,
)
