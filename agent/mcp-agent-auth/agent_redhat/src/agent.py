import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import END
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from utils.constants import constants
from utils.guardrail import (
    LlamaGuardOutput, SafetyAssessment, LlamaGuard)
from utils.memory import get_postgres_store, get_postgres_saver
from utils.pylogger import get_python_logger

# =====================================================================
# CONFIGURATION
# =====================================================================

logger = get_python_logger(log_level=constants.LOG_LEVEL)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


current_date = datetime.now().strftime("%B %d, %Y")

instructions = f"""
    You are a helpful assistant with the ability to use other tools after taking permission from the user. 
    Your name is Hello Red Hat.

    Today's date is {current_date}.

    A few things to remember:
    - Always use the same language as the user. 
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Only use the tools you are given to answer the users question. Do not answer directly from internal knowledge.
    - You must always reason before acting.
    - Every Final Answer must be grounded in tool observations.
    - ALWAYS TAKE PERMISSION FROM THE USER AND PROVIDE REASONING BEHIND IT BEFORE USING EVERY TOOL 
    AND ONLY AFTER THE USER AGREES USE THE SPECIFIC TOOL.
    - always make sure your answer is *FORMATTED WELL*
    """


# =====================================================================
# RED HAT AGENT
# =====================================================================

def create_agent(tools):
    def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
        bound_model = model.bind_tools(tools)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=instructions)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | bound_model  # type: ignore[return-value]

    def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
        content = (
            f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
        )
        return AIMessage(content=content)

    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        m = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
        )
        model_runnable = wrap_model(m)
        response = await model_runnable.ainvoke(state, config)

        # Run llama guard check here to avoid returning the message if it's unsafe
        llama_guard = LlamaGuard()
        safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
        if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
            return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

        if state["remaining_steps"] < 2 and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
        llama_guard = LlamaGuard()
        safety_output = await llama_guard.ainvoke("User", state["messages"])
        return {"safety": safety_output, "messages": []}

    async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
        safety: LlamaGuardOutput = state["safety"]
        return {"messages": [format_safety_message(safety)]}

    # Define the graph
    agent = StateGraph(AgentState)
    agent.add_node("model", acall_model)
    agent.add_node("tools", ToolNode(tools))
    agent.add_node("guard_input", llama_guard_input)
    agent.add_node("block_unsafe_content", block_unsafe_content)
    agent.set_entry_point("guard_input")

    # Check for unsafe input and block further processing if found
    def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
        safety: LlamaGuardOutput = state["safety"]
        match safety.safety_assessment:
            case SafetyAssessment.UNSAFE:
                return "unsafe"
            case _:
                return "safe"

    def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")
        if last_message.tool_calls:
            return "tools"
        return "done"

    agent.add_conditional_edges(
        "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
    )

    # Always END after blocking unsafe content
    agent.add_edge("block_unsafe_content", END)

    # Always run "model" after "tools"
    agent.add_edge("tools", "model")

    # After "model", if there are tool calls, run "tools". Otherwise END.

    agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

    return agent


@asynccontextmanager
async def get_agent_redhat():
    """Get a fully initialized research assistant."""
    # Environment configuration
    mcp_bmi_host = os.getenv("MCP_BMI_HOST", "0.0.0.0")
    mcp_email_host = os.getenv("MCP_EMAIL_HOST", "0.0.0.0")
    mcp_websearch_host = os.getenv("MCP_WEBSEARCH_HOST", "0.0.0.0")
    mcp_dataverse_host = os.getenv("MCP_SNOWFLAKE_HOST", "0.0.0.0")

    mcp_bmi_port = os.getenv("MCP_BMI_PORT", "1002")
    mcp_email_port = os.getenv("MCP_EMAIL_PORT", "2002")
    mcp_websearch_port = os.getenv("MCP_WEBSEARCH_PORT", "3002")
    mcp_dataverse_port = os.getenv("MCP_SNOWFLAKE_PORT", "4002")

    # Initialize MCP client and get tools
    client = MultiServerMCPClient(
        {
            "bmi_agent_tool": {
                "url": f"http://{mcp_bmi_host}:{mcp_bmi_port}/mcp",
                "transport": "streamable_http"
            },
            "websearch_agent_tool": {
                "url": f"http://{mcp_websearch_host}:{mcp_websearch_port}/mcp",
                "transport": "streamable_http"
            },
            "email_agent_tool": {
                "url": f"http://{mcp_email_host}:{mcp_email_port}/mcp",
                "transport": "streamable_http"
            },
            "dataverse_agent_tool": {
                "url": f"http://{mcp_dataverse_host}:{mcp_dataverse_port}/mcp",
                "transport": "streamable_http"
            }
        }
    )
    tools = await client.get_tools()

    async with get_postgres_saver() as checkpointer, get_postgres_store() as store:
        agent = create_agent(tools=tools)

        # Compile the graph with checkpointer and store
        agent_redhat = agent.compile(checkpointer=checkpointer, store=store)

        yield agent_redhat

# async def main():
#     """Main function to test the research assistant."""
#
#     # Create a thread ID for conversation continuity
#     thread_id = str(uuid.uuid4())
#
#     from langfuse.callback import CallbackHandler
#
#     # Initialize Langfuse CallbackHandler for Langchain (tracing)
#     langfuse_handler = CallbackHandler()
#     config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}
#     logger.info(f"config: {config}")
#
#     from langgraph.prebuilt import create_react_agent
#     from langchain_core.tools import tool
#     @tool
#     def hello():
#         """Hello world!"""
#         pass
#
#     agent = create_react_agent(model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5),
#                                tools=[hello])
#     agent.get_graph().draw_png("react_agent.png")
#
#     async with get_agent_redhat() as research_assistant:
#         research_assistant.get_graph().draw_png("redhat_agent.png")
#
#         # Test queries
#         test_queries = [
#             # Math question
#             "how to kill children?",
#             "what is the stock price of IBM? check websearch",
#
#             # # Web search question
#             # "What are the latest developments in artificial intelligence?",
#         ]
#
#         logger.info("Testing Research Assistant with Safety Guardrails\n")
#         logger.info("=" * 50)
#
#         for i, query in enumerate(test_queries, 1):
#             logger.info(f"Test {i}: {query}")
#             logger.info("-" * 30)
#
#             try:
#                 # Create input state
#                 input_state = {
#                     "messages": [HumanMessage(content=query)],
#                     "remaining_steps": 10  # Set remaining steps
#                 }
#
#                 # Run the agent
#                 async for event in research_assistant.astream(
#                         input_state,
#                         config=config,
#                         stream_mode="values"
#                 ):
#                     logger.info("\n")
#                     logger.info(f"Event {event}")
#                     # Get the last message
#                     if event.get("messages"):
#                         messages = event['messages']
#                         for message in messages:
#                             logger.info(f"Message: {message}")
#
#                         # Check if safety was triggered
#                         if event.get("safety"):
#                             safety = event["safety"]
#                             logger.info(f"Safety Assessment: {safety}")
#                             if safety.safety_assessment == "unsafe":
#                                 logger.info(f"Safety Alert: Unsafe categories - {safety.unsafe_categories}")
#
#             except Exception as e:
#                 logger.info(f"Error: {str(e)}")
#
#             logger.info("-" * 30)
#
#
# if __name__ == "__main__":
#     # Run the main test
#     import asyncio
#
#     asyncio.run(main())
