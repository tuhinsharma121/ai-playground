import math
import os
import re
from datetime import datetime

import httpx
import numexpr
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from constants import constants
from pylogger import get_python_logger

logger = get_python_logger(log_level=constants.LOG_LEVEL)


@tool
async def calculator(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API for current and relevant information.

    This tool should be used when:
    - Information beyond the knowledge cutoff date is needed
    - Current facts, news, or data are required
    - Real-time information is necessary
    - Verification of claims or statements is needed

    Usage guidelines:
    - Formulate clear, specific search queries for best results
    - For time-sensitive queries, include terms like 'today', 'this week', or 'recent'
    - Prefer concise queries (3-7 words) for more targeted results

    Args:
        query (str): A specific search query. Examples:
            - "latest AI research trends"
            - "current weather in New York"
            - "recent space exploration news"
        max_results (int): Maximum number of results to return (default: 5)

    Returns:
        str: Formatted search results with title, snippet, and URL
    """
    logger.info(f"Searching for: {query}")

    if not os.getenv("TAVILY_API_KEY", None):
        error_msg = "Tavily API key not configured. Please set TAVILY_API_KEY environment variable."
        logger.error(error_msg)
        return error_msg

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": os.getenv("TAVILY_API_KEY", None),
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                    "include_images": False,
                    "search_depth": "basic"
                },
                timeout=30.0
            )

            data = response.json()

            # Format results
            results = []

            # Include the answer if available
            if data.get("answer"):
                results.append(f"**Answer**: {data['answer']}")
                results.append("")  # Empty line

            # Format search results
            for i, result in enumerate(data.get("results", []), 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "")
                url = result.get("url", "")

                results.append(f"{i}. **{title}**")
                results.append(f"   {snippet}")
                results.append(f"   URL: {url}")
                results.append("")  # Empty line between results

            formatted_results = "\n".join(results) if results else "No results found."
            logger.info(f"Found {len(data.get('results', []))} results")
            return formatted_results

    except httpx.HTTPError as e:
        error_msg = f"Search request failed: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error performing search: {str(e)}"
        logger.error(error_msg)
        return error_msg


# class AgentState(MessagesState, total=False):
#     """`total=False` is PEP589 specs.
#
#     documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
#     """
#
#     safety: LlamaGuardOutput
#     remaining_steps: RemainingSteps


# web_search = TavilySearchResults(k=1)
tools = [web_search, calculator]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """

# def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
#     bound_model = model.bind_tools(tools)
#     preprocessor = RunnableLambda(
#         lambda state: [SystemMessage(content=instructions)] + state["messages"],
#         name="StateModifier",
#     )
#     return preprocessor | bound_model  # type: ignore[return-value]
#
#
# def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
#     content = (
#         f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
#     )
#     return AIMessage(content=content)
#
#
# async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
#     m = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, streaming=True)
#     model_runnable = wrap_model(m)
#     response = await model_runnable.ainvoke(state, config)
#
#     # Run llama guard check here to avoid returning the message if it's unsafe
#     llama_guard = LlamaGuard()
#     safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
#     if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
#         return {"messages": [format_safety_message(safety_output)], "safety": safety_output}
#
#     if state["remaining_steps"] < 2 and response.tool_calls:
#         return {
#             "messages": [
#                 AIMessage(
#                     id=response.id,
#                     content="Sorry, need more steps to process this request.",
#                 )
#             ]
#         }
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}
#
#
# async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
#     llama_guard = LlamaGuard()
#     safety_output = await llama_guard.ainvoke("User", state["messages"])
#     return {"safety": safety_output, "messages": []}
#
#
# async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
#     safety: LlamaGuardOutput = state["safety"]
#     return {"messages": [format_safety_message(safety)]}
#
#
# # Define the graph
# agent = StateGraph(AgentState)
# agent.add_node("model", acall_model)
# agent.add_node("tools", ToolNode(tools))
# agent.add_node("guard_input", llama_guard_input)
# agent.add_node("block_unsafe_content", block_unsafe_content)
# agent.set_entry_point("guard_input")
#
#
# # Check for unsafe input and block further processing if found
# def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
#     safety: LlamaGuardOutput = state["safety"]
#     match safety.safety_assessment:
#         case SafetyAssessment.UNSAFE:
#             return "unsafe"
#         case _:
#             return "safe"
#
#
# agent.add_conditional_edges(
#     "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
# )
#
# # Always END after blocking unsafe content
# agent.add_edge("block_unsafe_content", END)
#
# # Always run "model" after "tools"
# agent.add_edge("tools", "model")
#
#
# # After "model", if there are tool calls, run "tools". Otherwise END.
# def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
#     last_message = state["messages"][-1]
#     if not isinstance(last_message, AIMessage):
#         raise TypeError(f"Expected AIMessage, got {type(last_message)}")
#     if last_message.tool_calls:
#         return "tools"
#     return "done"
#
#
# agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
#
# research_assistant = agent.compile(checkpointer=MemorySaver(), store=InMemoryStore())

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, streaming=True)

research_assistant = create_react_agent(
    model=model,
    tools=[calculator, web_search],
    store=InMemoryStore(),
    checkpointer=MemorySaver())

# async def main():
#     """Main function to test the research assistant."""
#
#     # Create a thread ID for conversation continuity
#     thread_id = str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}
#     logger.info(f"config: {config}")
#
#     from langgraph.prebuilt import create_react_agent
#
#     ag = create_react_agent(model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, streaming=True),
#                             tools=[calculator, web_search], store=InMemoryStore())
#
#     ag.get_graph().draw_png("react_agent_diagram.png")
#
#     research_assistant.get_graph().draw_png("research_agent_diagram.png")
#
#     # Test queries
#     test_queries = [
#         # Math question
#         "What is 300 * 200?",
#
#         # # Web search question
#         # "What are the latest developments in artificial intelligence?",
#     ]
#
#     logger.info("Testing Research Assistant with Safety Guardrails\n")
#     logger.info("=" * 50)
#
#     for i, query in enumerate(test_queries, 1):
#         logger.info(f"Test {i}: {query}")
#         logger.info("-" * 30)
#
#         try:
#             # Create input state
#             input_state = {
#                 "messages": [HumanMessage(content=query)],
#                 "remaining_steps": 10  # Set remaining steps
#             }
#
#             # Run the agent
#             async for event in research_assistant.astream(
#                     input_state,
#                     config=config,
#                     stream_mode="values"
#             ):
#                 logger.info("\n")
#                 logger.info(f"Event {event}")
#                 # Get the last message
#                 if event.get("messages"):
#                     messages = event['messages']
#                     for message in messages:
#                         logger.info(f"Message: {message}")
#
#                     # Check if safety was triggered
#                     if event.get("safety"):
#                         safety = event["safety"]
#                         logger.info(f"Safety Assessment: {safety}")
#                         if safety.safety_assessment == "unsafe":
#                             logger.info(f"Safety Alert: Unsafe categories - {safety.unsafe_categories}")
#
#         except Exception as e:
#             logger.info(f"Error: {str(e)}")
#
#         logger.info("-" * 30)
#
#
# if __name__ == "__main__":
#     # Run the main test
#     asyncio.run(main())
