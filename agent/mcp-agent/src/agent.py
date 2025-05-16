import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from constants import constants
from pylogger import get_python_logger
from src.memory import initialize_database, initialize_store

logger = get_python_logger(log_level=constants.LOG_LEVEL)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, streaming=True)

current_date = datetime.now().strftime("%B %d, %Y")

instructions = f"""
    You are a helpful research assistant with the ability to use other tools. 
    Your name is Red Hat and you are extremely intelligent.
    
    Today's date is {current_date}.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Only use the tools you are given to answer the users question. Do not answer directly from internal knowledge.
    - You must always reason before acting.
    - Every Final Answer must be grounded in tool observations.
    - always make sure your answer is *FORMATTED WELL*
    - Show how you are thinking and reasoning step-by-step and then respond with Final answer.
    """

# instructions = f"""
# You areToday's date is {current_date}.
# You reason step-by-step to understand the user’s query, choose the correct tool, and use tool outputs to form your response. You always follow the pattern:
# **Thought → Action → Observation → (repeat if needed) → Final Answer**.
#
# You have access to the following tools:
#
# 1. **Email Agent Tool**
#    - Use this tool **only if the user explicitly requests to send an email**.
#    - If the name of the recipient is not mentioned then send the email to Tuhin.
#    - Remember your name and signature are "Fedora" when you send an email.
#    - This tool requires an email subject and an email body.
#    - You must use information from the other tools to help compose the message.
#
# 2. **BMI Agent Tool**
#    - Use this tool the user asks about calculating BMI.
#    - This tool requires a height in cms and weight in kgs for a person. If not provided use other tools to infer it.
#    - You can use information from the other tools if height and weight are not provided.
#
# 3. **Websearch Tool**
#    - Use this when the user question cannot be answered by the Christmas or Medium or BMI tools.
#    - Only call this when the topic is outside the coverage of the other three tools.
#
# ### 🔐 Rules:
# - DO NOT answer directly from internal knowledge.
# - You must always reason before acting.
# - Every Final Answer must be grounded in tool observations.
# - If tool output is insufficient, continue the loop with another Thought → Action or ask the user for more information.
# - Only send emails when clearly instructed by the user.
# - Please include markdown-formatted links to any citations used in your response.
# Only include one or two citations per response unless more are needed.
# always make sure your answer is *FORMATTED WELL*
#
# Show how you are thinking and reasoning step-by-step and then respond with Final answer.
#
# """


@asynccontextmanager
async def get_research_assistant():
    mcp_bmi_host = os.getenv("MCP_BMI_HOST", "0.0.0.0")
    mcp_email_host = os.getenv("MCP_EMAIL_HOST", "0.0.0.0")
    mcp_websearch_host = os.getenv("MCP_WEBSEARCH_HOST", "0.0.0.0")

    mcp_bmi_port = os.getenv("MCP_BMI_PORT", "1002")
    mcp_email_port = os.getenv("MCP_EMAIL_PORT", "2002")
    mcp_websearch_port = os.getenv("MCP_WEBSEARCH_PORT", "3002")

    async with MultiServerMCPClient(
            {
                "bmi_agent_tool": {
                    "url": f"http://{mcp_bmi_host}:{mcp_bmi_port}/sse",
                    "transport": "sse"
                },
                "websearch_agent_tool": {
                    "url": f"http://{mcp_websearch_host}:{mcp_websearch_port}/sse",
                    "transport": "sse"
                },
                "email_agent_tool": {
                    "url": f"http://{mcp_email_host}:{mcp_email_port}/sse",
                    "transport": "sse"
                }
            }
    ) as client, initialize_database() as saver, initialize_store() as store:
        # Set up both components
        if hasattr(saver, "setup"):  # ignore: union-attr
            await saver.setup()
        # Only setup store for Postgres as InMemoryStore doesn't need setup
        if hasattr(store, "setup"):  # ignore: union-attr
            await store.setup()
        agent = create_react_agent(
            model=model,
            tools=client.get_tools(),
            prompt=instructions,
            store=store,
            checkpointer=saver
        )
        yield agent


async def main():
    """Main function to test the research assistant."""

    # Create a thread ID for conversation continuity
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"config: {config}")

    async with get_research_assistant() as research_assistant:
        research_assistant.get_graph().draw_png("research_agent_diagram.png")

        # Test queries
        test_queries = [
            # Math question
            "What is 300 * 200?",

            # # Web search question
            # "What are the latest developments in artificial intelligence?",
        ]

        logger.info("Testing Research Assistant with Safety Guardrails\n")
        logger.info("=" * 50)

        for i, query in enumerate(test_queries, 1):
            logger.info(f"Test {i}: {query}")
            logger.info("-" * 30)

            try:
                # Create input state
                input_state = {
                    "messages": [HumanMessage(content=query)],
                    "remaining_steps": 10  # Set remaining steps
                }

                # Run the agent
                async for event in research_assistant.astream(
                        input_state,
                        config=config,
                        stream_mode="values"
                ):
                    logger.info("\n")
                    logger.info(f"Event {event}")
                    # Get the last message
                    if event.get("messages"):
                        messages = event['messages']
                        for message in messages:
                            logger.info(f"Message: {message}")

                        # Check if safety was triggered
                        if event.get("safety"):
                            safety = event["safety"]
                            logger.info(f"Safety Assessment: {safety}")
                            if safety.safety_assessment == "unsafe":
                                logger.info(f"Safety Alert: Unsafe categories - {safety.unsafe_categories}")

            except Exception as e:
                logger.info(f"Error: {str(e)}")

            logger.info("-" * 30)


if __name__ == "__main__":
    # Run the main test
    asyncio.run(main())
