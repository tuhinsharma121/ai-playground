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

@asynccontextmanager
async def get_research_assistant():
    mcp_bmi_host = os.getenv("MCP_BMI_HOST", "0.0.0.0")
    mcp_email_host = os.getenv("MCP_EMAIL_HOST", "0.0.0.0")
    mcp_websearch_host = os.getenv("MCP_WEBSEARCH_HOST", "0.0.0.0")

    mcp_bmi_port = os.getenv("MCP_BMI_PORT", "1002")
    mcp_email_port = os.getenv("MCP_EMAIL_PORT", "2002")
    mcp_websearch_port = os.getenv("MCP_WEBSEARCH_PORT", "3002")

    async with initialize_database() as saver, initialize_store() as store:
        client = MultiServerMCPClient(
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
        )

        tools = await client.get_tools()
        # Set up both components
        if hasattr(saver, "setup"):
            logger.info("Setting up saver")# ignore: union-attr
            await saver.setup()
        # Only setup store for Postgres as InMemoryStore doesn't need setup
        if hasattr(store, "setup"):
            logger.info("Setting up store")# ignore: union-attr
            await store.setup()
        agent = create_react_agent(
            model=model,
            tools=tools,
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
            "what is the stock price of IBM? check websearch",

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
