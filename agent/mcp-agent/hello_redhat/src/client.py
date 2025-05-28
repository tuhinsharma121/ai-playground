import asyncio
import os
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent

from pylogger import get_python_logger

logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Initialize gemini-2.0-flash model
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_retries=10)

# # Initialize o4-mini model
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="o3-mini", max_retries=10)

react_prompt = """

You are a helpful and intelligent assistant that NEVER answers questions directly from your own knowledge. 

You reason step-by-step to understand the user’s query, choose the correct tool, and use tool outputs to form your response. You always follow the pattern:
**Thought → Action → Observation → (repeat if needed) → Final Answer**.

You have access to the following tools:

1. **Email Agent Tool**
   - Use this tool **only if the user explicitly requests to send an email**.
   - If the name of the recipient is not mentioned then send the email to Tuhin.
   - Remember your name and signature are "Fedora" when you send an email.
   - This tool requires an email subject and an email body.
   - You must use information from the other tools to help compose the message.

2. **BMI Agent Tool**
   - Use this tool the user asks about calculating BMI.
   - This tool requires a height in cms and weight in kgs for a person. If not provided use other tools to infer it.
   - You can use information from the other tools if height and weight are not provided.

3. **Websearch Tool**
   - Use this when the user question cannot be answered by the Christmas or Medium or BMI tools.
   - Only call this when the topic is outside the coverage of the other three tools.

### 🔐 Rules:
- DO NOT answer directly from internal knowledge.
- You must always reason before acting.
- Every Final Answer must be grounded in tool observations.
- If tool output is insufficient, continue the loop with another Thought → Action.
- Only send emails when clearly instructed by the user.

The response must contain the entire chain of thought, process and reasoning step-by-step 
with nuanced detail without skipping or summarizing any step as:
**Thought → Action → Observation → (repeat if needed) → Final Answer**. 
The steps should be as detailed and nuanced as possible.
Each of these steps must be separated by **two \n\n character**.

"""


@asynccontextmanager
async def main():
    mcp_bmi_host = os.getenv("MCP_BMI_HOST", "0.0.0.0")
    mcp_email_host = os.getenv("MCP_EMAIL_HOST", "0.0.0.0")
    mcp_websearch_host = os.getenv("MCP_WEBSEARCH_HOST", "0.0.0.0")

    mcp_bmi_port = os.getenv("MCP_BMI_PORT", "1002")
    mcp_email_port = os.getenv("MCP_EMAIL_PORT", "3002")
    mcp_websearch_port = os.getenv("MCP_WEBSEARCH_PORT", "5002")

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
    ) as client:
        agent = create_react_agent(
            llm,
            tools=client.get_tools(),
            prompt=react_prompt
        )
        yield agent


async def invoke_agent(query):
    async with main() as agent:
        agent_response = await agent.ainvoke({"messages": query})
        logger.info("==== Final Answer ====")
        logger.info(agent_response['messages'][-1].content)


if __name__ == "__main__":
    # You can use any of these queries

    # query = """
    #         How does Dickens establish Scrooge's character through
    #         environmental imagery rather than direct description?
    #         """
    # query = """What is the use of langchain?"""
    # query = """What is the stock price of IBM today?"""
    # query = """What is Tuhin's opinion about Hollywood?"""
    # query = """Did Charles Dickens use langchain in Christmas Carol?"""

    # query = "send an email to me with the latest details of IBM stock price."

    # query = "I am a healthy male of age 50. My height is 1.86 meter and weight is 70 kg. Send me my BMI analysis in my inbox."

    query = "send me the current BMI report via email of the lead actor/actress in the Netflix movie Kumari"

    asyncio.run(invoke_agent(query=query))

    # import google.generativeai as genai
    #
    # client = genai.GenerativeModel(
    #     model_name="gemini-2.0-flash")
    #
    # response = client.generate_content(
    #     contents="Explain how AI works in detail",
    # )
    #
    # print(response.text)
