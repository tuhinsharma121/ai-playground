import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from pylogger import get_python_logger

# Set up logging
logger = get_python_logger(log_level=os.getenv("PYTHON_LOG_LEVEL", "INFO"))

# Create FastAPI app
app = FastAPI(title="Fedora Agent API", description="API for a Fedora Agent")


# Pydantic model for query requests
class QueryRequest(BaseModel):
    """Query request"""
    query: str


# Pydantic model for responses
class QueryResponse(BaseModel):
    """Query response"""
    response: str


# Define the prompt for the React agent
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



# Initialize gemini-2.0-flash model
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_retries=10)

# # Initialize o4-mini model
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="o3-mini", max_retries=10)


@asynccontextmanager
async def get_agent():
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


async def process_query(query: str) -> str:
    """Process a query using the agent and return the final answer."""
    try:
        async with get_agent() as agent:
            agent_response = await agent.ainvoke({"messages": query})
            return agent_response['messages'][-1].content
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Endpoint to process queries through the MCP agent."""
    logger.info(f"Received query: {request.query}")
    response = await process_query(request.query)
    return QueryResponse(response=response)


@app.get("/")
def read_root():
    """Read root"""
    logger.info("Read root")
    return {"Hey": "Fedora Agent"}


# Error handler for unexpected exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )


if __name__ == "__main__":
    logger.info("Starting Fedora Agent server")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
