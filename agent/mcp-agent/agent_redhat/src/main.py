import inspect
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt

from agent_redhat.src.agent import get_agent_redhat
from agent_redhat.src.constants import constants
from utils.agent_utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from utils.pylogger import get_python_logger
from utils.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput, )

logger = get_python_logger(constants.LOG_LEVEL)

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler(trace_name="agent-redhat")

client = Langfuse()


def verify_bearer(
        http_auth: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
        ],
) -> None:
    if not constants.AUTH_SECRET:
        return
    auth_secret = constants.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer and store
    based on settings.
    """
    try:

        async with get_agent_redhat() as agent:
            yield
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])


async def _handle_input(user_input: UserInput, agent: Pregel) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    session_id = user_input.session_id or str(uuid4())

    configurable = {"thread_id": thread_id, "session_id": session_id, "langfuse_session_id": session_id,
                    "message_id": run_id}

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=[langfuse_handler],
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


async def message_generator(
        user_input: StreamInput
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    async with get_agent_redhat() as agent:
        # agent: Pregel = get_research_assistant()
        kwargs, run_id = await _handle_input(user_input, agent)

        try:
            logger.info(f"Running agent with kwargs: {kwargs}")
            # Process streamed events from the graph and yield messages over the SSE stream.
            async for stream_event in agent.astream(
                    **kwargs, stream_mode=["updates", "messages", "custom"]
            ):
                if not isinstance(stream_event, tuple):
                    continue
                stream_mode, event = stream_event
                logger.info(f"Stream mode: {stream_mode}, event: {event}")
                new_messages = []
                if stream_mode == "updates":
                    for node, updates in event.items():
                        # A simple approach to handle agent interrupts.
                        # In a more sophisticated implementation, we could add
                        # some structured ChatMessage type to return the interrupt value.
                        if node == "__interrupt__":
                            interrupt: Interrupt
                            for interrupt in updates:
                                new_messages.append(AIMessage(content=interrupt.value))
                            continue
                        updates = updates or {}
                        update_messages = updates.get("messages", [])
                        # special cases for using langgraph-supervisor library
                        if node == "supervisor":
                            # Get only the last AIMessage since supervisor includes all previous messages
                            ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                            if ai_messages:
                                update_messages = [ai_messages[-1]]
                        if node in ("research_expert", "math_expert"):
                            # By default the sub-agent output is returned as an AIMessage.
                            # Convert it to a ToolMessage so it displays in the UI as a tool response.
                            msg = ToolMessage(
                                content=update_messages[0].content,
                                name=node,
                                tool_call_id="",
                            )
                            update_messages = [msg]
                        new_messages.extend(update_messages)

                if stream_mode == "custom":
                    new_messages = [event]

                # LangGraph streaming may emit tuples: (field_name, field_value)
                # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
                # We accumulate only supported fields into `parts` and skip unsupported metadata.
                # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
                processed_messages = []
                current_message: dict[str, Any] = {}
                for message in new_messages:
                    if isinstance(message, tuple):
                        key, value = message
                        # Store parts in temporary dict
                        current_message[key] = value
                    else:
                        # Add complete message if we have one in progress
                        if current_message:
                            processed_messages.append(_create_ai_message(current_message))
                            current_message = {}
                        processed_messages.append(message)

                # Add any remaining message parts
                if current_message:
                    processed_messages.append(_create_ai_message(current_message))

                for message in processed_messages:
                    try:
                        chat_message = langchain_to_chat_message(message)
                        chat_message.run_id = str(run_id)
                    except Exception as e:
                        logger.error(f"Error parsing message: {e}")
                        yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                        continue
                    # LangGraph re-sends the input message, which feels weird, so drop it
                    if chat_message.type == "human" and chat_message.content == user_input.message:
                        continue
                    yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

                if stream_mode == "messages":
                    if not user_input.stream_tokens:
                        continue
                    msg, metadata = event
                    if "skip_stream" in metadata.get("tags", []):
                        continue
                    # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                    # Drop them.
                    if not isinstance(msg, AIMessageChunk):
                        continue
                    content = remove_tool_calls(msg.content)
                    if content:
                        # Empty content in the context of OpenAI usually means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content.
                        yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
        except Exception as e:
            logger.error(f"Error in message generator: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use session_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to Langfuse.

    This is a simple wrapper for the Langfuse create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """

    kwargs = feedback.kwargs or {}

    # Langfuse uses different parameter names
    client.score(
        trace_id=feedback.run_id,  # Assuming run_id maps to trace_id
        name=feedback.key,  # 'key' becomes 'name' in Langfuse
        value=feedback.score,  # 'score' becomes 'value' in Langfuse
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    async with get_agent_redhat() as agent:
        try:
            state_snapshot = await agent.aget_state(
                config=RunnableConfig(configurable={"thread_id": input.thread_id})
            )
            messages: list[AnyMessage] = state_snapshot.values["messages"]
            chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
            return ChatHistory(messages=chat_messages)
        except Exception as e:
            logger.error(f"An exception occurred: {e}")
            raise HTTPException(status_code=500, detail="Unexpected error")


# Then expose this in your API
# @router.get("/threads")
# async def list_threads() -> list[str]:
#     """
#     Get a list of all thread IDs in the system.
#     """
#     try:
#         # Connect to the SQLite database
#         with sqlite3.connect(constants.SQLITE_DB_PATH) as conn:
#             cursor = conn.cursor()
#
#             # First, let's check what tables exist
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#             tables = [table[0] for table in cursor.fetchall()]
#
#             logger.info(f"Tables: {tables}")
#
#             if 'checkpoints' not in tables:
#                 logger.warning("Checkpoints table not found in SQLite database")
#                 return []
#
#             # Examine the schema of the checkpoints table
#             cursor.execute("PRAGMA table_info(checkpoints);")
#             columns = [column[1] for column in cursor.fetchall()]
#
#             logger.info(f"Columns: {columns}")
#
#             # Now query based on the actual schema
#             if 'thread_id' in columns:
#                 cursor.execute("SELECT DISTINCT thread_id FROM checkpoints;")
#                 keys = [row[0] for row in cursor.fetchall()]
#
#                 # Try to extract thread IDs from keys
#                 thread_ids = set()
#                 for key in keys:
#                     # Assuming the format is typically 'thread_id:...'
#                     parts = key.split(':', 1)
#                     if len(parts) > 0:
#                         thread_ids.add(parts[0])
#
#                 return list(thread_ids)
#             else:
#                 logger.warning("Could not find key column in checkpoints table")
#                 return []
#     except Exception as e:
#         logger.error(f"An exception occurred: {e}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/threads")
async def list_threads() -> list[str]:
    """
    Get a list of all thread IDs in the system.
    """
    try:
        # Connect to the SQLite database
        import psycopg2
        from agent_redhat.src.memory import get_postgres_connection_string
        with psycopg2.connect(get_postgres_connection_string()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT distinct thread_id FROM checkpoints")
            rows = cur.fetchall()
            thread_ids = [row[0] for row in rows]
            return thread_ids
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, )
