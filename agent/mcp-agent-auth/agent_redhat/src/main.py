import inspect
import json
import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import psycopg2
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from jwt import PyJWKClient
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt
from requests_oauthlib import OAuth2Session
from starlette.middleware.sessions import SessionMiddleware

from agent_redhat.src.agent import get_agent_redhat
from utils.agent_utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from utils.constants import constants
from utils.memory import get_postgres_connection_string
from utils.pylogger import get_python_logger
from utils.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput, )

# OAuth2 scheme for authorization
oauth_2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{constants.JWT_SSO_BASE_URL}/protocol/openid-connect/auth",
    tokenUrl=f"{constants.JWT_SSO_BASE_URL}/protocol/openid-connect/token",
)

# Initialize the JWK client
url = f"{constants.JWT_SSO_BASE_URL}/protocol/openid-connect/certs"
optional_custom_headers = {"User-agent": "custom-user-agent"}
jwks_client = PyJWKClient(url, headers=optional_custom_headers)

SSO_CALLBACK_URL="http://0.0.0.0:8000/callback"
SCOPE = ['session:role:HELLO_REDHAT_GROUP', 'refresh_token']

class OAuth2Handler:
    """OAuth2 handler class - equivalent to fastify.customOAuth2"""

    @staticmethod
    def create_oauth_session(state=None):
        return OAuth2Session(
            constants.SSO_CLIENT_ID,
            scope=SCOPE,
            redirect_uri=SSO_CALLBACK_URL,
            state=state
        )

    @staticmethod
    def get_authorization_url():
        oauth = OAuth2Handler.create_oauth_session()
        authorization_url, state = oauth.authorization_url(constants.AUTHORIZATION_BASE_URL)
        return authorization_url, state

    @staticmethod
    def get_access_token_from_authorization_code_flow(code: str, state: str):
        oauth = OAuth2Handler.create_oauth_session(state=state)
        token = oauth.fetch_token(
            constants.TOKEN_URL,
            code=code,
            client_secret=constants.SSO_CLIENT_SECRET,
            include_client_id=True
        )
        return token


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
        app.logger.error(f"Error during database/store initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware (equivalent to Fastify session)
app.add_middleware(SessionMiddleware, secret_key=os.getenv('SESSION_SECRET', secrets.token_hex(32)))

app.logger = get_python_logger(constants.LOG_LEVEL)

# Initialize Langfuse CallbackHandler for Langchain (tracing)
app.langfuse_handler = CallbackHandler(trace_name="agent-redhat", environment=constants.LANGFUSE_TRACING_ENVIRONMENT)

app.client = Langfuse(environment=constants.LANGFUSE_TRACING_ENVIRONMENT)

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def add_user_id_to_callback(auth_url, user_id):
    # Parse the authorization URL
    parsed_url = urlparse(auth_url)

    # Parse query parameters
    query_params = parse_qs(parsed_url.query)

    # Get the redirect_uri and decode it
    redirect_uri = query_params['redirect_uri'][0]

    # Parse the redirect URI
    parsed_redirect = urlparse(redirect_uri)
    redirect_query_params = parse_qs(parsed_redirect.query)

    # Add user_id to the callback URL parameters
    redirect_query_params['user_id'] = [user_id]

    # Reconstruct the redirect URI
    new_redirect_query = urlencode(redirect_query_params, doseq=True)
    new_redirect_uri = urlunparse((
        parsed_redirect.scheme,
        parsed_redirect.netloc,
        parsed_redirect.path,
        parsed_redirect.params,
        new_redirect_query,
        parsed_redirect.fragment
    ))

    # Update the original URL with the new redirect_uri
    query_params['redirect_uri'] = [new_redirect_uri]
    new_query = urlencode(query_params, doseq=True)

    # Reconstruct the full authorization URL
    new_auth_url = urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))

    return new_auth_url



# Routes - equivalent to your fastify routes
@app.get("/login")
async def login(request: Request, user_id: str):
    """Start redirect path - equivalent to startRedirectPath in Node.js"""
    authorization_url, state = OAuth2Handler.get_authorization_url()
    updated_url = add_user_id_to_callback(authorization_url, user_id)
    app.logger.info(updated_url)
    app.logger.info(authorization_url)
    app.logger.info(state)
    # autorization_url  = authorization_url + "?username=tuhin"
    # Store state in session for security
    request.session['oauth_state'] = state

    return RedirectResponse(url=updated_url)


app.fake_db = dict()


@app.get("/callback")
async def callback(request: Request, user_id: str, code: str = None):
    """OAuth callback - equivalent to your fastify.get('/callback')"""

    app.logger.info(f"Code from auth: {code}")
    app.logger.info(user_id)
    app.logger.info(request.session)



    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")

    try:
        # Get state from session
        state = request.session.get('oauth_state')

        # Get access token - equivalent to getAccessTokenFromAuthorizationCodeFlow
        token_set = OAuth2Handler.get_access_token_from_authorization_code_flow(code, state)

        print(f"Token Set: {token_set}")
        app.fake_db[user_id] = token_set

        # Store user in session - equivalent to request.session.user = tokenSet
        request.session['user'] = token_set

        # Redirect to home - equivalent to reply.redirect("/")
        return RedirectResponse(url="/")

    except Exception as error:
        print(f"Error: {error}")
        # Still send the code as in original - equivalent to reply.send({ access_token: code })
        return JSONResponse(content={"access_token": code})


# This route serves as the heartbeat or health check endpoint for the application.
@app.get("/")
async def home(request: Request):
    """Home route to check authentication status"""
    user = request.session.get('user')
    if user:
        return JSONResponse(content={
            "message": "Authenticated successfully",
            "user": user
        })
    else:
        return JSONResponse(content={
            "message": "Not authenticated",
            "login_url": "/login"
        })


@app.get("/logout")
async def logout(request: Request):
    """Logout route"""
    request.session.pop('user', None)
    request.session.pop('oauth_state', None)
    return RedirectResponse(url="/")


# Helper function to get authenticated user
def get_current_user(request: Request):
    """Dependency to get current authenticated user"""
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


router = APIRouter()


# Example of using the authenticated user in other routes
@router.get("/protected")
async def protected_route(current_user=Depends(get_current_user)):
    """Example protected route that requires authentication"""
    return JSONResponse(content={
        "message": "This is a protected route",
        "user": current_user
    })


async def _handle_input(user_input: UserInput, agent: Pregel) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    session_id = user_input.session_id or str(uuid4())
    user_id = user_input.user_id

    configurable = {"thread_id": thread_id,
                    "session_id": session_id,
                    "message_id": run_id,
                    "user_id": user_id,
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                    "langfuse_observation_id": thread_id}

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
        callbacks=[app.langfuse_handler],
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
    sf_access_token = app.fake_db.get(user_input.user_id)
    app.logger.info(sf_access_token)
    async with get_agent_redhat(sf_access_token) as agent:
        kwargs, run_id = await _handle_input(user_input, agent)
        try:
            app.logger.info(f"Running agent with kwargs: {kwargs}")
            # Process streamed events from the graph and yield messages over the SSE stream.
            async for stream_event in agent.astream(
                    **kwargs, stream_mode=["updates", "messages", "custom"]
            ):
                if not isinstance(stream_event, tuple):
                    continue
                stream_mode, event = stream_event
                app.logger.info(f"Stream mode: {stream_mode}, event: {event}")
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
                        app.logger.error(f"Error parsing message: {e}")
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
            app.logger.error(f"Error in message generator: {e}")
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
    app.client.score(
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
            app.logger.error(f"An exception occurred: {e}")
            raise HTTPException(status_code=500, detail="Unexpected error")


@router.get("/threads/{user_id}")
async def list_threads(user_id: str) -> list[str]:
    """
    Get a list of all thread IDs in the system.
    """
    try:
        # Connect to the SQLite database
        with psycopg2.connect(get_postgres_connection_string()) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT distinct thread_id FROM checkpoints where metadata->>'user_id'='{user_id}'")
            rows = cur.fetchall()
            thread_ids = [row[0] for row in rows]
            return thread_ids
    except Exception as e:
        app.logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
