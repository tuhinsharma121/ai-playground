import asyncio
import uuid
from collections.abc import AsyncGenerator

import streamlit as st

from pylogger import get_python_logger
from src.client import AgentClient, AgentClientError
from src.schema import ChatHistory, ChatMessage

logger = get_python_logger()
# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Hello Red Hat"
# APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"

# Replace the emoji icon with custom icon
# APP_ICON = "🧰"  # Comment out or remove this line

# Add this near the top of your file, after the imports
import base64


def get_img_as_base64(file_path):
    """Convert an image file to base64 string"""
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Function to display image in sidebar header
def get_image_html(image_path, width=30):
    """Create HTML for displaying an image"""
    return f'<img src="data:image/png;base64,{get_img_as_base64(image_path)}" width="{width}">'


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        # page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient()
        except AgentClientError as e:
            st.error(f"Error connecting to agent service : {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            logger.info(f"Thread ID not provided. Using {thread_id}.")
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        # st.header(f"{APP_ICON} {APP_TITLE}")

        icon_path = "fedora.png"  # Update with your actual icon path
        # Custom CSS to center and align the header
        st.markdown(
            """
            <style>
            .app-header {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            }
            .app-header img {
                margin-right: 10px;
            }
            .app-header h1 {
                font-size: 24px;
                margin: 0;
                padding: 0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # App header with icon and title
        st.markdown(
            f"""
                <div class="app-header">
                    {get_image_html(icon_path, width=200)}
                </div>
                <div class="app-header">
                    <h1>{APP_TITLE}</h1>
                </div>
                """,
            unsafe_allow_html=True
        )

        # Subtitle text - centered
        st.markdown(
            """
            <p style="text-align: center;">
            Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit
            </p>
            """,
            unsafe_allow_html=True
        )

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        # Display chat history in sidebar
        st.subheader("Chat History")

        # Initialize chat_history in session state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {}

        # Store current thread in chat history if it has messages
        if len(st.session_state.messages) > 0:
            # Generate a title from the first user message or use default
            thread_title = "New Chat"
            for msg in st.session_state.messages:
                if msg.type == "human":
                    thread_title = msg.content[:20] + "..." if len(msg.content) > 20 else msg.content
                    break

            # Store or update in chat history
            st.session_state.chat_history[st.session_state.thread_id] = {
                "title": thread_title,
                "last_updated": asyncio.get_event_loop().time(),
            }

        # Get all thread IDs from the backend
        try:
            # Fetch all thread IDs
            all_thread_ids = agent_client.get_all_thread_ids()

            # If this is the first load and there are threads, fetch the first message for each thread
            # to generate proper titles (only for threads not in our cache)
            if "first_load" not in st.session_state and all_thread_ids:
                st.session_state.first_load = False

                # Limit how many threads we prefetch to avoid performance issues (e.g., fetch at most 10)
                threads_to_fetch = [tid for tid in all_thread_ids[:10] if tid not in st.session_state.chat_history]

                # Fetch first message for each thread
                for thread_id in threads_to_fetch:
                    try:
                        # We might want to add a method to just get the first few messages to improve performance
                        thread_history = agent_client.get_history(thread_id=thread_id)
                        thread_messages = thread_history.messages

                        # Generate a title from the first user message
                        thread_title = f"Chat {thread_id[:8]}..."  # Default fallback
                        for msg in thread_messages:
                            if msg.type == "human":
                                thread_title = msg.content[:20] + "..." if len(msg.content) > 20 else msg.content
                                break

                        # Store in chat history
                        st.session_state.chat_history[thread_id] = {
                            "title": thread_title,
                            "last_updated": 0,  # We don't know the actual timestamp, so use 0
                        }
                    except AgentClientError:
                        # If we can't load this thread, just skip it
                        continue

            # Display chat history
            if all_thread_ids:
                for thread_id in all_thread_ids:
                    # If we have info about this thread in the session, use it
                    if thread_id in st.session_state.chat_history:
                        thread_info = st.session_state.chat_history[thread_id]
                        thread_title = thread_info.get("title", f"Chat {thread_id[:8]}...")
                    else:
                        # For threads we haven't seen yet, use a default title
                        thread_title = f"Chat {thread_id[:8]}..."

                    # Mark current thread with a different icon or style
                    if thread_id == st.session_state.thread_id:
                        # Current conversation - show as active/selected
                        st.button(f"🟢 {thread_title}", key=f"history_{thread_id}", use_container_width=True,
                                  disabled=True)
                    else:
                        # Other conversations - clickable to switch
                        if st.button(f"💬 {thread_title}", key=f"history_{thread_id}", use_container_width=True):
                            # Switch to this thread
                            st.session_state.thread_id = thread_id
                            try:
                                # Get thread history
                                history = agent_client.get_history(thread_id=thread_id)
                                st.session_state.messages = history.messages

                                # Update the title in our local cache based on actual messages
                                thread_title = "Chat"
                                for msg in st.session_state.messages:
                                    if msg.type == "human":
                                        thread_title = msg.content[:20] + "..." if len(
                                            msg.content) > 20 else msg.content
                                        break

                                st.session_state.chat_history[thread_id] = {
                                    "title": thread_title,
                                    "last_updated": asyncio.get_event_loop().time(),
                                }

                                st.rerun()
                            except AgentClientError as e:
                                st.error(f"Could not load chat history: {e}")
            else:
                st.info("No previous chats")
        except AgentClientError as e:
            st.error(f"Could not fetch chat history: {e}")

        st.divider()

        # Additional sidebar options
        agent_client.agent = "research-assistant"
        use_streaming = True

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
        messages_agen: AsyncGenerator[ChatMessage | str, None],
        is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Remote MCP Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
