from typing import Any, Literal
from typing import NotRequired

from pydantic import BaseModel, Field
from typing_extensions import TypedDict



class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    session_id: str | None = Field(
        description="Session ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


class TaskData(BaseModel):
    name: str | None = Field(
        description="Name of the task.", default=None, examples=["Check input safety"]
    )
    run_id: str = Field(
        description="ID of the task run to pair state updates to.",
        default="",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    state: Literal["new", "running", "complete"] | None = Field(
        description="Current state of given task instance.",
        default=None,
        examples=["running"],
    )
    result: Literal["success", "error"] | None = Field(
        description="Result of given task instance.",
        default=None,
        examples=["running"],
    )
    data: dict[str, Any] = Field(
        description="Additional data generated by the task.",
        default={},
    )

    def completed(self) -> bool:
        return self.state == "complete"

    def completed_with_error(self) -> bool:
        return self.state == "complete" and self.result == "error"


class TaskDataStatus:
    def __init__(self) -> None:
        import streamlit as st

        self.status = st.status("")
        self.current_task_data: dict[str, TaskData] = {}

    def add_and_draw_task_data(self, task_data: TaskData) -> None:
        status = self.status
        status_str = f"Task **{task_data.name}** "
        match task_data.state:
            case "new":
                status_str += "has :blue[started]. Input:"
            case "running":
                status_str += "wrote:"
            case "complete":
                if task_data.result == "success":
                    status_str += ":green[completed successfully]. Output:"
                else:
                    status_str += ":red[ended with error]. Output:"
        status.write(status_str)
        status.write(task_data.data)
        status.write("---")
        if task_data.run_id not in self.current_task_data:
            # Status label always shows the last newly started task
            status.update(label=f"""Task: {task_data.name}""")
        self.current_task_data[task_data.run_id] = task_data
        if all(entry.completed() for entry in self.current_task_data.values()):
            # Status is "error" if any task has errored
            if any(entry.completed_with_error() for entry in self.current_task_data.values()):
                state = "error"
            # Status is "complete" if all tasks have completed successfully
            else:
                state = "complete"
        # Status is "running" until all tasks have completed
        else:
            state = "running"
        status.update(state=state)  # type: ignore[arg-type]
