from typing import Literal, Optional

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel


class Message(BaseModel):
    content: Optional[str] = None
    """The contents of the message."""

    role: Literal["user", "assistant", "system", "tool", "model"] = "tool"
    """The role of the author of this message."""

    tool_calls: Optional[list[ChatCompletionMessageToolCall]] = None
    """The tool calls generated by the model, such as function calls."""

    tool_call_id: Optional[str] = None


class ToolCall(BaseModel):
    id: Optional[str] = None
    function: str
    args: str


class Usage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage"):
        return Usage(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
