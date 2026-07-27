from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query_id: str = Field(..., description="Unique identifier for this query")
    messages: list[ChatMessage]
    model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ChatResponse(BaseModel):
    query_id: str
    reply: str
    model: str
    input_tokens: int | None = None  # prompt eval count
    output_tokens: int | None = None  # eval count
    total_duration_ms: float | None = None
