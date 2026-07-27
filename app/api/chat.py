from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ollama import ollama_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{model_name}", response_model=ChatResponse)
async def chat(model_name: str, request: ChatRequest) -> ChatResponse:
    request.model = model_name
    return await ollama_service.generate(request)


@router.post("/{model_name}/stream")
async def chat_stream(model_name: str, request: ChatRequest) -> StreamingResponse:
    request.model = model_name
    return StreamingResponse(
        ollama_service.generate_stream(request),
        media_type="text/event-stream",
    )
