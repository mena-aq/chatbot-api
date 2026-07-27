from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.core.exceptions import (
    OllamaAPIError,
    OllamaConnectionError,
    OllamaModelError,
    ollama_api_error_handler,
    ollama_connection_error_handler,
    ollama_model_error_handler,
)

app = FastAPI(title="FTune Customer Service API", version="0.1.0")

app.add_exception_handler(OllamaConnectionError, ollama_connection_error_handler)
app.add_exception_handler(OllamaModelError, ollama_model_error_handler)
app.add_exception_handler(OllamaAPIError, ollama_api_error_handler)

app.include_router(chat_router)
