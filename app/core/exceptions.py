from fastapi import Request
from fastapi.responses import JSONResponse


class OllamaConnectionError(Exception):
    """Raised when the Ollama server is unreachable."""


class OllamaModelError(Exception):
    """Raised when the requested model is not available on Ollama."""


class OllamaAPIError(Exception):
    """Raised when Ollama returns a non-success response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def ollama_connection_error_handler(_req: Request, exc: OllamaConnectionError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "ollama_unavailable", "detail": str(exc)})


async def ollama_model_error_handler(_req: Request, exc: OllamaModelError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "model_not_found", "detail": str(exc)})


async def ollama_api_error_handler(_req: Request, exc: OllamaAPIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": "ollama_error", "detail": exc.detail})
