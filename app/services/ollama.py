import json
import logging
from typing import AsyncIterator

import httpx

from app.core.config import settings
from app.core.exceptions import OllamaAPIError, OllamaConnectionError, OllamaModelError
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=settings.REQUEST_TIMEOUT,
        )

    # ── health ────────────────────────────────────────────────

    async def health_check(self) -> bool:
        async with await self._client() as client:
            try:
                resp = await client.get("/api/tags")
                if resp.status_code != 200:
                    raise OllamaAPIError(resp.status_code, resp.text)
                return True
            except (OllamaAPIError, OllamaConnectionError):
                raise
            except httpx.HTTPError as exc:
                logger.error("Ollama health check failed: %s", exc)
                raise OllamaConnectionError(str(exc)) from exc

    async def ensure_model(self, model_name: str) -> None:
        async with await self._client() as client:
            try:
                resp = await client.get("/api/tags")
                if resp.status_code != 200:
                    raise OllamaAPIError(resp.status_code, resp.text)
                models = [m["name"] for m in resp.json().get("models", [])]
                if not any(model_name in m for m in models):
                    raise OllamaModelError(
                        f"Model '{model_name}' not found on Ollama server. "
                        f"Available: {models}"
                    )
            except (OllamaAPIError, OllamaModelError):
                raise
            except httpx.HTTPError as exc:
                raise OllamaConnectionError(str(exc)) from exc

    # ── synchronous generate ──────────────────────────────────

    async def generate(self, request: ChatRequest) -> ChatResponse:
        messages = self._build_messages(request)
        payload: dict = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature or settings.MODEL_TEMPERATURE,
                "top_p": settings.MODEL_TOP_P,
                "num_predict": request.max_tokens or settings.MODEL_MAX_TOKENS,
                "repeat_penalty": settings.REPEAT_PENALTY,
            },
        }

        async with await self._client() as client:
            try:
                resp = await client.post("/api/chat", json=payload)
                if resp.status_code != 200:
                    raise OllamaAPIError(resp.status_code, resp.text)
            except OllamaAPIError:
                raise
            except httpx.HTTPError as exc:
                raise OllamaConnectionError(str(exc)) from exc

        data = resp.json()
        prompt_eval_count = data.get("prompt_eval_count", 0)  # input tokens
        eval_count = data.get("eval_count", 0)  # output tokens
        return ChatResponse(
            query_id=request.query_id,
            reply=data["message"]["content"],
            model=data.get("model"),
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            total_duration_ms=data.get("total_duration", 0) / 1e6,
        )

    # ── streaming generate ────────────────────────────────────

    async def generate_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        messages = self._build_messages(request)
        payload: dict = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature or settings.MODEL_TEMPERATURE,
                "top_p": settings.MODEL_TOP_P,
                "num_predict": request.max_tokens or settings.MODEL_MAX_TOKENS,
                "repeat_penalty": settings.REPEAT_PENALTY,
            },
        }
        async with await self._client() as client:
            try:
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise OllamaAPIError(resp.status_code, body.decode())
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get("done"):
                            break
                        yield chunk["message"]["content"]
            except OllamaAPIError as exc:
                logger.error("Ollama API error during stream: %s", exc)
                yield f"[error] {exc.detail}"
            except httpx.HTTPError as exc:
                logger.error("Ollama connection error during stream: %s", exc)
                yield f"[error] Ollama unavailable: {exc}"

    # ── embed ─────────────────────────────────────────────────

    async def embed(self, text: str, model_name: str) -> list[float]:
        payload = {"model": model_name, "input": text}
        async with await self._client() as client:
            try:
                resp = await client.post("/api/embed", json=payload)
                if resp.status_code != 200:
                    raise OllamaAPIError(resp.status_code, resp.text)
                return resp.json()["embeddings"][0]
            except OllamaAPIError:
                raise
            except httpx.HTTPError as exc:
                raise OllamaConnectionError(str(exc)) from exc

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_messages(request: ChatRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        return messages


ollama_service = OllamaService()
