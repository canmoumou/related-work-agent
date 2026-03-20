"""统一的 OpenAI-compatible LLM 客户端。"""

from __future__ import annotations

import json
import logging
from typing import Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.models.schemas import LLMMessage
from app.utils.json_utils import extract_json_object


LOGGER = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMClient:
    """封装基础对话接口与结构化 JSON 输出能力。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def is_configured(self) -> bool:
        """检查 Qwen 相关环境变量是否已配置。"""
        return bool(self.settings.qwen_api_key and self.settings.qwen_base_url and self.settings.qwen_model)

    async def chat(self, messages: list[LLMMessage], temperature: float = 0.2) -> str:
        """发送一次基础聊天请求，返回模型原始文本。"""
        if not self.is_configured():
            raise RuntimeError(
                "Qwen client is not configured. Please set QWEN_API_KEY environment variable "
                "or add it to .env file. Get your API key from: https://dashscope.console.aliyun.com/apiKey"
            )

        headers = {
            "Authorization": f"Bearer {self.settings.qwen_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.qwen_model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.qwen_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {json.dumps(data)[:500]}") from exc

    async def chat_json(
        self,
        messages: list[LLMMessage],
        model_cls: type[ModelT],
        fallback_factory: Callable[[], ModelT],
        temperature: float = 0.2,
    ) -> ModelT:
        """优先要求模型返回 JSON；解析失败时自动重试，最后回退到兜底结果。"""
        last_error: Exception | None = None

        for attempt in range(self.settings.max_llm_retries + 1):
            try:
                content = await self.chat(messages=messages, temperature=temperature)
                payload = extract_json_object(content)
                return model_cls.model_validate(payload)
            except (httpx.HTTPError, json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as exc:
                last_error = exc
                LOGGER.warning("LLM structured parse failed on attempt %s: %s", attempt + 1, exc)

        LOGGER.warning("Using fallback structured output because LLM parsing failed: %s", last_error)
        return fallback_factory()
