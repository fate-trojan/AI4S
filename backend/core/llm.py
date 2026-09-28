"""LLM 客户端的唯一构造点：Agent / Judge / 蒸馏共用一个 client，key 校验也在这里。"""

from typing import Optional

from openai import AsyncOpenAI

from backend.core.config import settings

_client: Optional[AsyncOpenAI] = None


def client() -> AsyncOpenAI:
    """惰性单例。未配置 Key 时抛错 —— 这是所有 LLM 路径的统一前置校验。"""
    global _client
    if not settings.llm_configured:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请 `cp .env.example .env` 后填入真实 Key。"
        )
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return _client
