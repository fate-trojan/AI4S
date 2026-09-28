"""路由公共依赖。

响应信封（cheese 的 REQ-P2 `{code,message,data}`）刻意不在本层实现 —— cheese 已经有
`backend/app/api/response.py` 与 `register_exception_handlers`，合并进 cheese 后由它统一
包裹。这里重复一遍就是造轮子。
"""

from fastapi import HTTPException

from backend.core.config import settings


def require_llm() -> None:
    """所有 LLM 路径的统一前置校验。"""
    if not settings.llm_configured:
        raise HTTPException(status_code=400, detail="DEEPSEEK_API_KEY 未配置")
