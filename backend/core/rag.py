"""文档检索：直接转发 Wolfram 官方 MCP 的 WolframContext。

这是 agent 的一个工具（见 enhancer.py 的 wolfram_context）：
模型自己决定要不要查文档、查什么。
"""

from backend.core import mcp
from backend.core.config import settings


def lookup(query: str) -> str:
    """查官方参考资料，返回截断后的文档文本。

    不可达时返回空串：检索是旁路，拿不到就当没查过，不能反噬主流程。
    """
    res = mcp.context((query or "").strip())
    return (res.text if res else "")[: settings.DOC_CHARS]
