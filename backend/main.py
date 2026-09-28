"""AI4S Agent 服务入口。

    uvicorn backend.main:app --port 8000
    # 或
    python -m backend.main

流水线：自然语言 → 第一层语义增强(πθ) → 第二层 Wolfram MCP 执行 → 自然语言总结。
文档检索是 agent 的一个工具（<lookup> → WolframContext）。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.core import mcp
from backend.core.config import settings
from backend.routes import execute, training

app = FastAPI(
    title="AI4S Agent",
    version="0.1.0",
    description=(
        "自然语言 → Wolfram Language 智能计算闭环。"
        "第一层负责语义增强，第二层通过 Wolfram 官方 MCP 执行，文档检索由 agent 自己调工具完成。"
    ),
)

app.include_router(execute.router)
app.include_router(training.router)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    """LLM/执行器这类外部依赖失败时给出可读错误。

    完整的错误信封是 cheese 的 core/errors.py + register_exception_handlers 的事，
    合并进 cheese 后应由它接管，这里不重复实现。
    """
    return JSONResponse(status_code=502, content={"detail": f"上游依赖失败：{exc}"})


_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(_FRONTEND / "index.html"))


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """健康检查。

    第二层就是唯一那条 Wolfram MCP 通道，它通不通决定「利用率 / 可验证奖励」拿不拿得到，
    所以这里必须真去探一次。
    """
    return {
        "status": "ok",
        "llm_configured": settings.llm_configured,
        "model": settings.DEEPSEEK_MODEL,
        "wolfram": {
            "channel": "mcp",
            "url": settings.MCP_URL,
            "available": mcp.available(),
        },
        "safety_enforced": settings.SAFETY_ENFORCE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=False)
