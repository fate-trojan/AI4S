"""训练路由：GRPO + Self-Judge 闭环的启停与观测（指令 15-16）。

训练跑在后台协程里，接口立即返回；进度看 GET /training/status，
明细看 runs/ 下的 run-*.jsonl 与 rule-*.jsonl（指令 13 的审计面）。
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException

from backend.core.asset import DEFAULT_BASE_PROMPT, AssetStore
from backend.core.grpo import GRPOTrainer
from backend.core.models import TrainingRequest, TrainingStatus
from backend.routes import require_llm

router = APIRouter(prefix="/training", tags=["training"])

_trainer: Optional[GRPOTrainer] = None
_task: Optional[asyncio.Task] = None


def _status_of(task_type: str) -> TrainingStatus:
    if _trainer and _trainer.task_type == task_type:
        return _trainer.status
    store = AssetStore(task_type)
    return TrainingStatus(
        running=False,
        task_type=task_type,
        asset_version=store.version,
        rules_count=len(store.asset.rules),
        message="尚未训练过（θ 仅有手写的 base_prompt）",
    )


@router.get("/status", response_model=TrainingStatus)
async def status(task_type: str = "codegen") -> TrainingStatus:
    return _status_of(task_type)


@router.post("/start")
async def start(req: TrainingRequest) -> dict:
    global _trainer, _task
    require_llm()
    if _task and not _task.done():
        raise HTTPException(status_code=409, detail="已有训练在跑，先 /training/stop 或等它结束")

    _trainer = GRPOTrainer(req.task_type)
    _trainer.status.running = True
    _trainer.status.total_epochs = req.epochs
    _trainer.status.total_rollouts = req.num_rollouts
    _trainer.status.message = "已入队，正在准备…"
    _task = asyncio.create_task(_run(req))
    return {
        "started": True,
        "task_type": req.task_type,
        "num_rollouts": req.num_rollouts,
        "epochs": req.epochs,
        "note": "训练在后台进行，进度见 GET /training/status",
    }


async def _run(req: TrainingRequest) -> None:
    assert _trainer is not None
    try:
        await _trainer.train(
            num_rollouts=req.num_rollouts, epochs=req.epochs, group_size=req.group_size
        )
    except Exception as e:
        # train() 已把异常写进 status / escalations，这里只保证不产生未捕获的任务异常
        _trainer.status.running = False
        _trainer.status.error = _trainer.status.error or f"{type(e).__name__}: {e}"


@router.post("/stop")
async def stop() -> dict:
    if _trainer is None or (_task is None or _task.done()):
        return {"stopped": False, "reason": "当前没有在跑的训练"}
    _trainer.stop()
    return {"stopped": True, "note": "已请求停止，训练会在当前 group 结束后退出"}


@router.get("/result")
async def result() -> dict:
    if _trainer is None or _trainer.result is None:
        return {"available": False, "reason": "还没有完成的训练结果"}
    return {"available": True, "result": _trainer.result}


@router.get("/asset")
async def asset(task_type: str = "codegen") -> dict:
    """当前 θ：base_prompt + 规则库（含每条规则的经验溯源）。"""
    store = AssetStore(task_type)
    a = store.asset
    return {
        "task_type": task_type,
        "version": a.version,
        "updated_at": a.updated_at,
        "base_prompt_chars": len(a.base_prompt),
        "uses_default_prompt": a.base_prompt == DEFAULT_BASE_PROMPT,
        "rules": [r.model_dump(mode="json") for r in a.rules],
    }


@router.post("/asset/reset")
async def reset_asset(task_type: str = "codegen") -> dict:
    """清空规则库回到 θ₀，用于「回滚之后想从头再来」的场景。"""
    store = AssetStore(task_type)
    before = len(store.asset.rules)
    store.reset()
    return {"task_type": task_type, "rules_before": before, "rules_after": 0,
            "version": store.version}
