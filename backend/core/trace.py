"""轨迹与训练运行的落盘（指令 13：可审计）。

为什么用 JSONL ：可回溯性优先，一行一条 JSON 可以被 grep / jq 直接消费，
不引入依赖。写操作失败一律吞掉并返回空路径 —— 审计是旁路，不能反噬主流程。

三类记录：
    traj-YYYYMMDD.jsonl   一条完整轨迹（含每步动作、观测、耗时、token、判定）
    run-YYYYMMDD.jsonl    一次训练运行（含前后对照、回滚、升级人工事件）
    rule-YYYYMMDD.jsonl   一次规则写入（含它来自哪条轨迹的哪一步）

会话记忆（Step 7）也存在这里，不另开一套存储 —— 服务重启后依然接得上。
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.config import settings
from backend.core.models import Rule, TaskSpec, TrainResult, Trajectory


def checks_fingerprint(task: TaskSpec) -> str:
    """校验规则的指纹。轨迹不存规则正文，只存指纹：既能验证「这次判定用的是哪一版规则」，
    又不会让轨迹文件膨胀。"""
    payload = json.dumps(
        {"expr": task.expr_checks, "result": task.result_checks},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TraceStore:
    """append-only 审计日志。"""

    def __init__(self) -> None:
        self.root = Path(settings.TRACE_DIR)
        self.enabled = settings.TRACE_ENABLED

    # ---------- 写 ----------

    def _append(self, kind: str, payload: Dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        try:
            day = datetime.now().strftime("%Y%m%d")
            path = self.root / f"{kind}-{day}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return str(path)
        except Exception:
            return ""

    def write_trajectory(self, traj: Trajectory, task: TaskSpec) -> str:
        return self._append(
            "traj",
            {
                "kind": "trajectory",
                "at": traj.finished_at or datetime.now().isoformat(timespec="milliseconds"),
                "trajectory_id": traj.trajectory_id,
                "run_id": traj.run_id,
                "task": {
                    "id": task.id,
                    "split": task.split,
                    "difficulty": task.difficulty,
                    "checks_fingerprint": checks_fingerprint(task),
                },
                "trajectory": traj.model_dump(mode="json"),
            },
        )

    def write_run(self, result: TrainResult) -> str:
        return self._append(
            "run",
            {
                "kind": "training_run",
                "at": datetime.now().isoformat(timespec="milliseconds"),
                "run_id": result.run_id,
                "result": result.model_dump(mode="json"),
            },
        )

    def write_rules(
        self,
        run_id: str,
        task: TaskSpec,
        rules: List[Rule],
        trajectory_ids: List[str],
        step: int,
    ) -> str:
        """记录规则写入，且把「规则 → 轨迹 → 步」的溯源链写清楚。"""
        return self._append(
            "rule",
            {
                "kind": "rule_write",
                "at": datetime.now().isoformat(timespec="milliseconds"),
                "run_id": run_id,
                "task_id": task.id,
                "trajectory_ids": trajectory_ids,
                "origin_step": step,
                "rules": [r.model_dump(mode="json") for r in rules],
            },
        )

    # ---------- 读 ----------

    def _read(self, kind: str, limit: int) -> List[Dict[str, Any]]:
        if not self.root.exists():
            return []
        files = sorted(self.root.glob(f"{kind}-*.jsonl"))
        out: List[Dict[str, Any]] = []
        for path in reversed(files):
            try:
                lines = path.read_text("utf-8").strip().splitlines()
            except Exception:
                continue
            for line in reversed(lines):
                if len(out) >= limit:
                    return out
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def recent_trajectories(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._read("traj", limit)

    def recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._read("run", limit)

    def recent_rules(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._read("rule", limit)

    def find_trajectory(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        for item in self.recent_trajectories(limit=500):
            if item.get("trajectory_id") == trajectory_id:
                return item
        return None

    def last_turns(self, session_id: str, rounds: Optional[int] = None, scan: int = 80) -> List[Dict[str, str]]:
        """取该会话最近 N 轮的「提问 + 产出」，作为下一轮的对话前缀（指令 3、Step 7）。

        产出可能是 WL 表达式（codegen）也可能是一段回答（chat），两种都要能回放，
        否则闲聊一轮之后再问「接着说」模型就断片了。
        """
        rounds = rounds or settings.MEMORY_ROUNDS
        if not self.enabled or not session_id or rounds <= 0:
            return []

        turns: List[List[Dict[str, str]]] = []
        for item in self.recent_trajectories(limit=scan):
            traj = item.get("trajectory") or {}
            if traj.get("session_id") != session_id:
                continue
            if traj.get("enhanced_query"):
                answer = f"<final>\n```wl\n{traj['enhanced_query']}\n```\n</final>"
            elif traj.get("reply"):
                answer = traj["reply"]
            else:
                continue
            turns.append(
                [
                    {"role": "user", "content": traj.get("task_prompt", "")},
                    {"role": "assistant", "content": answer},
                ]
            )
            if len(turns) >= rounds:
                break

        # 时间倒序取出的，翻回正序再拼
        out: List[Dict[str, str]] = []
        for turn in reversed(turns):
            out.extend(turn)
        return out
