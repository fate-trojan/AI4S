"""GRPO 优势估计 + Self-Judge 闭环的编排（指令 15）。

GRPO 在本方案里贡献的是「信用分配」：

    1. 同一道任务采样 G 条轨迹，构成一个 group
    2. 组内相对优势  A_i = (r_i - mean(r)) / (std(r) + eps)      ← 原论文公式
    3. 用 A_i 的正负挑出「正样本 / 负样本」，让 LLM 对照蒸馏出可复用规则
    4. 规则写回 θ（backend/core/asset.py）

第 3 步只是一次启发式搜索，因此用三件事兜住不确定性：
θ 质量闸门（eval 变差自动回滚）、步级信用、自治边界（安全红线 / 无信号 / 预算 / 异常
→ 停机升级人工）。训练是否有效只认 eval 集的前后对照；组内奖励无方差则优势恒为 0。
"""

import asyncio
import json
import re
import statistics
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from backend.core.asset import AssetStore
from backend.core.config import settings
from backend.core.enhancer import Enhancer
from backend.core.judge import Judge
from backend.core.llm import client
from backend.core.models import (
    Escalation,
    EvalResult,
    PolicyAsset,
    Rule,
    TaskSpec,
    TrainingStatus,
    TrainResult,
    Trajectory,
)
from backend.core.tasks import eval_tasks, train_tasks
from backend.core.trace import TraceStore


def group_advantages(rewards: List[float], eps: Optional[float] = None) -> List[float]:
    """GRPO 的组内相对优势：A_i = (r_i - mean(r)) / (std(r) + eps)。

    用总体标准差（pstdev，ddof=0），与 GRPO 原式一致；方差为 0 时返回全 0 ——
    组内没有差异就没有学习信号，不该人为造梯度。
    """
    eps = settings.ADVANTAGE_EPS if eps is None else eps
    if not rewards:
        return []
    std = statistics.pstdev(rewards)
    if std < 1e-9:
        return [0.0] * len(rewards)
    mean = statistics.fmean(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


_DISTILL_PROMPT = """这是一次对照实验的结果。请从中总结「可复用的经验」，用于改进未来所有同类任务的 WL 改写。

【用户需求】
{prompt}
{blocks}
要求：
1. 最多输出 3 条经验，每条一句话、20~80 字，必须能直接指导改写 Wolfram Language 表达式。
2. 每条都必须是跨任务可复用的方法论或陷阱提醒。
3. 严禁写成针对本题具体输入的答案（例如「矩阵 {{1,2},{3,4}} 的特征值是 ...」），那是过拟合单个用例。
4. 如果给了步级信用（哪一步带来了提升），优先总结那一步的做法。
5. kind 取 "success"（从成功中总结）或 "antipattern"（从失败中总结的陷阱）。

只输出 JSON 数组，不要任何其他文字：
[{{"text": "...", "kind": "success"}}]"""


def _feedback_of(traj: Trajectory) -> str:
    if not traj.observations:
        return "（模型未执行任何尝试）"
    return " / ".join(
        o.content[:120].replace("\n", " ") for o in traj.observations[-2:]
    )


def _credit_of(traj: Trajectory) -> str:
    """把步级信用渲染成一行 —— 「能追溯到哪一步」的落地形式。"""
    rows = [
        f"第{s.step + 1}步 s={s.verify_score:.2f} Δ={s.verify_delta:+.2f}"
        for s in traj.steps
        if not s.denied and s.action_type == "tool_call"
    ]
    if not rows:
        return "（没有可评估的尝试）"
    adopted = (
        f"第{traj.critical_step + 1}步"
        if traj.critical_step >= 0
        else "（无采用表达式）"
    )
    return "；".join(rows) + f"｜最终采用 {adopted}"


def _util_note(traj: Trajectory) -> str:
    """把利用率拆成一行，让蒸馏能看见「哪一项没达标」。"""
    weak = [str(s.get("signal")) for s in traj.utilization_detail if float(s.get("score", 0)) < 1.0]
    return f"Wolfram 资源利用率 {traj.utilization:.0%}" + (
        f"（未达标：{'、'.join(weak)}）" if weak else "（四项全达标）"
    )


def _block(title: str, traj: Trajectory) -> str:
    return (
        f"\n【{title}】组内优势 {traj.advantage:+.2f}，"
        f"校验通过率 {traj.verify_score:.0%}，{_util_note(traj)}，"
        f"第二层执行 {'成功' if traj.exec_ok else '失败'}\n"
        f"步级信用：{_credit_of(traj)}\n"
        f"```wl\n{traj.enhanced_query or '（空）'}\n```\n"
        f"执行反馈：{_feedback_of(traj)}\n"
    )


def _parse_drafts(text: str) -> List[Rule]:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return []
    out: List[Rule] = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        t = str(item.get("text", "")).strip()
        if not t or len(t) > 240:
            continue
        kind = item.get("kind", "success")
        out.append(
            Rule(text=t, kind=kind if kind in ("success", "antipattern") else "success")
        )
    return out


class GRPOTrainer:
    """一轮训练 = 若干 group × (采样 → 打分 → 优势 → 更新 θ) 后做外部验证。"""

    def __init__(self, task_type: str = "codegen") -> None:
        self.task_type = task_type
        self.store = AssetStore(task_type)
        self.judge = Judge()
        self.client = client()
        self.trace = TraceStore()
        self.status = TrainingStatus(task_type=task_type)
        self.result: Optional[TrainResult] = None
        self.run_id = ""
        self.escalations: List[Escalation] = []
        self._llm_failures: List[str] = []
        self._stop = False
        self._halt = False
        self._sem = asyncio.Semaphore(max(1, settings.MAX_CONCURRENCY))
        self._reset_meters()

    def _reset_meters(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cache_hit_tokens = 0
        self._cache_miss_tokens = 0
        self._judge_tokens = 0
        self._distill_tokens = 0
        self._llm_calls = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hit_tokens + self._cache_miss_tokens
        return round(self._cache_hit_tokens / total, 4) if total else 0.0

    # ---------- 对外控制 ----------

    def stop(self) -> None:
        self._stop = True

    # ---------- 自治边界 ----------

    def _escalate(self, code: str, detail: str) -> None:
        """登记一条需要人工介入的事件（安全红线 / 回归 / 无信号 / 预算 / 异常）。"""
        self.escalations.append(
            Escalation(
                code=code,  # type: ignore[arg-type]
                detail=detail,
                at=datetime.now().isoformat(timespec="seconds"),
            )
        )
        self.status.escalations = list(self.escalations)
        self.status.needs_human = True

    def _tokens_used(self) -> int:
        return (
            self._prompt_tokens
            + self._completion_tokens
            + self._judge_tokens
            + self._distill_tokens
        )

    def _budget_exceeded(self) -> bool:
        return settings.TOKEN_BUDGET > 0 and self._tokens_used() >= settings.TOKEN_BUDGET

    def _sync_meters(self) -> None:
        self.status.prompt_tokens = self._prompt_tokens
        self.status.completion_tokens = self._completion_tokens
        self.status.cache_hit_tokens = self._cache_hit_tokens
        self.status.cache_miss_tokens = self._cache_miss_tokens
        self.status.judge_tokens = self._judge_tokens
        self.status.distill_tokens = self._distill_tokens
        self.status.llm_calls = self._llm_calls

    # ---------- 内部工具 ----------

    async def _one(
        self, agent: Enhancer, task: TaskSpec, temperature: float
    ) -> Optional[Trajectory]:
        """跑一条轨迹。任何异常都被吞成 None —— 单条失败不该炸掉整轮训练。"""
        async with self._sem:
            try:
                traj = await agent.rollout(
                    task, temperature=temperature, run_id=self.run_id
                )
                await self.judge.score(traj, task)
            except Exception as e:
                self._llm_failures.append(f"{task.id}: {type(e).__name__}: {e}")
                return None

            self._llm_calls += traj.llm_calls + 1  # +1 = judge 那次调用
            self._prompt_tokens += traj.prompt_tokens
            self._completion_tokens += traj.completion_tokens
            self._cache_hit_tokens += traj.cache_hit_tokens
            self._cache_miss_tokens += traj.cache_miss_tokens
            self._judge_tokens += traj.judge_tokens
            traj.trace_path = self.trace.write_trajectory(traj, task)
            self._sync_meters()
            return traj

    async def _evaluate(self, agent: Enhancer, tasks: List[TaskSpec]) -> EvalResult:
        """贪心采样（temperature=0）跑 eval 集，作为 θ 的外部体检。"""
        results = await asyncio.gather(*[self._one(agent, t, 0.0) for t in tasks])
        trajs = [t for t in results if t is not None]
        n = max(1, len(trajs))
        return EvalResult(
            asset_version=agent.asset.version,
            rules_count=len(agent.asset.rules),
            num_tasks=len(trajs),
            success_rate=sum(t.success for t in trajs) / n,
            avg_reward=sum(t.total_reward for t in trajs) / n,
            avg_verify_score=sum(t.verify_score for t in trajs) / n,
            avg_utilization=sum(t.utilization for t in trajs) / n,
        )

    async def _distill(
        self, task: TaskSpec, group: List[Trajectory]
    ) -> Tuple[List[Rule], str, List[str], int]:
        """找出组内正负样本，让 LLM 对照蒸馏出规则。"""
        winner = max(group, key=lambda t: t.advantage)
        loser = min(group, key=lambda t: t.advantage)

        blocks = _block("本次优势最高的一次尝试", winner)
        evidence = [winner.trajectory_id]
        if loser.advantage < 0 and loser is not winner:
            blocks += _block("本次优势最低的一次尝试", loser)
            evidence.append(loser.trajectory_id)

        prompt = _DISTILL_PROMPT.format(prompt=task.prompt, blocks=blocks)
        try:
            resp = await self.client.chat.completions.create(
                model=settings.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            return [], f"蒸馏调用失败 {type(e).__name__}: {str(e)[:100]}", evidence, -1

        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._distill_tokens += int(getattr(usage, "total_tokens", 0) or 0)

        rules = _parse_drafts(text)
        if not rules:
            return [], f"蒸馏输出解析不出规则: {text[:100]!r}", evidence, -1

        for r in rules:
            r.origin_task = task.id
            r.gain = round(winner.advantage, 4)
            r.origin_run_id = self.run_id
            r.origin_trajectory_ids = list(evidence)
            r.origin_step = winner.critical_step
            r.created_at = datetime.now().isoformat(timespec="seconds")
        return rules, "", evidence, winner.critical_step

    @staticmethod
    def _is_regression(baseline: EvalResult, final: EvalResult, tol: float = 1e-6) -> bool:
        """判定 θ 是否变差。主判据是利用率 —— 它是奖励里占多数的那一项，
        其次才是客观通过率与成功率。只看后两者的话，「利用率掉下去」的 θ 会被放行。"""
        if final.avg_utilization < baseline.avg_utilization - tol:
            return True
        if final.avg_verify_score < baseline.avg_verify_score - tol:
            return True
        return final.success_rate < baseline.success_rate - tol

    # ---------- 主循环 ----------

    async def train(
        self,
        num_rollouts: Optional[int] = None,
        epochs: int = 1,
        group_size: Optional[int] = None,
    ) -> TrainResult:
        num_rollouts = num_rollouts or settings.MAX_ROLLOUTS
        group_size = group_size or settings.GRPO_GROUP_SIZE
        if group_size < 2:
            raise ValueError("group_size 必须 >= 2，否则组内无方差、优势恒为 0")

        tasks, evals = train_tasks(), eval_tasks()
        if not tasks or not evals:
            raise RuntimeError("任务集为空：训练集与评测集都必须存在")

        self._stop = False
        self._halt = False
        self.run_id = uuid.uuid4().hex[:12]
        self.escalations = []
        self._llm_failures = []
        self._reset_meters()
        self.status = TrainingStatus(
            running=True,
            task_type=self.task_type,
            total_epochs=epochs,
            total_rollouts=num_rollouts,
            asset_version=self.store.version,
            rules_count=len(self.store.asset.rules),
            run_id=self.run_id,
            message="基线评测中…",
        )

        snapshot: PolicyAsset = self.store.snapshot()
        rollback_reason = ""
        regressed: Optional[EvalResult] = None

        try:
            agent = Enhancer(self.store.asset)
            baseline = await self._evaluate(agent, evals)

            n_groups = max(1, num_rollouts // group_size)
            done = 0
            updates = 0
            flat_groups = 0
            flat_streak = 0
            all_rewards: List[float] = []
            all_success = 0
            all_util: List[float] = []
            log: List[str] = []

            for epoch in range(1, epochs + 1):
                agent = Enhancer(self.store.asset)
                for g in range(n_groups):
                    if self._stop or self._halt:
                        break

                    if self._budget_exceeded():
                        self._halt = True
                        self._escalate(
                            "budget_exceeded",
                            f"token 用量 {self._tokens_used()} 已达预算 {settings.TOKEN_BUDGET}，停机",
                        )
                        break

                    task = tasks[g % len(tasks)]
                    results = await asyncio.gather(
                        *[
                            self._one(agent, task, settings.GROUP_TEMPERATURE)
                            for _ in range(group_size)
                        ]
                    )
                    group = [t for t in results if t is not None]

                    dropped = group_size - len(group)
                    if dropped:
                        self._escalate(
                            "llm_failure",
                            f"{task.id}: {dropped}/{group_size} 条轨迹因调用失败被丢弃"
                            f"（{self._llm_failures[-1][:120]}）",
                        )
                    if len(group) < 2:
                        log.append(
                            f"e{epoch} {task.id} 有效样本 {len(group)}/{group_size}，不足 2 条，跳过该 group"
                        )
                        continue

                    # ---- 安全红线：运行级失败，直接升级人工 ----
                    unsafe = [t for t in group if not t.safety_passed]
                    if unsafe:
                        v = unsafe[0].safety_violations[0] if unsafe[0].safety_violations else None
                        self._escalate(
                            "safety_violation",
                            f"{task.id}: 轨迹 {unsafe[0].trajectory_id} 命中安全红线 "
                            f"{v.rule if v else '?'}（{v.detail if v else ''}），该轨迹已判失败",
                        )

                    rewards = [t.total_reward for t in group]
                    advs = group_advantages(rewards)
                    for t, a in zip(group, advs):
                        t.advantage = a

                    done += len(group)
                    all_rewards.extend(rewards)
                    all_success += sum(1 for t in group if t.success)
                    all_util.extend(t.utilization for t in group)

                    is_flat = statistics.pstdev(rewards) < 1e-9
                    if is_flat:
                        flat_groups += 1
                        flat_streak += 1
                    else:
                        flat_streak = 0

                    self.status.epoch = epoch
                    self.status.rollout = done
                    self.status.avg_reward = round(sum(all_rewards) / len(all_rewards), 4)
                    self.status.success_rate = round(all_success / done, 4)
                    self.status.avg_utilization = round(sum(all_util) / len(all_util), 4)
                    self.status.flat_groups = flat_groups
                    self.status.message = (
                        f"epoch {epoch}/{epochs} · rollout {done}/{n_groups * group_size}"
                    )
                    self.status.asset_version = self.store.version
                    self.status.rules_count = len(self.store.asset.rules)
                    self._sync_meters()

                    # ---- 收敛失败：连续多个 group 无方差 → θ 拿不到任何信号 ----
                    if flat_streak >= settings.MAX_FLAT_GROUPS:
                        self._halt = True
                        self._escalate(
                            "no_learning_signal",
                            f"连续 {flat_streak} 个 group 组内奖励无方差，优势恒为 0，"
                            f"继续训练只会烧钱。建议调高 GROUP_TEMPERATURE 或换难度更合适的任务",
                        )
                        log.append(
                            f"e{epoch} r{done} {task.id} 连续 {flat_streak} 个 group 无方差，停机并升级人工"
                        )
                        break

                    # ---- 策略改进：只在存在正优势时才动 θ ----
                    best = max(advs)
                    if best <= settings.MIN_RULE_GAIN:
                        log.append(
                            f"e{epoch} r{done} {task.id} v{self.store.version} "
                            f"优势不足(max={best:+.3f})，跳过更新"
                        )
                        continue

                    rules, note, evidence, origin_step = await self._distill(task, group)
                    added, pruned = self.store.add_rules(rules)
                    if added:
                        updates += 1
                        self.status.updates_applied = updates
                        self.status.asset_version = self.store.version
                        self.status.rules_count = len(self.store.asset.rules)
                        self.trace.write_rules(
                            self.run_id, task, added, evidence, origin_step
                        )
                    log.append(
                        f"e{epoch} r{done} {task.id} v{self.store.version} "
                        f"reward={[round(r, 3) for r in rewards]} "
                        f"adv={[round(a, 2) for a in advs]} "
                        f"+{len(added)}规则 -{len(pruned)}规则"
                        + (f" ｜{note}" if note else "")
                    )

                if self._stop or self._halt:
                    break

            # ---- 外部验证：唯一能判定「训练是否真的有效」的依据 ----
            agent = Enhancer(self.store.asset)
            self.status.message = "训练后评测中…"
            final = await self._evaluate(agent, evals)

            # ---- θ 质量闸门：变差就回滚，不让坏规则留在线上 ----
            if updates > 0 and self._is_regression(baseline, final):
                detail = (
                    f"eval 变差：利用率 {baseline.avg_utilization:.0%}→{final.avg_utilization:.0%}，"
                    f"成功率 {baseline.success_rate:.0%}→{final.success_rate:.0%}，"
                    f"平均通过率 {baseline.avg_verify_score:.3f}→{final.avg_verify_score:.3f}"
                )
                if settings.ROLLBACK_ON_REGRESSION:
                    regressed = final
                    new_version = self.store.restore(snapshot)
                    rollback_reason = f"{detail}；已回滚 θ 到本轮基线（回滚后版本 v{new_version}）"
                    self._escalate(
                        "regression",
                        f"{detail}。θ 已自动回滚，但本轮蒸馏出的经验被判定无效，建议人工检查蒸馏提示词",
                    )
                    agent = Enhancer(self.store.asset)
                    self.status.message = "回滚后复测中…"
                    final = await self._evaluate(agent, evals)
                else:
                    self._escalate(
                        "regression", f"{detail}。ROLLBACK_ON_REGRESSION=false，未回滚，请人工处置"
                    )

            if self._llm_failures:
                self._escalate(
                    "llm_failure",
                    f"本轮共 {len(self._llm_failures)} 次 LLM 调用失败，首次：{self._llm_failures[0][:160]}",
                )

            self._sync_meters()
            result = TrainResult(
                task_type=self.task_type,
                epochs=epochs,
                total_rollouts=done,
                updates_applied=updates,
                baseline=baseline,
                final=final,
                delta_success_rate=round(final.success_rate - baseline.success_rate, 4),
                delta_avg_reward=round(final.avg_reward - baseline.avg_reward, 4),
                updates_log=log,
                run_id=self.run_id,
                rolled_back=bool(rollback_reason),
                rollback_reason=rollback_reason,
                regressed_final=regressed,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                cache_hit_tokens=self._cache_hit_tokens,
                cache_miss_tokens=self._cache_miss_tokens,
                judge_tokens=self._judge_tokens,
                distill_tokens=self._distill_tokens,
                llm_calls=self._llm_calls,
                needs_human=bool(self.escalations),
                escalations=list(self.escalations),
            )
            result.trace_path = self.trace.write_run(result)
            self.result = result

            msg = (
                f"完成：{done} 次 rollout，θ 更新 {updates} 次，"
                f"v{baseline.asset_version}→v{final.asset_version}，"
                f"eval 利用率 {baseline.avg_utilization:.0%}→{final.avg_utilization:.0%}，"
                f"成功率 {baseline.success_rate:.0%}→{final.success_rate:.0%}，"
                f"tokens {self._tokens_used()}"
                f"（输入 {self._prompt_tokens} 中缓存命中 {self.cache_hit_rate:.0%}）"
            )
            if rollback_reason:
                msg += "｜已回滚 θ（本轮变差）"
            if self._stop:
                msg = "已手动停止。" + msg
            elif self._halt:
                msg = "已自动停机（触发人工介入条件）。" + msg
            if updates == 0 and flat_groups == n_groups * epochs:
                msg += (
                    f"｜注意：{flat_groups} 个 group 组内奖励无方差，"
                    "GRPO 优势恒为 0，本次没有产生任何学习信号"
                )
            self.status.message = msg
            self.status.running = False
            return result

        except Exception as e:  # 服务化场景：错误走 status 通道，不炸后台任务
            self._escalate("unexpected_error", f"{type(e).__name__}: {e}")
            self.status.running = False
            self.status.error = f"{type(e).__name__}: {e}"
            self.status.message = "训练失败，详见 error / escalations 字段"
            raise
