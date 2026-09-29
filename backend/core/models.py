"""数据模型。沿用 agent 内核的六元组形式化：

    S 状态空间 -> State          A 动作空间 -> Action
    O 观测空间 -> Observation    R 奖励     -> Trajectory.rewards / total_reward
    P 转移     -> backend/core/enhancer.py 的 rollout
    π 策略     -> enhancer.py + asset.py

与 agent 的差别只有一处口径：动作产出的是「增强后的 Wolfram 查询」，
因此 Trajectory 里 final_code 换成 enhanced_query，验证信号换成第二层执行结果。
"""

import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


# ===================== 六元组 =====================


class State(BaseModel):
    env_state: Dict[str, Any] = Field(default_factory=dict)
    step: int = 0


class Action(BaseModel):
    type: Literal["think", "tool_call", "answer"]
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    parse_failed: bool = False


class Observation(BaseModel):
    content: str
    success: bool


class SafetyViolation(BaseModel):
    rule: str
    detail: str
    line: int = 0


class StepRecord(BaseModel):
    """单步的审计与信用记录。"""

    step: int
    action_type: str
    parse_failed: bool = False
    tool_name: Optional[str] = None

    # ---- 环境判定 ----
    passed: int = 0
    total: int = 0
    verify_score: float = 0.0
    verify_delta: float = 0.0
    credited: bool = False

    # ---- 越界与安全 ----
    denied: bool = False
    deny_reason: str = ""
    error: str = ""

    # ---- 计量 ----
    llm_latency_ms: int = 0
    exec_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    started_at: str = ""
    finished_at: str = ""


class Trajectory(BaseModel):
    """完整轨迹 τ = (s₀, a₀, o₀, r₀, ..., s_T)。"""

    task_id: str
    task_prompt: str = ""

    trajectory_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    run_id: str = ""
    session_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    trace_path: str = ""

    states: List[State] = Field(default_factory=list)
    actions: List[Action] = Field(default_factory=list)
    observations: List[Observation] = Field(default_factory=list)
    rewards: List[float] = Field(default_factory=list)
    steps: List[StepRecord] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False

    #: codegen = 进 Wolfram 计算闭环；chat = 直接自然语言回答，不执行代码
    mode: Literal["codegen", "chat"] = "codegen"
    reply: str = ""

    # ---- 第一层产出（指令 1-4）----
    enhanced_query: str = ""

    # ---- 第二层产出 ----
    exec_ok: bool = False
    exec_output: str = ""
    exec_strategy: str = ""        # mcp / blocked / none
    exec_error: str = ""

    # ---- 官方文档检索（agent 调 wolfram_context 工具）----
    docs_used: bool = False
    doc_hits: List[Dict[str, Any]] = Field(default_factory=list)

    # ---- Wolfram|Alpha 参考视图（agent 调 wolfram_alpha 工具）----
    alpha_queries: List[str] = Field(default_factory=list)
    alpha_hits: List[Dict[str, Any]] = Field(default_factory=list)

    # ---- 图像产物（第二层执行带回的 base64 PNG）----
    #: 只用于当次响应渲染，排除在审计日志之外：一张约 14KB，24 条轨迹就是几百 KB 噪声
    images: List[str] = Field(default_factory=list, exclude=True)

    # ---- 参考资料原文（W|A 的 pod 标注与公式，agent 查 `<alpha>` 得来）----
    #: 只喂给 Step 4 的总结，让它把标注嵌进正文；同样不落审计（正文几百字，日志里
    #: 已有 alpha_hits 记着查过什么）。
    reference_text: str = Field(default="", exclude=True)

    # ---- 训练拆解 ----
    verify_score: float = 0.0
    verify_detail: List[Dict[str, Any]] = Field(default_factory=list)
    #: Wolfram 资源利用率（五信号等权均值），奖励里占比最大的一项
    utilization: float = 0.0
    utilization_detail: List[Dict[str, Any]] = Field(default_factory=list)
    judge_score: float = 0.0
    judge_rationale: str = ""
    advantage: float = 0.0
    asset_version: int = 0
    used_fallback_query: bool = False
    critical_step: int = -1

    # ---- 安全闭环 ----
    safety_passed: bool = True
    safety_violations: List[SafetyViolation] = Field(default_factory=list)

    # ---- 成本计量 ----
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    judge_tokens: int = 0

    @computed_field
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return round(self.cache_hit_tokens / total, 4) if total else 0.0

    @computed_field
    @property
    def image_count(self) -> int:
        """审计里只记张数：base64 正文已 exclude，利用率信号也只看这个计数。"""
        return len(self.images)

    final_env_state: Dict[str, Any] = Field(default_factory=dict)


# ===================== 策略资产 θ =====================


class Rule(BaseModel):
    """θ 的一个可训练单元。文本即参数。"""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    text: str
    kind: Literal["success", "antipattern"] = "success"
    origin_task: str = ""
    gain: float = 0.0
    asset_version: int = 0

    # ---- 经验溯源 ----
    origin_run_id: str = ""
    origin_trajectory_ids: List[str] = Field(default_factory=list)
    origin_step: int = -1
    created_at: str = ""


class PolicyAsset(BaseModel):
    """策略资产 θ = base_prompt + rules。πθ 的输出分布完全由 render() 决定。"""

    task_type: str = "codegen"
    version: int = 0
    base_prompt: str = ""
    rules: List[Rule] = Field(default_factory=list)
    updated_at: str = ""

    def render(self) -> str:
        if not self.rules:
            return self.base_prompt
        lines = ["", "## 历史经验（由训练自动写入，按重要性排序）"]
        for i, r in enumerate(self.rules, 1):
            tag = "反例" if r.kind == "antipattern" else "经验"
            lines.append(f"{i}. [{tag}] {r.text}")
        lines.append("")
        lines.append("以上经验来自你此前在同类任务上的失败与成功，请优先遵守。")
        return self.base_prompt + "\n".join(lines)


# ===================== 任务 =====================


class TaskSpec(BaseModel):
    """一道增强任务。

    可验证信号分两类：
      expr_checks   必须出现在「增强后的 WL 表达式」里的子串 —— 离线可验证，不依赖第二层
      result_checks 必须出现在「第二层执行结果」里的子串 —— 需要执行器可用

    两类都为空 = 该任务没有可验证信号，奖励退化为纯 judge 分（不假装可验证）。
    """

    id: str
    prompt: str
    expr_checks: List[str] = Field(default_factory=list)
    result_checks: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    split: Literal["train", "eval"] = "train"

    @property
    def has_verifier(self) -> bool:
        return bool(self.expr_checks or self.result_checks)


# ===================== API 契约 =====================


class ExecuteRequest(BaseModel):
    query: str = Field(..., description="用户的自然语言输入")
    session_id: str = Field(
        default="default",
        description="多轮会话标识。不给就是 default 会话；传空串表示单轮无记忆。",
    )
    max_steps: int = 3
    temperature: float = 0.7
    task_id: Optional[str] = Field(
        default=None, description="给了才有可验证信号；不给则奖励退化为纯 judge 分"
    )


class ExecuteResponse(BaseModel):
    """一次完整流水线（Step 1-7）的结果摘要。"""

    query: str
    mode: str
    enhanced_query: str = ""
    exec_ok: bool = False
    exec_strategy: str = ""
    result: str
    utilization: float = 0.0
    #: 第二层执行带回的图像（base64 PNG，不含 data: 前缀），由前端内联渲染
    images: List[str] = Field(default_factory=list)
    success: bool = False
    verifiable: bool = False
    trajectory_id: str = ""
    trace_path: str = ""
    summary_line: str = ""


class TrainingRequest(BaseModel):
    task_type: str = "codegen"
    num_rollouts: int = 24
    epochs: int = 1
    group_size: Optional[int] = None


class Escalation(BaseModel):
    code: Literal[
        "safety_violation",
        "regression",
        "no_learning_signal",
        "llm_failure",
        "budget_exceeded",
        "unexpected_error",
    ]
    detail: str
    at: str = ""


class TrainingStatus(BaseModel):
    running: bool = False
    task_type: str = "codegen"
    epoch: int = 0
    total_epochs: int = 0
    rollout: int = 0
    total_rollouts: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    avg_utilization: float = 0.0
    asset_version: int = 0
    rules_count: int = 0
    updates_applied: int = 0
    flat_groups: int = 0
    message: str = ""
    error: Optional[str] = None

    run_id: str = ""
    needs_human: bool = False
    escalations: List[Escalation] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    judge_tokens: int = 0
    distill_tokens: int = 0
    llm_calls: int = 0


class EvalResult(BaseModel):
    asset_version: int
    rules_count: int
    num_tasks: int
    success_rate: float
    avg_reward: float
    avg_verify_score: float
    avg_utilization: float = 0.0


class TrainResult(BaseModel):
    task_type: str
    epochs: int
    total_rollouts: int
    updates_applied: int
    baseline: EvalResult
    final: EvalResult
    delta_success_rate: float
    delta_avg_reward: float
    updates_log: List[str] = Field(default_factory=list)

    run_id: str = ""
    trace_path: str = ""

    rolled_back: bool = False
    rollback_reason: str = ""
    regressed_final: Optional[EvalResult] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    judge_tokens: int = 0
    distill_tokens: int = 0
    llm_calls: int = 0
    needs_human: bool = False
    escalations: List[Escalation] = Field(default_factory=list)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.judge_tokens
            + self.distill_tokens
        )

    @computed_field
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return round(self.cache_hit_tokens / total, 4) if total else 0.0
