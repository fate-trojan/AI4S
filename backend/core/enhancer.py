"""第一层：πθ 与 rollout 控制面。

    Enhancer._chat()  —— 一次 LLM 采样，返回文本 + 本次调用的计量
    parse_action()    —— 把模型输出解析成动作（提取 WL 表达式）
    probe()           —— 输入路由：WolframContext 查得到 → codegen，查不到 → chat
    rollout()         —— 驱动 WlEnv 闭环：增强 → 执行 → 拿反馈 → 修正 → ...
    reply()           —— chat 模式：一次调用直接回答，不执行代码

本文件不产生任何梯度。θ 是 backend/core/asset.py 里的策略资产（prompt + 规则库）。

执行、安全闸门、参数校验全部在 backend/core/executor.py 与 backend/core/safety.py；
本文件只管"感知—决策"这一半，以及把每一步的计量写进轨迹。

agent 有三个工具：execute_wl（主路径）、wolfram_context（查官方文档）、
wolfram_alpha（看 W|A 对同一问题给了哪几种表示）。后两个都只是参考资料，
它们的正文回给模型，不进对外答案 —— 最终答案必须由最终的 WL 表达式算出来。
"""

import asyncio
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from backend.core import mcp
from backend.core.asset import CHAT_SUFFIX
from backend.core.config import settings
from backend.core.executor import ExecResult, execute
from backend.core.llm import client
from backend.core.models import (
    Action,
    Observation,
    PolicyAsset,
    SafetyViolation,
    State,
    StepRecord,
    TaskSpec,
    Trajectory,
)
from backend.core.safety import scan
from backend.core.utils import elapsed_ms, now_iso

_FINAL_RE = re.compile(r"<final>(.*?)</final>", re.S | re.I)
_ATTEMPT_RE = re.compile(r"<attempt>(.*?)</attempt>", re.S | re.I)
_LOOKUP_RE = re.compile(r"<lookup>(.*?)</lookup>", re.S | re.I)
_ALPHA_RE = re.compile(r"<alpha>(.*?)</alpha>", re.S | re.I)
_WL_FENCE_RE = re.compile(r"```(?:wl|wolfram|wolframlanguage|mathematica)?[ \t]*\n?(.*?)```", re.S)
_WL_TAG_RE = re.compile(r"<wl>(.*?)</wl>", re.S | re.I)

#: 工具名 -> 该工具接受的唯一参数字段
_TOOL_ARGS: Dict[str, str] = {
    "execute_wl": "code",
    "wolfram_context": "query",
    "wolfram_alpha": "query",
}
ALLOWED_TOOLS = tuple(_TOOL_ARGS)

#: 阶段名 -> 前端显示的标签。
STAGE_LABELS: Dict[str, str] = {
    "nl": "Natural Language",
    "enhance": "Semantic Enhancement",
    "wolfram": "Wolfram",
    "summary": "Summary",
}

#: 阶段进度回调：(阶段名, 补充说明, 载荷) -> 协程。
#: 载荷是该阶段的可观察产物（探针原文 / 改写出的 WL / Summary 思维链），没有就传空 dict。
StageFn = Callable[[str, str, Dict[str, Any]], Awaitable[None]]


async def _stage(
    fn: Optional[StageFn], name: str, detail: str = "", data: Optional[Dict[str, Any]] = None
) -> None:
    """上报一个阶段。回调是旁路：没人接就跳过，不影响流水线本身。"""
    if fn:
        await fn(name, detail, data or {})


#: 载荷是「将要执行的代码」的工具 —— 只有这些要过安全闸门。查询词不是可执行内容，
#: 扫它会把「Import 怎么用」这类正常检索词误判成越界。
_CODE_TOOLS = ("execute_wl",)


# ===================== 动作解析 =====================


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _looks_like_wl(text: str) -> bool:
    """判断一段裸文本是不是 WL 表达式。

    模型偶尔会不带围栏直接输出表达式，需要兜住；但不能把解释性文字也当成表达式
    （中文散文 + 函数名混在一起时误判代价很高：会被送进第二层执行）。
    判据很朴素：不含中文，且含方括号 —— WL 的函数调用必须带 []。
    """
    return bool(text) and "[" in text and not _CJK_RE.search(text)


def _extract_wl(body: str) -> Optional[str]:
    """从协议块里取出 WL 表达式。顺序：<wl> 标签 → 围栏块 → 整段裸表达式。"""
    m = _WL_TAG_RE.search(body)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = _WL_FENCE_RE.search(body)
    if m and m.group(1).strip():
        return m.group(1).strip()
    text = body.strip()
    return text if _looks_like_wl(text) else None


#: codegen 协议的标签。chat 的对外回答里不该出现这些痕迹。
_PROTOCOL_TAG_RE = re.compile(r"</?(?:final|attempt|lookup|alpha|wl)>", re.I)


def strip_protocol(text: str) -> str:
    """剥掉 codegen 协议壳，只留内容。

    CHAT_SUFFIX 压不住长 base_prompt：模型偶尔仍会把回答套进 `<final>```wl … ```</final>`。
    chat 的回答是对外文本，不该带这种「用过 Wolfram」的痕迹，所以拆壳留内容。
    """
    body = _PROTOCOL_TAG_RE.sub("", text or "")
    body = _WL_FENCE_RE.sub(lambda m: m.group(1).strip(), body)
    return body.strip()


def parse_action(text: str) -> Action:
    """把 LLM 原始输出解析成动作。模型不遵守格式时降级为 think（parse_failed=True），
    用一次步数换一次重新对齐的机会。"""
    text = text or ""

    m = _FINAL_RE.search(text)
    if m:
        wl = _extract_wl(m.group(1))
        if wl:
            return Action(type="answer", content="", tool_args={"code": wl})
        return Action(type="think", content=text.strip(), parse_failed=True)

    m = _ATTEMPT_RE.search(text)
    if m:
        wl = _extract_wl(m.group(1))
        if wl:
            return Action(
                type="tool_call", content="", tool_name="execute_wl", tool_args={"code": wl}
            )

    # 查文档排在 <attempt>/<final> 之后、裸表达式之前：模型如果顺手贴了一段裸 WL，
    # 那是想执行，不该被当成查询词送进文档检索。
    m = _LOOKUP_RE.search(text)
    if m and m.group(1).strip():
        return Action(
            type="tool_call",
            content="",
            tool_name="wolfram_context",
            tool_args={"query": m.group(1).strip()},
        )

    # 查 W|A 参考视图。与 <lookup> 同层：都是「先看一眼再改写」，都吃一步预算。
    m = _ALPHA_RE.search(text)
    if m and m.group(1).strip():
        return Action(
            type="tool_call",
            content="",
            tool_name="wolfram_alpha",
            tool_args={"query": m.group(1).strip()},
        )

    wl = _extract_wl(text)
    if wl:
        return Action(
            type="tool_call",
            content="",
            tool_name="execute_wl",
            tool_args={"code": wl},
            parse_failed=True,
        )
    return Action(type="think", content=text.strip(), parse_failed=True)


# ===================== 输入路由 =====================


def probe(query: str) -> Tuple[str, str]:
    """Step 1 的路由：拿 WolframContext 探一次，Wolfram 认得这个问题就走 codegen。

    不由关键词决定 —— 词表得按学科无限扩张，而「认不认得」本来也不是词表擅长的事。
    判据是 mcp.has_answer；MCP 不可达时判为 chat，因为 codegen 闭环依赖同一个 MCP。

    这一探要打一次网络，且 WolframContext 会抖动（实测同一条查询第一次回空、第二次
    才有内容），所以 mcp.context 里留着重试：换学科不用改代码的代价就是这段等待。

    返回 (模式, WolframContext 原文)：原文一并交给调用方，前端要显示出来 —— 这一探
    几十秒，不给出内容就是个无法解释的省略号。
    """
    out = mcp.context((query or "").strip())
    text = out.text if out else ""
    return ("codegen" if out and mcp.has_answer(text) else "chat"), text


def _meter(traj: Trajectory, meta: Dict[str, int]) -> None:
    traj.llm_calls += 1
    traj.prompt_tokens += meta.get("prompt_tokens", 0)
    traj.completion_tokens += meta.get("completion_tokens", 0)
    traj.cache_hit_tokens += meta.get("cache_hit_tokens", 0)
    traj.cache_miss_tokens += meta.get("cache_miss_tokens", 0)


def _usage_of(resp: Any) -> Dict[str, int]:
    """从响应里取 token 用量。取不到就当 0（审计里要能看出缺失）。

    DeepSeek 的缓存命中/未命中是它自己的扩展字段，OpenAI SDK 的 Usage 模型里没有，
    靠 pydantic 的 extra 透传；透传到 model_extra 还是直接取属性随 SDK 版本而定。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return dict.fromkeys(
            ("prompt_tokens", "completion_tokens", "cache_hit_tokens", "cache_miss_tokens"), 0
        )
    extra = getattr(usage, "model_extra", None) or {}

    def num(name: str) -> int:
        v = getattr(usage, name, None)
        if v is None:
            v = extra.get(name)
        return int(v or 0)

    return {
        "prompt_tokens": num("prompt_tokens"),
        "completion_tokens": num("completion_tokens"),
        "cache_hit_tokens": num("prompt_cache_hit_tokens"),
        "cache_miss_tokens": num("prompt_cache_miss_tokens"),
    }


# ===================== 环境（S, A, T, O, G, terminate）=====================


class WlEnv:
    """WL 增强任务的环境：试执行(execute_wl) / 查文档(wolfram_context) /
    查 W|A 参考(wolfram_alpha) / 提交(final)。

    S 状态 —— attempts、best、denied、step、doc_hits、alpha_hits、reference_blocks
    T 转移 —— dispatch() 是唯一入口，同时承担参数校验与安全闸门
    O 观测 —— 第二层的真实执行结果 / 官方文档片段 / W|A 参考（或安全拒绝的说明）
    G 目标 —— task 的 checks 全部命中；无 checks 则没有客观目标，交给 judge
    """

    def __init__(self, task: TaskSpec, max_steps: int, cache: Dict[str, ExecResult]) -> None:
        self.task = task
        self.max_steps = max_steps
        self.cache = cache  # 表达式 -> 执行结果，跨步复用，避免重复打网络
        self.has_verifier = task.has_verifier
        self.total_checks = len(task.expr_checks) + len(task.result_checks)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.attempts: List[Dict[str, Any]] = []
        self.best_score = -1.0
        self.best_code = ""
        self.best_step = -1
        self.last: Optional[ExecResult] = None
        self.denied = 0
        self.violations: List[SafetyViolation] = []
        self.terminated_by = ""
        self.doc_queries: List[str] = []
        self.doc_hits: List[Dict[str, Any]] = []
        self.alpha_queries: List[str] = []
        self.alpha_hits: List[Dict[str, Any]] = []
        #: W|A 参考资料原文（带 `# pod 标题` 标注），供 Step 4 总结把标注嵌进正文
        self.reference_blocks: List[str] = []

    # ---------- 状态 ----------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.id,
            "step": self.step,
            "max_steps": self.max_steps,
            "has_verifier": self.has_verifier,
            "attempts": list(self.attempts),
            "best": {"score": self.best_score, "step": self.best_step},
            "denied_count": self.denied,
            "safety_violations": len(self.violations),
            "terminated_by": self.terminated_by,
        }

    @property
    def done(self) -> bool:
        return bool(self.terminated_by) or self.step >= self.max_steps

    @property
    def safety_passed(self) -> bool:
        """一旦命中红线就永久置否：越界是运行级失败，不因后续自愈而洗白。"""
        return not self.violations

    # ---------- 转移 ----------

    async def dispatch(
        self, action: Action, llm: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Observation], StepRecord]:
        llm = llm or {}
        index = self.step
        rec = StepRecord(
            step=index,
            action_type=action.type,
            parse_failed=action.parse_failed,
            tool_name=action.tool_name,
            started_at=now_iso(),
            llm_latency_ms=int(llm.get("latency_ms", 0)),
            prompt_tokens=int(llm.get("prompt_tokens", 0)),
            completion_tokens=int(llm.get("completion_tokens", 0)),
            cache_hit_tokens=int(llm.get("cache_hit_tokens", 0)),
            cache_miss_tokens=int(llm.get("cache_miss_tokens", 0)),
        )
        args = action.tool_args if isinstance(action.tool_args, dict) else {}
        code = args.get("code")

        # ---- 提交最终表达式：终态转移 ----
        if action.type == "answer" and isinstance(code, str) and code.strip():
            deny = self._validate(action)
            if deny:
                self.step += 1
                return self._finish(rec, deny), rec
            self.best_code, self.best_step = code, index
            rec.credited = True
            self.terminated_by = "submit"
            return self._finish(rec, None), rec

        # ---- 试执行 ----
        if action.type == "tool_call" and action.tool_name == "execute_wl" and isinstance(code, str):
            deny = self._validate(action)
            if deny:
                self.step += 1
                return self._finish(rec, deny), rec
            obs = await self._execute(code, rec)
            self.step += 1
            rec.finished_at = now_iso()
            if self.step >= self.max_steps:
                self.terminated_by = "max_steps"
            return obs, rec

        # ---- 查官方文档：吃一步预算，所以不能靠反复查刷利用率 ----
        if action.type == "tool_call" and action.tool_name == "wolfram_context":
            deny = self._validate(action)
            if deny:
                self.step += 1
                return self._finish(rec, deny), rec
            obs = await self._lookup(str(args.get("query") or ""), rec)
            self.step += 1
            rec.finished_at = now_iso()
            if self.step >= self.max_steps:
                self.terminated_by = "max_steps"
            return obs, rec

        # ---- 查 W|A 参考视图：同样吃一步预算 ----
        if action.type == "tool_call" and action.tool_name == "wolfram_alpha":
            deny = self._validate(action)
            if deny:
                self.step += 1
                return self._finish(rec, deny), rec
            obs = await self._alpha(str(args.get("query") or ""), rec)
            self.step += 1
            rec.finished_at = now_iso()
            if self.step >= self.max_steps:
                self.terminated_by = "max_steps"
            return obs, rec

        # ---- 协议外动作：浪费一步 ----
        self.step += 1
        if self.step >= self.max_steps:
            self.terminated_by = "max_steps"
        rec.error = "未检测到合法的 <lookup> / <attempt> / <final> 块"
        rec.finished_at = now_iso()
        return (
            Observation(
                content="未检测到合法的 <lookup> / <attempt> / <final> 块，请严格按协议输出。",
                success=False,
            ),
            rec,
        )

    # ---------- 校验与安全闸门 ----------

    def _validate(self, action: Action) -> Optional[str]:
        """按工具分别校验参数。三个工具各只收一个字段，形状不对就直接拒，
        不让「未知字段」悄悄流进执行层。"""
        if action.type == "tool_call":
            field_ = _TOOL_ARGS.get(action.tool_name or "")
            if field_ is None:
                return f"未知工具 {action.tool_name!r}，本环境只开放 {', '.join(ALLOWED_TOOLS)}"
            if not isinstance(action.tool_args, dict):
                return "tool_args 必须是对象"
            if set(action.tool_args) != {field_}:
                return (
                    f"{action.tool_name} 只接受 {field_} 字段，收到 {sorted(action.tool_args)}"
                )
            payload = str(action.tool_args.get(field_) or "")
        else:
            payload = str((action.tool_args or {}).get("code") or "")

        if not payload.strip():
            return "参数为空"
        if len(payload) > settings.MAX_CODE_CHARS:
            return f"参数超过 {settings.MAX_CODE_CHARS} 字符上限（实际 {len(payload)}）"
        # 只扫将要执行的代码。<lookup>/<alpha> 的查询词不是可执行内容，扫它会把
        # 「Import 怎么用」这类正常检索词误判成越界 —— 所以这里用正向判断（哪些工具
        # 的载荷是代码），否则再加工具时又会漏。
        is_code = action.type == "answer" or action.tool_name in _CODE_TOOLS
        if settings.SAFETY_ENFORCE and is_code:
            hits = scan(payload)
            if hits:
                self.violations.extend(hits)
                detail = "；".join(f"{h.rule}({h.detail})" for h in hits[:3])
                return f"违反安全红线（{detail}）"
        return None

    def _finish(self, rec: StepRecord, deny: Optional[str]) -> Optional[Observation]:
        rec.finished_at = rec.finished_at or now_iso()
        if not deny:
            return None
        rec.denied = True
        rec.deny_reason = deny
        self.denied += 1
        return Observation(
            content=f"已拒绝执行：{deny}。请改写表达式后重新提交。", success=False
        )

    # ---------- 执行 ----------

    async def _lookup(self, query: str, rec: StepRecord) -> Observation:
        """查官方文档。不走 cache：同一个查询词在不同任务里语义相同，但这里没有
        跨轨迹复用点，加了反而要在 judge 里分辨命中的是缓存还是真调用。"""
        t0 = time.perf_counter()
        out = await asyncio.to_thread(mcp.context, query)
        rec.exec_latency_ms = elapsed_ms(t0)
        self.doc_queries.append(query)
        text = (out.text if out else "")[: settings.DOC_CHARS]
        if not text:
            return Observation(
                content="文档检索没有返回内容（MCP 不可达或查不到该词），请直接按你的理解改写表达式。",
                success=False,
            )
        self.doc_hits.append({"query": query, "chars": len(text)})
        return Observation(content=f"[Wolfram 官方文档]\n{text}", success=True)

    async def _alpha(self, query: str, rec: StepRecord) -> Observation:
        """查 W|A 的参考视图，看同一问题官方给出了哪几种表示。

        只回给模型当参考：它的正文不进对外答案，也不进 Step 4 的总结 —— 否则就等于
        把刚删掉的「返回 W|A 页面结果」的网页档从后门放回来。
        """
        t0 = time.perf_counter()
        out = await asyncio.to_thread(mcp.alpha, query)
        rec.exec_latency_ms = elapsed_ms(t0)
        self.alpha_queries.append(query)
        text = (out.text if out else "")[: settings.RESULT_CHARS]
        if not text:
            return Observation(
                content="Wolfram|Alpha 没有返回参考内容（MCP 不可达或查不到），"
                "请按自己的理解改写表达式。",
                success=False,
            )
        self.alpha_hits.append({"query": query, "chars": len(text)})
        # 返回是带标注的结构化文本，但**条目的类别与数量随问题而变**，所以一律不解析、
        # 原样留存：先给 agent 读，由 agent 自己判断该取哪几种表示；Step 4 的总结拿到的
        # 也是这份原文，自己去读、自己判断怎么用。
        self.reference_blocks.append(f"【参考资料：Wolfram|Alpha 查询「{query}」的返回原文】\n{text}")
        return Observation(
            content="[Wolfram|Alpha 参考返回]（仅供你判断该取哪几种表示，不要照抄；"
            f"最终答案必须由你的 WL 表达式算出来）\n{text}",
            success=True,
        )

    async def _execute(self, code: str, rec: StepRecord) -> Observation:
        before = self.best_score
        t0 = time.perf_counter()
        res = self.cache.get(code)
        if res is None:
            res = await execute(code)
            self.cache[code] = res
        rec.exec_latency_ms = elapsed_ms(t0)
        self.last = res

        if res.violations:
            self.violations.extend(res.violations)

        score, detail = self._verify(code, res)
        rec.verify_score, rec.total = score, self.total_checks
        rec.passed = int(round(score * self.total_checks))
        rec.verify_delta = round(max(0.0, score - before), 4)
        if score > self.best_score:
            self.best_score, self.best_code, self.best_step = score, code, rec.step
        self.attempts.append(
            {"step": rec.step, "score": round(score, 4), "ok": res.ok,
             "strategy": res.strategy, "error": res.error[:200]}
        )

        if res.ok:
            text = f"执行成功（{res.strategy}）。结果：\n{res.output}"
            # Plot 这类结果文本只有 `Out[1]= `，图在图像块里。不告诉模型它拿到了图，
            # 它会以为自己交了个空结果。
            if res.images:
                text += f"\n（本次结果含 {len(res.images)} 张图像，会随最终答案一起展示）"
            if self.has_verifier:
                text += f"\n校验：{'全部命中' if score >= 1.0 else f'通过率 {score:.0%}'}"
            if score >= 1.0:
                text += "\n校验通过，请输出 <final> 提交。"
        else:
            text = f"执行失败（{res.strategy}）：{res.error}"
            if res.strategy == "none":
                text += "\n注意：第二层当前不可用，请不要反复重试执行，直接把最可靠的一版作为 <final> 提交。"
        return Observation(content=text, success=res.ok)

    def _verify(self, code: str, res: ExecResult) -> Tuple[float, List[Dict[str, Any]]]:
        """可验证信号。

        expr_checks 只看改写出来的表达式（离线可判定，与第二层是否可用无关）；
        result_checks 看执行结果（第二层不可用时一律不通过，不假装成功）。
        """
        if not self.total_checks:
            return 0.0, []
        detail: List[Dict[str, Any]] = []
        for c in self.task.expr_checks:
            detail.append({"check": c, "scope": "expr", "ok": c.lower() in code.lower()})
        for c in self.task.result_checks:
            ok = bool(res.ok) and c.lower() in res.output.lower()
            detail.append({"check": c, "scope": "result", "ok": ok})
        hit = sum(1 for d in detail if d["ok"])
        return hit / self.total_checks, detail

    # ---------- 终态 ----------

    def resolve_final(self) -> str:
        """模型始终没提交 <final> 时，回退到环境记录的最好一次尝试。"""
        return self.best_code

    async def verify_final(self, code: str) -> Tuple[float, List[Dict[str, Any]], str, bool]:
        """对最终表达式取一次权威判定（复用缓存；没执行过就执行一次）。"""
        res = self.cache.get(code) if code else None
        if res is None and code:
            res = await execute(code)
            self.cache[code] = res
        if res is None:
            return 0.0, [], "（未产出表达式）", False
        score, detail = self._verify(code, res)
        return score, detail, res.output, res.ok

    def final_state(self) -> Dict[str, Any]:
        snap = self.snapshot()
        snap["terminated_by"] = self.terminated_by or "max_steps"
        return snap


# ===================== 策略 πθ =====================


class Enhancer:
    """πθ(a_t | s_t)：给定 θ（渲染为 system prompt）与观测历史，采样下一步动作。"""

    def __init__(self, asset: PolicyAsset):
        self.asset = asset
        self.client = client()
        self.cache: Dict[str, ExecResult] = {}

    async def _chat(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int = 2048
    ) -> Tuple[str, Dict[str, int]]:
        last: Optional[Exception] = None
        t0 = time.perf_counter()
        for _ in range(2):
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.DEEPSEEK_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                meta = _usage_of(resp)
                meta["latency_ms"] = elapsed_ms(t0)
                return resp.choices[0].message.content or "", meta
            except Exception as e:  # 网络/限流类错误重试一次即可
                last = e
                await asyncio.sleep(1.0)
        raise RuntimeError(f"LLM 调用失败：{last}")

    async def rollout(
        self,
        task: TaskSpec,
        max_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        run_id: str = "",
        prior: Optional[List[Dict[str, str]]] = None,
        session_id: str = "",
        on_stage: Optional[StageFn] = None,
    ) -> Trajectory:
        """跑完一条完整轨迹：增强 → （查文档 / 查 W|A 参考 / 执行）→ 观测 → 增强 ...

        prior 是最近 N 轮会话的对话前缀（指令 3 的指代消解就靠它）。
        文档与 W|A 参考都不由流水线预检索注入，而是 agent 自己用对应工具去取。
        on_stage 是给前端看的进度上报：每一步采样前报 enhance，每次工具调用前报 wolfram。
        """
        max_steps = max_steps or settings.MAX_STEPS
        temperature = settings.ACT_TEMPERATURE if temperature is None else temperature

        t0 = time.perf_counter()
        env = WlEnv(task, max_steps, self.cache)
        traj = Trajectory(
            task_id=task.id,
            task_prompt=task.prompt,
            mode="codegen",
            asset_version=self.asset.version,
            run_id=run_id,
            session_id=session_id,
            started_at=now_iso(),
        )
        history: List[Dict[str, str]] = list(prior or [])
        history.append({"role": "user", "content": task.prompt})

        while not env.done:
            traj.states.append(State(env_state=env.snapshot(), step=env.step))
            await _stage(on_stage, "enhance", f"第 {env.step + 1} 步：改写表达式")
            text, meta = await self._chat(
                [{"role": "system", "content": self.asset.render()}] + history, temperature
            )
            _meter(traj, meta)

            action = parse_action(text)
            traj.actions.append(action)
            history.append({"role": "assistant", "content": action.content or ""})

            # 改写出的 WL 就地亮出来，别拖到最后才在结果页见到
            code = str((action.tool_args or {}).get("code") or "")
            if code:
                await _stage(on_stage, "enhance", f"第 {env.step + 1} 步：改写结果", {"wl": code})

            if action.type == "tool_call":
                await _stage(on_stage, "wolfram", action.tool_name or "")
            obs, rec = await env.dispatch(action, meta)
            traj.steps.append(rec)
            traj.rewards.append(0.0)
            if obs is None:  # 提交动作：没有环境反馈，循环结束
                break
            traj.observations.append(obs)
            history.append({"role": "user", "content": f"[执行反馈]\n{obs.content}"})

        if env.terminated_by == "submit":
            traj.enhanced_query = env.best_code
        else:
            traj.enhanced_query = env.resolve_final()
            traj.used_fallback_query = True

        traj.critical_step = env.best_step
        if 0 <= env.best_step < len(traj.steps):
            traj.steps[env.best_step].credited = True

        traj.safety_passed = env.safety_passed
        traj.safety_violations = list(env.violations)
        traj.final_env_state = env.final_state()
        traj.docs_used = bool(env.doc_hits)
        traj.doc_hits = list(env.doc_hits)
        traj.alpha_queries = list(env.alpha_queries)
        traj.alpha_hits = list(env.alpha_hits)
        traj.reference_text = "\n\n".join(env.reference_blocks)

        # 终态：对最终表达式取一次权威判定。直接给 <final> 时这一步才是真正的 Wolfram
        # 调用，也是整条链路里最长的一段等待，必须报出来，不能让它藏在 Summary 之前。
        await _stage(on_stage, "wolfram", "执行最终表达式")
        score, detail, output, ok = await env.verify_final(traj.enhanced_query)
        traj.verify_score, traj.verify_detail = score, detail
        traj.exec_output, traj.exec_ok = output, ok
        last = env.cache.get(traj.enhanced_query)
        if last is not None:
            traj.exec_strategy = last.strategy
            # 图像是呈现物：只留在当次响应里，审计日志只记张数
            traj.images = list(last.images)
            if not ok:
                traj.exec_error = last.error

        # 提交动作没有经过执行分支，判定要到终态才拿到；回填给那一步，
        # 否则 step 记录里「最终采用的那一步」会显示成 0 分，步级信用就自相矛盾了。
        if 0 <= env.best_step < len(traj.steps):
            submitted = traj.steps[env.best_step]
            submitted.verify_score, submitted.total = score, env.total_checks
            submitted.passed = int(round(score * env.total_checks))

        traj.finished_at = now_iso()
        traj.duration_ms = elapsed_ms(t0)
        return traj

    async def reply(
        self,
        task: TaskSpec,
        temperature: Optional[float] = None,
        prior: Optional[List[Dict[str, str]]] = None,
        session_id: str = "",
        on_stage: Optional[StageFn] = None,
    ) -> Trajectory:
        """通用对话：一次调用直接回答，不进执行闭环。"""
        temperature = settings.ACT_TEMPERATURE if temperature is None else temperature

        t0 = time.perf_counter()
        traj = Trajectory(
            task_id=task.id,
            task_prompt=task.prompt,
            mode="chat",
            asset_version=self.asset.version,
            session_id=session_id,
            started_at=now_iso(),
        )
        history: List[Dict[str, str]] = list(prior or [])
        history.append({"role": "user", "content": task.prompt})

        await _stage(on_stage, "enhance", "直接作答")
        text, meta = await self._chat(
            [{"role": "system", "content": self.asset.render() + CHAT_SUFFIX}] + history,
            temperature,
        )
        _meter(traj, meta)
        traj.reply = strip_protocol(text)

        traj.finished_at = now_iso()
        traj.duration_ms = elapsed_ms(t0)
        return traj
