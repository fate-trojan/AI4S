"""Self-Judge：三层奖励。

    利用率 utilization —— Wolfram 资源究竟被用上没有、用对没有、用透了没有。
                          五信号等权，权重最大。
    第一层 verifier   —— 表达式命中 checks + 第二层执行成功。客观。
    第二层 LLM judge  —— 表达式质量（准确性/完整性/可执行性）。主观，权重最低。

用户口径：Wolfram 资源利用率必须占多数。所以 UTIL_WEIGHT(0.55) > VERIFY_WEIGHT(0.30)
> JUDGE_WEIGHT(0.15)。关键是利用率这五项全部由可离线复核的客观信号算出，不交给 LLM
打分 —— 把「是不是真在用 Wolfram」交给同厂商模型判，等于把主奖励交回给 self-preference。

风险与缓解同 agent 内核：因此 (1) judge 权重压到最低；(2) success 只由 verify_score==1.0
决定；(3) judge 走独立的 DEEPSEEK_JUDGE_MODEL；(4) 训练是否有效只认 eval 集前后对照。
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Tuple

from backend.core import mcp
from backend.core.config import settings
from backend.core.llm import client
from backend.core.models import TaskSpec, Trajectory

_JUDGE_PROMPT = """你是一个严格的 Wolfram Language 专家。请评估这次「自然语言需求 → WL 表达式」的改写质量。

【用户需求】
{prompt}

【改写出的 WL 表达式】
```wl
{wl}
```

【第二层执行结果】{exec_ok}（策略 {strategy}）
{output}

【过程统计】尝试了 {steps} 步；是否在未提交 <final> 时回退到最优尝试：{fallback}

请就以下三个维度各占三分之一打分（每维 0.0~1.0）：
1. 准确性：表达式是否确切回答了用户需求，函数选用是否恰当；
2. 完整性：是否补全了省略的区间、单位、实体等成分（指令 1）；
3. 可执行性：语法是否规范、是否可直接在 Wolfram 引擎里跑通。

只输出 JSON，不要输出任何其他文字：
{{"accuracy": 0.0, "completeness": 0.0, "executability": 0.0, "reason": "一句话理由"}}
"""


_CALL_RE = re.compile(r"\b([A-Za-z$][A-Za-z0-9$]*)\s*\[")
#: 赋值左侧（f[x_] := ... / a = ...）—— 这些是代码自己定义的符号
_DEF_RE = re.compile(r"\b([A-Za-z$][A-Za-z0-9$]*)\s*(?:\[[^\]\n]*\])?\s*(?::=|=(?!=))")
#: WL 字符串字面量（含转义）。剥掉它再找函数名，否则 PlotLabel -> "... over [0, Pi]"
#: 里的 `over` 会被当成函数调用，把利用率误判低。
_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
#: 短小写名（y / x / f / dy 之类）视为用户自己的未知量，不参与真实性校验：
#: `DSolveValue[y''[x] + y[x] == 0, y[x], x]` 里的 `y` 不是编造的系统函数。
#: 解析 DSolve / Solve 的未知量参数，把它从待校验集合里剔掉。
_LOCAL_NAME_RE = re.compile(r"^[a-z][a-z0-9]?$")


def _called_functions(code: str) -> List[str]:
    """表达式里以 `名字[` 形式出现的、**值得校验真实性**的函数名。

    先剥掉字符串字面量与注释，只在真正会被求值的代码上找；再扣掉代码自己定义的符号
    （赋值左侧）与短小写的未知量。
    """
    code = _STR_RE.sub('""', code or "")
    code = re.sub(r"\(\*.*?\*\)", "", code, flags=re.S)
    defined = set(_DEF_RE.findall(code))
    return sorted(
        n
        for n in set(_CALL_RE.findall(code))
        if n not in defined and not _LOCAL_NAME_RE.match(n)
    )


def _is_multiview(code: str) -> bool:
    """判断最终表达式是不是「一次取回多视图」：顶层是 List 或 Association 且至少两个元素。

    WL 里 `{...}` 是 List、`<|...|>` 是 Association，都是「一次调用带回多个结果」的写法，
    对应 W|A 那种「符号解 + 数值近似 + 图像 + 不定积分」一起给。

    只认顶层元素：数括号深度回到 0 处的逗号。这样 `Integrate[x^2 Sin[x], {x, 0, Pi}]`
    （逗号在 `{x, 0, Pi}` 里，深度为 1）不会被误判成多视图。
    """
    s = (code or "").strip().rstrip(";").strip()
    if s.startswith("<|") and s.endswith("|>"):
        inner = s[2:-2]
    elif s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
    else:
        return False

    depth = 0
    for ch in inner:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
    return False


async def _utilization(traj: Trajectory) -> Tuple[float, List[Dict[str, Any]]]:
    """Wolfram 资源利用率 = 五信号等权均值。每一项都能离线复核，没有 LLM 参与。

    1. channel    结果确实出自 Wolfram MCP —— 网页抓来的文本 / 压根没执行都算 0
    2. docs       主动调过 wolfram_context 查官方文档
    3. result     执行成功且拿回非空结果（$Failed 在 executor 里就判成 ok=False）
    4. functions  表达式里的函数名真的是系统符号，按比例给分；没有函数可查给 0
    5. rich       答案丰富度：带回了图像，或最终表达式是多视图形态
    """
    signals: List[Dict[str, Any]] = [
        {
            "signal": "channel",
            "score": 1.0 if traj.exec_strategy == "mcp" else 0.0,
            "detail": traj.exec_strategy or "（未执行）",
        },
        {
            "signal": "docs",
            "score": 1.0 if traj.docs_used else 0.0,
            "detail": "；".join(str(h.get("query", "")) for h in traj.doc_hits[:3])
            or "未查官方文档",
        },
        {
            "signal": "result",
            "score": 1.0 if (traj.exec_ok and traj.exec_output.strip()) else 0.0,
            "detail": (traj.exec_output or traj.exec_error or "（无结果）")[:120],
        },
    ]

    funcs = _called_functions(traj.enhanced_query)
    if not funcs:
        signals.append({"signal": "functions", "score": 0.0, "detail": "没有可校验的函数调用"})
    else:
        real = await asyncio.to_thread(mcp.check_symbols, funcs)
        fake = [f for f in funcs if f not in real]
        signals.append(
            {
                "signal": "functions",
                "score": round(len(real) / len(funcs), 4),
                "detail": f"编造的函数：{', '.join(fake[:5])}" if fake else f"{len(funcs)} 个函数都存在",
            }
        )

    # 刷分的天花板是「无意义地包一层 {...}」，但那会同时压低 judge 分。若日后要更准，
    # 升级路径是解析顶层元素类型（数值 / 图像 / 规则）来判断视图是否互补。
    if traj.image_count > 0:
        rich, rich_detail = 1.0, f"带回 {traj.image_count} 张图像"
    elif _is_multiview(traj.enhanced_query):
        rich, rich_detail = 1.0, "多视图表达式（一次取回多结果）"
    else:
        rich, rich_detail = 0.0, "单一结果，无图像"
    signals.append({"signal": "rich", "score": rich, "detail": rich_detail})

    return round(sum(s["score"] for s in signals) / len(signals), 4), signals


class Judge:
    def __init__(self) -> None:
        self.client = client()

    async def _llm_score(self, traj: Trajectory, task: TaskSpec) -> Tuple[float, str, bool, int]:
        history_ok = "成功" if traj.exec_ok else "失败"
        prompt = _JUDGE_PROMPT.format(
            prompt=task.prompt,
            wl=traj.enhanced_query or "（空）",
            exec_ok=history_ok,
            strategy=traj.exec_strategy or "-",
            output=(traj.exec_output or traj.exec_error or "（无输出）")[:800],
            steps=len(traj.actions),
            fallback=traj.used_fallback_query,
        )
        try:
            resp = await self.client.chat.completions.create(
                model=settings.DEEPSEEK_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            return 0.5, f"judge 调用失败：{e}", True, 0

        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return 0.5, f"judge 输出非 JSON：{text[:120]}", True, tokens
        try:
            obj = json.loads(m.group(0))
            dims = ["accuracy", "completeness", "executability"]
            vals = [max(0.0, min(1.0, float(obj.get(d, 0.5)))) for d in dims]
            return sum(vals) / len(vals), str(obj.get("reason", ""))[:200], False, tokens
        except Exception:
            return 0.5, f"judge JSON 解析失败：{text[:120]}", True, tokens

    async def score(self, traj: Trajectory, task: TaskSpec) -> None:
        """就地填好 traj 的 judge_score / total_reward / success。"""
        if not traj.rewards:
            traj.rewards.append(0.0)

        # 安全红线优先于一切：命中即判失败，不调 judge、不留补偿空间。
        if settings.SAFETY_ENFORCE and not traj.safety_passed:
            rules = "；".join(f"{v.rule}@{v.detail}" for v in traj.safety_violations[:3])
            traj.judge_score = 0.0
            traj.judge_rationale = (
                f"[安全红线] 命中 {rules}，直接判失败（未调用 judge，reward 归零）"
            )
            traj.total_reward = 0.0
            traj.success = False
            traj.rewards[-1] = 0.0
            return

        util, util_detail = await _utilization(traj)
        traj.utilization, traj.utilization_detail = util, util_detail

        score, reason, failed, tokens = await self._llm_score(traj, task)
        traj.judge_tokens += tokens
        traj.judge_score, traj.judge_rationale = score, reason

        if task.has_verifier:
            traj.total_reward = (
                settings.UTIL_WEIGHT * util
                + settings.VERIFY_WEIGHT * traj.verify_score
                + settings.JUDGE_WEIGHT * score
            )
        else:
            # 无可验证信号时按剩余两项的权重归一：不能因为「这题没有 checks」就把
            # 利用率这一项的权重也摊薄掉，那正好削弱用户要的主信号。
            traj.total_reward = (settings.UTIL_WEIGHT * util + settings.JUDGE_WEIGHT * score) / (
                settings.UTIL_WEIGHT + settings.JUDGE_WEIGHT
            )

        # success 只认可验证信号，judge 分再高也不能把失败洗成成功
        traj.success = task.has_verifier and traj.verify_score >= 1.0
        traj.rewards[-1] = round(traj.total_reward, 6)
        if failed:
            traj.judge_rationale = "[judge 降级] " + traj.judge_rationale
