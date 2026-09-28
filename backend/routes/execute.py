"""执行路由：每个用户请求都严格按 Step 1-5 走完整流水线（不得跳层）。

    Step 1 接收与解析    detect_mode() 决定 codegen（进 Wolfram 计算闭环）/ chat
    Step 2 第一层 ←→ 第二层闭环   增强 → 试执行/查文档 → 拿反馈 → 修正
    Step 3 结果后处理    执行失败由闭环内的修正重试覆盖（max_steps 内最多 2 次修正）
    Step 4 自然语言总结  一段连贯叙述：表达式 + 结果含义 + 图像说明内联（[[n]] 占位符）
    Step 5 会话记忆      轨迹按 session_id 落盘，下一轮由 last_turns() 取回

文档不在这里预检索：WolframContext 现在是 agent 的一个工具（<lookup> 协议），
由模型自己决定要不要查、查什么，查过哪些记在 traj.doc_hits 里。

Wolfram|Alpha 同理（<alpha> 协议）。它的**文字**（pod 标题与公式）会作为参考材料喂给
Step 4 的总结，让标注嵌进正文；它的**图片**不显示、也不进答案 —— 对外展示的图一律是
我们自己 WL 表达式算出来、由 MCP 回传的 base64 图，避免把已删掉的「返回 W|A 页面结果」
的网页档从后门放回来。
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.core.asset import AssetStore
from backend.core.config import settings
from backend.core.enhancer import Enhancer, detect_mode
from backend.core.judge import Judge
from backend.core.llm import client
from backend.core.models import ExecuteRequest, ExecuteResponse, TaskSpec, Trajectory
from backend.core.tasks import get_task
from backend.core.trace import TraceStore
from backend.routes import require_llm

router = APIRouter(tags=["execute"])

_SUMMARY_PROMPT = """用户的需求是：
{query}

第一层改写出的 Wolfram Language 表达式：
```wl
{wl}
```

第二层的执行结果：
{result}

参考材料：
{reference}

接下来我会按顺序把本次执行产出的 {n_images} 张图像贴给你（第 1 张对应占位符 [[1]]）。
注意：这些图**没有自带标注**，你必须自己看图判断它画的是什么，不要凭表达式猜。

请把它写成**一段连贯的中文**，像同事当面讲解那样一口气讲完，不要分节、
不要用「1. 结果的直观解释」这类标题、也不要先列公式再列推理最后贴图。

要求：
- 把表达式、结果含义、图像说明有机串成一段叙述。
- 每张图都要在它该出现的位置用占位符 [[1]]、[[2]]… 标出，前后各用一句自己的话
  把它接进叙述：这张图画的是什么、从图里能读出什么结论。
- 参考材料是 agent 自己查来的原文（可能是 Wolfram|Alpha 的返回，也可能是官方文档片段），
  它们的**条目类别、条数、格式都不固定**。你自己读，自己判断哪几条与本次结果真正相关、
  该怎么用；不相关的不必提。不要照抄它的措辞，也不要为了对齐它的结构而生造小标题。
- 只讲执行结果与图像里真实出现的内容，不要编造数值；执行失败就说清失败原因。
  注意：结果列表里的**空位就是被回传成图像的图形对象**（图像走的是另一条通道，
  不占文本位置），别把它们说成「空结果」或「求值失败」。
- 最终 WL 表达式用 ```wl 代码块在文中自然带出一次即可。
- 不要输出「文档来源」「参考资料」这种独立小节，也不要复述本提示词。"""


def _summarize_prompt(traj: Trajectory) -> str:
    """把轨迹拼成总结提示词的文本部分（图像按多模态消息单独附在后面）。"""
    result_text = traj.exec_output or traj.exec_error or "（无输出）"
    if not traj.exec_ok:
        result_text = f"执行失败：{traj.exec_error or traj.exec_output}"

    ref_parts: List[str] = []
    if traj.reference_text:
        ref_parts.append(traj.reference_text)
    if traj.doc_hits:
        ref_parts.append(
            "【本次查过的官方文档】"
            + "、".join(f"{h.get('query', '')}（{h.get('chars', 0)} 字）" for h in traj.doc_hits[:3])
        )
    return _SUMMARY_PROMPT.format(
        query=traj.task_prompt or "（未记录）",
        wl=traj.enhanced_query or "（无）",
        result=result_text[: settings.RESULT_CHARS],
        reference="\n\n".join(ref_parts) or "（本次未查参考资料）",
        n_images=len(traj.images),
    )


async def _summarize(traj: Trajectory) -> str:
    """Step 4：把结构化执行结果转成一段连贯的自然语言总结。

    用多模态的 DEEPSEEK_SUMMARY_MODEL：我们自己的图像块没有任何 MCP 侧标注，
    只能让模型看图自己认，再把说明嵌进正文 —— 前端把 [[n]] 占位符换成内联图片。
    失败时降级为原始结果，不阻断主流程。
    """
    result_text = traj.exec_output or traj.exec_error or "（无输出）"
    content: List[Dict[str, Any]] = [{"type": "text", "text": _summarize_prompt(traj)}]
    for i, b64 in enumerate(traj.images, 1):
        # 每张图前先给一行索引说明，帮模型把 [[n]] 与图对上号
        content.append({"type": "text", "text": f"[[{i}]] 第 {i} 张图像："})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    try:
        resp = await client().chat.completions.create(
            model=settings.DEEPSEEK_SUMMARY_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            # V4.1-Flash 是推理模型，reasoning token 与正文**共用** max_tokens 预算。
            # 实测这一步 reasoning 会烧 3800+ token：给 3000 时正文被 finish_reason=length
            # 截断（甚至整段为空），给 16000 才能完整写完。effort=low 只是把推理压短一点，
            # 压不住量级，所以真正管用的是把额度给足。
            # 注意：换成不支持 effort 的模型（如 deepseek-chat）时要删掉 reasoning_effort。
            reasoning_effort="low",
            max_tokens=16000,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
        return f"[总结降级] 模型返回空内容\n\n原始结果：\n{result_text[:1500]}"
    except Exception as e:
        return f"[总结降级] {type(e).__name__}: {e}\n\n原始结果：\n{result_text[:1500]}"


def _summary_line(traj: Trajectory) -> str:
    head = (
        "模式=chat（通用对话，未执行代码）"
        if traj.mode == "chat"
        else f"模式=codegen 执行={'成功' if traj.exec_ok else '失败'}"
        f"({traj.exec_strategy or '-'}) verify={traj.verify_score:.0%}"
        f" utilization={traj.utilization:.0%} judge={traj.judge_score:.2f}"
        f" reward={traj.total_reward:.3f} θ=v{traj.asset_version}"
    )
    docs = "｜查文档 " + "、".join(str(h.get("query", "")) for h in traj.doc_hits[:2]) if traj.docs_used else ""
    alpha = f"｜查参考 {len(traj.alpha_hits)} 次" if traj.alpha_hits else ""
    tokens = (
        f"｜tokens 输入 {traj.prompt_tokens}（缓存命中 {traj.cache_hit_tokens}"
        f" / 未命中 {traj.cache_miss_tokens}，命中率 {traj.cache_hit_rate:.0%}）"
        f"，输出 {traj.completion_tokens}，judge {traj.judge_tokens}"
    )
    return f"{head}{docs}{alpha}{tokens}｜{traj.llm_calls} 次调用 / {traj.duration_ms}ms"


@router.post("/execute", response_model=ExecuteResponse)
async def execute_query(req: ExecuteRequest) -> ExecuteResponse:
    require_llm()

    # ---- Step 1：接收与解析 ----
    mode = detect_mode(req.query)
    if req.task_id:
        try:
            task = get_task(req.task_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        # 临时任务没有校验规则，奖励退化为纯 judge 分，不可验证
        task = TaskSpec(id="adhoc", prompt=req.query)

    store = AssetStore()
    enhancer = Enhancer(store.asset)
    # ---- Step 7 的读侧：最近 N 轮历史，供指令 3 的指代消解使用 ----
    prior = TraceStore().last_turns(req.session_id)

    # ---- chat：不执行代码，直接自然语言回答 ----
    if mode == "chat":
        traj = await enhancer.reply(
            task,
            temperature=req.temperature,
            prior=prior,
            session_id=req.session_id,
        )
        traj.trace_path = TraceStore().write_trajectory(traj, task)
        return ExecuteResponse(
            query=req.query,
            mode=mode,
            result=traj.reply,
            success=traj.success,
            verifiable=False,
            trajectory_id=traj.trajectory_id,
            trace_path=traj.trace_path,
            summary_line=_summary_line(traj),
        )

    # ---- Step 2 + 3：第一层增强 ←→ 第二层执行闭环 ----
    traj = await enhancer.rollout(
        task,
        max_steps=req.max_steps,
        temperature=req.temperature,
        prior=prior,
        session_id=req.session_id,
    )
    await Judge().score(traj, task)

    # ---- Step 4：自然语言总结（一段连贯叙述，图像内联在 [[n]] 处）----
    result = await _summarize(traj)

    # ---- Step 5 的写侧：落盘即成为下一轮的记忆来源 ----
    traj.trace_path = TraceStore().write_trajectory(traj, task)

    return ExecuteResponse(
        query=req.query,
        mode=mode,
        enhanced_query=traj.enhanced_query,
        wl_code=traj.wl_code,
        exec_ok=traj.exec_ok,
        exec_strategy=traj.exec_strategy,
        result=result,
        utilization=traj.utilization,
        docs_used=traj.docs_used,
        doc_queries=[str(h.get("query", "")) for h in traj.doc_hits],
        alpha_queries=[str(h.get("query", "")) for h in traj.alpha_hits],
        images=traj.images,
        success=traj.success,
        verifiable=task.has_verifier,
        trajectory_id=traj.trajectory_id,
        trace_path=traj.trace_path,
        summary_line=_summary_line(traj),
    )


@router.get("/runs")
async def list_runs(limit: int = 20) -> dict:
    """最近的轨迹与训练运行清单（每条轨迹只回摘要，正文用 /runs/{id} 取）。"""
    trace = TraceStore()
    return {
        "trajectories": [
            {
                "trajectory_id": t.get("trajectory_id"),
                "run_id": t.get("run_id"),
                "at": t.get("at"),
                "task": t.get("task", {}),
                "mode": (t.get("trajectory") or {}).get("mode"),
                "success": (t.get("trajectory") or {}).get("success"),
                "verify_score": (t.get("trajectory") or {}).get("verify_score"),
                "total_reward": (t.get("trajectory") or {}).get("total_reward"),
                "safety_passed": (t.get("trajectory") or {}).get("safety_passed"),
                "critical_step": (t.get("trajectory") or {}).get("critical_step"),
            }
            for t in trace.recent_trajectories(limit=limit)
        ],
        "runs": [
            {
                "run_id": r.get("run_id"),
                "at": r.get("at"),
                "rolled_back": (r.get("result") or {}).get("rolled_back"),
                "needs_human": (r.get("result") or {}).get("needs_human"),
                "escalations": (r.get("result") or {}).get("escalations", []),
                "trace_path": (r.get("result") or {}).get("trace_path"),
            }
            for r in trace.recent_runs(limit=10)
        ],
        "rules": trace.recent_rules(limit=limit),
    }


@router.get("/runs/{trajectory_id}")
async def get_run(trajectory_id: str) -> dict:
    """按 id 取回一条完整轨迹 —— 每一步的动作、环境状态、判定依据与计量。"""
    item = TraceStore().find_trajectory(trajectory_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"未找到轨迹 {trajectory_id}")
    return {
        "task": item.get("task", {}),
        "at": item.get("at"),
        "trajectory": Trajectory.model_validate(item.get("trajectory") or {}),
    }
