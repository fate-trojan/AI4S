"""离线自检：不联网、不需要 API Key，覆盖两条核心逻辑链。

    python tests/test_pipeline.py      # 直接跑（无 pytest 依赖）
    pytest tests -q                    # 或用 pytest

覆盖：
  1. 安全闸门（指令 12）：符号清单、语法糖、HoldComplete 剥壳、正常表达式不误伤
  2. 动作解析：<attempt> / <final> / <lookup> / <alpha> / 裸围栏 / 协议外输出
  3. 输入路由：codegen vs chat
  4. 闭环 rollout（指令 1-4 + 14）：打分、步级信用、回退到最优尝试、安全红线
  5. GRPO 组内优势（指令 15）：均值/标准差、零方差、边界
  6. θ 资产（asset）：写入去重、超限剪枝、版本推进、快照回滚
  7. 参考资料工具（wolfram_context / wolfram_alpha）：解析、按工具校验、hits 记录
  8. 会话记忆（Step 5）：最近 3 轮读取
  9. judge：安全红线直接归零且不调用 LLM
 10. Wolfram 资源利用率：五信号合成、rich 形态代理、符号缓存命中、在奖励里占多数
 11. MCP content 异构块：text + image 都要收集，图像按 MAX_IMAGES 截断
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 必须在导入 backend.core.config 之前设置，否则 settings 单例已经定型
_TMP = tempfile.mkdtemp(prefix="ai4s-test-")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-used")
os.environ["ASSET_DIR"] = _TMP
os.environ["TRACE_DIR"] = _TMP

from backend.core import enhancer as enhancer_mod  # noqa: E402
from backend.core.asset import AssetStore, DEFAULT_BASE_PROMPT  # noqa: E402
from backend.core.config import settings  # noqa: E402
from backend.core.executor import ExecResult  # noqa: E402
from backend.core.grpo import group_advantages  # noqa: E402
from backend.core.judge import Judge  # noqa: E402
from backend.core.models import Action, Rule, TaskSpec, Trajectory  # noqa: E402
from backend.core.safety import scan, unwrap_hold  # noqa: E402
from backend.core.trace import TraceStore, checks_fingerprint  # noqa: E402

#: 有些用例会把 enhancer.execute 换成假执行器；每个用例开始前必须还原，
#: 否则前一个用例的假执行器会泄漏到后一个（pytest 靠 setup_function，直接跑靠 main）。
_ORIGINAL_EXECUTE = enhancer_mod.execute


def setup_function(function=None):
    enhancer_mod.execute = _ORIGINAL_EXECUTE


# ===================== 1. 安全闸门 =====================


def test_safety_blocks_and_allows():
    # 禁止符号
    assert scan('Run["rm -rf /"]')
    assert scan('DeleteFile["/etc/passwd"]')
    assert scan('Import["http://x.com/a.csv"]')
    assert scan('Export["/tmp/a.csv", data]')
    assert scan('URLSubmit["http://x.com"]')
    # 语法糖
    assert scan("!ls -la")
    assert scan('Get["<<file"]') or scan("<<file.m")
    assert scan("expr >> \"/tmp/out.txt\"")
    # 注释与字符串里的名字不算调用
    assert not scan('(* Run["x"] *)\nPlot[Sin[x], {x, 0, 1}]')
    assert not scan('TextString["Run and DeleteFile"]')
    # 正常数学表达式不误伤（含 -> 与 >= ）
    assert not scan("Solve[x^2 - 3 x + 2 == 0, x]")
    assert not scan("Plot[Sin[x], {x, -2 Pi, 2 Pi}, AxesLabel -> {\"x\", \"y\"}]")
    assert not scan("If[10 >= 5, Integrate[x^2, x], 0]")
    assert scan("") == []


def test_safety_unwraps_hold():
    """指令 12.3：HoldComplete 里的表达式必须先剥壳再检查。"""
    held = 'HoldComplete[Run["whoami"]]'
    assert unwrap_hold(held) == 'Run["whoami"]'
    assert scan(held), "剥壳前的 HoldComplete 不能当作安全"
    assert not scan('HoldComplete[Plot[Sin[x], {x, 0, 1}]]')


def test_executor_refuses_blocked_without_network():
    """命中安全闸门时连网络都不该出，strategy 直接是 blocked。"""
    res = asyncio.run(enhancer_mod.execute('Run["echo pwned"]'))
    assert not res.ok and res.strategy == "blocked"
    assert res.violations and res.latency_ms >= 0


# ===================== 2. 动作解析 =====================


def test_parse_action():
    final = enhancer_mod.parse_action(
        '好的\n<final>\n```wl\nPlot[Sin[x], {x, -2 Pi, 2 Pi}]\n```\n</final>'
    )
    assert final.type == "answer" and "Plot" in final.tool_args["code"]

    attempt = enhancer_mod.parse_action("<attempt>\n```wl\nEigenvalues[m]\n```\n</attempt>")
    assert attempt.type == "tool_call" and attempt.tool_name == "execute_wl"
    assert not attempt.parse_failed

    bare = enhancer_mod.parse_action("```wl\nIntegrate[x^2, x]\n```")
    assert bare.type == "tool_call" and bare.parse_failed, "裸围栏要标记 parse_failed"

    tagged = enhancer_mod.parse_action("<attempt><wl>Solve[x == 1, x]</wl></attempt>")
    assert tagged.type == "tool_call" and "Solve" in tagged.tool_args["code"]

    junk = enhancer_mod.parse_action("我觉得应该用 Plot 函数")
    assert junk.type == "think" and junk.parse_failed


def test_strip_protocol_removes_codegen_shell_from_chat_reply():
    """chat 的回答不该带 <final>/```wl 这类「用过 Wolfram」的痕迹 —— 拆壳留内容。"""
    raw = '好的\n<final>\n```wl\nPlot[Sin[x], {x, -2 Pi, 2 Pi}]\n```\n</final>'
    got = enhancer_mod.strip_protocol(raw)
    assert "<final>" not in got and "```" not in got
    assert got == "好的\n\nPlot[Sin[x], {x, -2 Pi, 2 Pi}]"

    assert enhancer_mod.strip_protocol("就是一段普通的中文回答。") == "就是一段普通的中文回答。"


def test_parse_action_lookup():
    """<lookup> 是查文档工具；裸 WL 表达式仍优先当成要执行，不被当成查询词。"""
    look = enhancer_mod.parse_action("<lookup>Eigenvalues 的语法</lookup>")
    assert look.type == "tool_call" and look.tool_name == "wolfram_context"
    assert look.tool_args == {"query": "Eigenvalues 的语法"}
    assert not look.parse_failed

    bare = enhancer_mod.parse_action("```wl\nEigenvalues[m]\n```")
    assert bare.tool_name == "execute_wl"

    empty = enhancer_mod.parse_action("<lookup>  </lookup>")
    assert empty.type == "think" and empty.parse_failed, "空查询词不该当成一次工具调用"


def test_parse_action_alpha():
    """<alpha> 是查 W|A 参考视图的工具；空内容降级为 think，不能算一次调用。"""
    a = enhancer_mod.parse_action("<alpha>Integrate[x^2 Sin[x], {x, 0, Pi}]</alpha>")
    assert a.type == "tool_call" and a.tool_name == "wolfram_alpha"
    assert a.tool_args == {"query": "Integrate[x^2 Sin[x], {x, 0, Pi}]"}
    assert not a.parse_failed

    empty = enhancer_mod.parse_action("<alpha>  </alpha>")
    assert empty.type == "think" and empty.parse_failed, "空查询词不该当成一次工具调用"


def test_probe_routes_by_wolfram_context():
    """路由不看关键词，只看 WolframContext 有没有可用返回。"""
    from backend.core import mcp as mcp_mod

    replies = {
        "画个正弦曲线": mcp_mod.ToolOutput(text="<result query='y = sin(x)'>plot</result>"),
        # 别的学科也算：Wolfram 认得出来就该走 codegen（这正是关键词表覆盖不到的那类）
        "法国的首都是哪里": mcp_mod.ToolOutput(text="<result>Paris</result>"),
        "帮我写一首关于秋天的诗": None,  # 查不到：MCP 回空内容
        # 回了一段但全是 No Results Found 占位，同样当作没查到
        "今天心情不错": mcp_mod.ToolOutput(text="No Results Found"),
    }
    original = mcp_mod.context
    mcp_mod.context = lambda q: replies.get(q)  # type: ignore[assignment]
    try:
        assert enhancer_mod.probe("画个正弦曲线")[0] == "codegen"
        assert enhancer_mod.probe("法国的首都是哪里")[0] == "codegen"
        assert enhancer_mod.probe("帮我写一首关于秋天的诗")[0] == "chat"
        assert enhancer_mod.probe("今天心情不错")[0] == "chat"
        # 原文要一并带出来：前端靠它显示探针到底探到了什么
        assert enhancer_mod.probe("画个正弦曲线")[1].startswith("<result")
        assert enhancer_mod.probe("帮我写一首关于秋天的诗")[1] == ""
    finally:
        mcp_mod.context = original  # type: ignore[assignment]


def test_harvest_symbols_learns_from_doc_links():
    """每次 WolframContext 拿回来的文档链接里的符号名，都要沉淀进 system_names.json。"""
    from backend.core import mcp as mcp_mod

    path = Path(settings.ASSET_DIR) / "system_names.json"
    before = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    text = (
        "see paclet:ref/Eigenvalues and reference.wolfram.com/language/ref/DSolveValue.html"
        " —— paclet:ref/Eigenvalues 重复出现也不该重复计数"
    )
    assert mcp_mod.harvest_symbols(text) == 2
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["Eigenvalues"] is True and after["DSolveValue"] is True
    assert mcp_mod.harvest_symbols(text) == 0, "已在缓存里的符号不该重复计入"
    # 只新增不覆盖：先前判为「不是系统符号」的结论不能被文档链接冲掉
    assert all(after.get(k) == v for k, v in before.items())


# ===================== 3. 闭环 rollout =====================


def _scripted_enhancer(script):
    """把 _chat 换成脚本化回复，从而在没有 LLM 的情况下跑完整闭环。"""
    store = AssetStore("codegen")
    enh = enhancer_mod.Enhancer(store.asset)
    calls = {"n": 0}

    async def fake_chat(messages, temperature, max_tokens=2048):
        i = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        return script[i], {"prompt_tokens": 10, "completion_tokens": 5,
                           "cache_hit_tokens": 0, "cache_miss_tokens": 10, "latency_ms": 1}

    enh._chat = fake_chat  # type: ignore[method-assign]
    return enh, calls


def test_called_functions_ignores_string_literals_and_comments():
    """字符串字面量里的 `over [0, Pi]` 不是函数调用：先把字面量与注释剥掉再找函数名。"""
    from backend.core.judge import _called_functions

    code = 'Plot[x, {x, 0, 1}, PlotLabel -> "x over [0, Pi]", AxesLabel -> {"a", "b"}]'
    assert _called_functions(code) == ["Plot"], _called_functions(code)
    assert _called_functions("(* tip: use Plot[ *) Sin[x]") == ["Sin"]


def test_called_functions_ignores_unknown_quantities():
    """被求解的未知量（y / x）不是编造的系统函数，不能因此扣利用率。"""
    from backend.core.judge import _called_functions

    code = "{sol = DSolveValue[y''[x] + y[x] == 0, y[x], x], N[sol /. {C[1] -> 1} /. x -> 1, 10]}"
    assert _called_functions(code) == ["C", "DSolveValue", "N"], _called_functions(code)


def _fake_execute_ok():
    async def fake(expr, retries=0):
        return ExecResult(ok=True, output=f"result of {expr}", strategy="fake", wl_code=expr)
    return fake


def test_rollout_scores_and_credits_steps():
    enhancer_mod.execute = _fake_execute_ok()  # type: ignore[assignment]
    task = TaskSpec(id="t-plot", prompt="画个正弦曲线", expr_checks=["Plot", "Sin", "{x,"])
    script = [
        "<attempt>\n```wl\nPlot[x, {x, 0, 1}]\n```\n</attempt>",   # 2/3：缺 Sin
        "<final>\n```wl\nPlot[Sin[x], {x, -2 Pi, 2 Pi}]\n```\n</final>",  # 3/3
    ]
    enh, calls = _scripted_enhancer(script)
    traj = asyncio.run(enh.rollout(task, max_steps=3))

    assert traj.enhanced_query.startswith("Plot[Sin[x]")
    assert traj.verify_score == 1.0 and traj.exec_ok
    assert not traj.used_fallback_query
    assert traj.critical_step == 1, "应采用第 2 步的表达式"
    assert traj.steps[1].credited
    assert traj.steps[0].verify_score < traj.steps[1].verify_score
    assert traj.llm_calls == 2 and traj.prompt_tokens == 20
    assert traj.cache_miss_tokens == 20 and traj.cache_hit_rate == 0.0


def test_rollout_falls_back_to_best_attempt():
    enhancer_mod.execute = _fake_execute_ok()  # type: ignore[assignment]
    task = TaskSpec(id="t-plot", prompt="画个正弦曲线", expr_checks=["Plot", "Sin"])
    script = [
        "<attempt>\n```wl\nPlot[Sin[x], {x, 0, 1}]\n```\n</attempt>",
        "我还没想好。",  # 协议外输出，浪费一步
        "<attempt>\n```wl\nPlot[Sin[y], {y, 0, 1}]\n```\n</attempt>",
    ]
    enh, _ = _scripted_enhancer(script)
    traj = asyncio.run(enh.rollout(task, max_steps=3))

    assert traj.used_fallback_query, "没提交 <final> 时应回退到最优尝试"
    assert traj.verify_score == 1.0
    assert traj.final_env_state["terminated_by"] == "max_steps"
    assert any(s.parse_failed for s in traj.steps)


def test_rollout_blocks_unsafe_final():
    task = TaskSpec(id="t-safety", prompt="删掉临时文件")
    script = ['<final>\n```wl\nDeleteFile["/tmp/x"]\n```\n</final>']
    enh, _ = _scripted_enhancer(script)
    traj = asyncio.run(enh.rollout(task, max_steps=2))

    assert not traj.safety_passed
    assert traj.safety_violations and traj.safety_violations[0].rule == "symbol:DeleteFile"
    assert traj.steps[0].denied and not traj.exec_ok


def test_rollout_without_verifier_is_not_pretend_verifiable():
    """临时任务（无 checks）verify 恒为 0，success 必须为 False，不能假装可验证。"""
    from backend.core import judge as judge_mod

    enhancer_mod.execute = _fake_execute_ok()  # type: ignore[assignment]
    task = TaskSpec(id="adhoc", prompt="随便算点什么")
    enh, _ = _scripted_enhancer(['<final>\n```wl\n1 + 1\n```\n</final>'])
    traj = asyncio.run(enh.rollout(task, max_steps=2))

    assert traj.verify_score == 0.0 and traj.verify_detail == []
    assert traj.exec_ok, "没有校验规则不代表不执行"

    # 没有 checks 也不能把利用率这一项一起摊薄：按剩余两项的权重归一。
    # judge 分用桩固定住，好把公式算准（真去打 LLM 会让断言不确定）。
    original = judge_mod.Judge._llm_score

    async def fake_llm(self, t, spec):  # noqa: ANN001
        return 0.4, "stub", False, 0

    judge_mod.Judge._llm_score = fake_llm  # type: ignore[method-assign]
    try:
        asyncio.run(Judge().score(traj, task))
    finally:
        judge_mod.Judge._llm_score = original  # type: ignore[method-assign]

    # fake 通道 0 + 未查文档 0 + 拿到结果 1 + 无可查函数 0 + 单一结果 0 -> 0.20
    assert traj.utilization == 0.20, traj.utilization_detail
    expected = (0.55 * 0.20 + 0.15 * 0.4) / 0.70
    assert abs(traj.total_reward - expected) < 1e-6, traj.total_reward
    assert traj.success is False


def test_exec_retry_within_max_steps():
    """执行失败时模型会在 max_steps 内看到失败并修正（Step 5 的修正重试）。"""
    calls = {"n": 0}

    async def flaky(expr, retries=0):
        calls["n"] += 1
        ok = calls["n"] > 1
        return ExecResult(ok=ok, output="Integrate: x^3/3" if ok else "",
                          strategy="fake", error="" if ok else "syntax error")

    enhancer_mod.execute = flaky  # type: ignore[assignment]
    task = TaskSpec(id="t-int", prompt="积分", expr_checks=["Integrate"])
    script = [
        "<attempt>\n```wl\nIntegrate[x^2 x]\n```\n</attempt>",   # 语法错 -> 观测到失败
        "<final>\n```wl\nIntegrate[x^2, x]\n```\n</final>",        # 修正后提交
    ]
    enh, _ = _scripted_enhancer(script)
    traj = asyncio.run(enh.rollout(task, max_steps=3))

    assert traj.exec_ok and traj.verify_score == 1.0
    assert "syntax error" in traj.observations[0].content


# ===================== 4. GRPO 组内优势 =====================


def test_group_advantages():
    advs = group_advantages([1.0, 0.0, 0.0, 1.0])
    assert abs(sum(advs)) < 1e-9, "组内优势之和应为 0"
    assert advs[0] > 0 and advs[2] < 0
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0], "零方差 -> 无学习信号"
    assert group_advantages([]) == []
    assert group_advantages([1.0]) == [0.0]


def test_group_advantages_approximately_scale_invariant():
    """优势对奖励的整体缩放近似不变（eps 会带来极小偏差，故不是精确相等）。"""
    a = group_advantages([0.1, 0.2, 0.3])
    b = group_advantages([1.0, 2.0, 3.0])
    assert all(abs(x - y) < 5e-3 for x, y in zip(a, b))


# ===================== 5. θ 资产 =====================


def test_asset_rules_and_rollback():
    store = AssetStore("codegen")
    store.reset()
    assert store.asset.base_prompt == DEFAULT_BASE_PROMPT
    v0 = store.version

    added, _ = store.add_rules([
        Rule(text="画图时始终显式给出绘图区间"),
        Rule(text="画图时始终显式给出绘图区间"),          # 重复
        Rule(text="   "),                                  # 空
        Rule(text="x" * 300, kind="antipattern"),          # 超长
        Rule(text="实体查询统一用 Entity[\"Type\", \"Name\"]"),
    ])
    assert len(added) == 2, f"去重与长度校验后应只剩 2 条，实际 {len(added)}"
    assert store.version == v0 + 1
    assert all(r.asset_version == v0 + 1 for r in added)

    snap = store.snapshot()
    store.add_rules([Rule(text="第三条经验")])
    assert len(store.asset.rules) == 3
    v1 = store.version
    assert store.restore(snap) == v1 + 1, "回滚也要推进版本号，方便审计"
    assert len(store.asset.rules) == 2
    assert store.asset.base_prompt == DEFAULT_BASE_PROMPT, "回滚不能丢手写 base_prompt"

    # 超限剪枝按 gain 排序保留
    store.asset.rules = []
    store.add_rules([Rule(text=f"经验{i}", gain=i / 10) for i in range(30)])
    assert len(store.asset.rules) <= 24


def test_prompt_render_includes_rules():
    store = AssetStore("codegen")
    store.reset()
    assert "历史经验" not in store.asset.render()
    store.add_rules([Rule(text="绘图必须补全区间")])
    assert "绘图必须补全区间" in store.asset.render()
    assert "[经验]" in store.asset.render()


# ===================== 6. 文档检索工具（wolfram_context）=====================


def test_lookup_tool_records_docs():
    """agent 用 <lookup> 查官方文档：吃一步预算，命中记进 doc_hits（利用率 docs 信号的依据）。"""
    from backend.core import mcp as mcp_mod

    original = mcp_mod.context
    mcp_mod.context = lambda q: mcp_mod.ToolOutput(  # type: ignore[assignment]
        text="Eigenvalues[m] 求矩阵特征值"
    )
    enhancer_mod.execute = _fake_execute_ok()  # type: ignore[assignment]
    try:
        task = TaskSpec(id="t-eig", prompt="特征值", expr_checks=["Eigenvalues"])
        script = [
            "<lookup>Eigenvalues</lookup>",
            "<final>\n```wl\nEigenvalues[m]\n```\n</final>",
        ]
        enh, calls = _scripted_enhancer(script)
        traj = asyncio.run(enh.rollout(task, max_steps=3))
    finally:
        mcp_mod.context = original  # type: ignore[assignment]

    assert calls["n"] == 2, "查文档要吃一步预算"
    assert traj.steps[0].tool_name == "wolfram_context"
    assert traj.docs_used and [h["query"] for h in traj.doc_hits] == ["Eigenvalues"]
    assert traj.doc_hits[0]["chars"] > 0
    assert traj.verify_score == 1.0


def test_lookup_empty_result_is_not_a_hit():
    """MCP 不可达 / 查不到词时返回空：不能算「查过文档」，否则利用率凭空虚高。"""
    from backend.core import mcp as mcp_mod

    original = mcp_mod.context
    mcp_mod.context = lambda q: None  # type: ignore[assignment]
    enhancer_mod.execute = _fake_execute_ok()  # type: ignore[assignment]
    try:
        task = TaskSpec(id="t-eig", prompt="特征值", expr_checks=["Eigenvalues"])
        script = ["<lookup>Eigenvalues</lookup>", "<final>\n```wl\nEigenvalues[m]\n```\n</final>"]
        enh, _ = _scripted_enhancer(script)
        traj = asyncio.run(enh.rollout(task, max_steps=3))
    finally:
        mcp_mod.context = original  # type: ignore[assignment]

    assert not traj.docs_used and traj.doc_hits == []
    assert "没有返回内容" in traj.observations[0].content


def test_alpha_reference_is_recorded():
    """<alpha> 查到的 W|A 视图带 pod 标注，要原样留存给 Step 4 的总结用。"""
    from backend.core import mcp as mcp_mod

    original = mcp_mod.alpha
    mcp_mod.alpha = lambda q: mcp_mod.ToolOutput(  # type: ignore[assignment]
        text="# Definite integral\nintegral_0^π x^2 sin(x) dx = π^2 - 4≈5.8696"
    )
    enhancer_mod.execute = _fake_execute_ok()  # type: ignore[assignment]
    try:
        task = TaskSpec(id="t-int", prompt="定积分", expr_checks=["Integrate"])
        script = [
            "<alpha>Integrate[x^2 Sin[x], {x, 0, Pi}]</alpha>",
            "<final>\n```wl\nIntegrate[x^2 Sin[x], {x, 0, Pi}]\n```\n</final>",
        ]
        enh, calls = _scripted_enhancer(script)
        traj = asyncio.run(enh.rollout(task, max_steps=3))
    finally:
        mcp_mod.alpha = original  # type: ignore[assignment]

    assert calls["n"] == 2, "查参考也要吃一步预算"
    assert traj.steps[0].tool_name == "wolfram_alpha"
    assert traj.alpha_queries == ["Integrate[x^2 Sin[x], {x, 0, Pi}]"]
    assert traj.alpha_hits and traj.alpha_hits[0]["chars"] > 0
    assert "# Definite integral" in traj.reference_text, "pod 标注必须留下来"
    assert "Integrate[x^2 Sin[x], {x, 0, Pi}]" in traj.reference_text, "要标明这是对哪个查询的视图"


def test_audit_excludes_images_and_reference_text():
    """base64 图与参考资料正文不落审计：日志只留张数（一张 ~14KB，会把 JSONL 撑爆）。"""
    import json

    traj = Trajectory(
        task_id="t-big", task_prompt="画图", session_id="s-audit",
        enhanced_query="Plot[Sin[x], {x, 0, 1}]", exec_ok=True,
        images=["x" * 200], reference_text="【Wolfram|Alpha 对「Q」给出的视图】\n# Pod",
    )
    TraceStore().write_trajectory(traj, TaskSpec(id="t-big", prompt="画图"))

    dumped = json.dumps(traj.model_dump(mode="json"), ensure_ascii=False)
    assert "x" * 200 not in dumped, "base64 图不能进审计"
    assert "# Pod" not in dumped, "参考资料正文不能进审计"
    assert '"image_count": 1' in dumped, "但张数要留：审计与 rich 信号都看它"


def test_execute_streams_stage_events_then_result():
    """/execute 走 SSE：阶段事件先推完，最后一条 result 带完整结果（前端靠它点亮思维链）。"""
    from fastapi.testclient import TestClient

    from backend.core.models import ExecuteResponse
    from backend.main import app
    from backend.routes import execute as exec_mod

    async def fake_run(req, on_stage):
        await on_stage("nl", "已收到问题")
        await on_stage("enhance", "查函数名与语法", {"probe": "<result>Sin</result>"})
        await on_stage("enhance", "改写结果", {"wl": "Plot[Sin[x], {x, -2 Pi, 2 Pi}]"})
        await on_stage("wolfram", "execute_wl")
        await on_stage("summary", "把结果写成一段话")
        await on_stage("summary", "Summary 思维链", {"reasoning": "先看形状…"})
        return ExecuteResponse(query=req.query, mode="codegen", result="一段话")

    original = exec_mod._run
    exec_mod._run = fake_run  # type: ignore[assignment]
    try:
        with TestClient(app) as c:
            resp = c.post("/execute", json={"query": "画个正弦曲线"})
    finally:
        exec_mod._run = original  # type: ignore[assignment]

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(l[5:]) for l in resp.text.splitlines() if l.startswith("data:")]
    assert [e["stage"] for e in events] == [
        "nl", "enhance", "enhance", "wolfram", "summary", "summary", "result"
    ]
    assert events[1]["label"] == "Semantic Enhancement"
    # 每个阶段都带着自己的可观察产物，别让整条链路只有最后一条 result 有内容
    assert events[1]["payload"]["probe"] == "<result>Sin</result>"
    assert events[2]["payload"]["wl"].startswith("Plot[")
    assert events[5]["payload"]["reasoning"] == "先看形状…"
    assert "payload" not in events[3], "没有产物的阶段不该凭空带 payload"
    assert events[-1]["data"]["result"] == "一段话"


def test_validate_per_tool_args():
    """两种工具各只收一个字段；且只对将执行的代码做安全扫描。"""
    env = enhancer_mod.WlEnv(TaskSpec(id="t", prompt="x"), 2, {})

    ok = Action(type="tool_call", tool_name="wolfram_context", tool_args={"query": "Eigenvalues"})
    assert env._validate(ok) is None

    wrong_field = Action(
        type="tool_call", tool_name="wolfram_context", tool_args={"code": "Eigenvalues[m]"}
    )
    assert "只接受 query" in (env._validate(wrong_field) or "")

    unknown = Action(type="tool_call", tool_name="import_file", tool_args={"code": "x"})
    assert "未知工具" in (env._validate(unknown) or "")

    # 检索词不是可执行内容：扫它会污染安全状态（Import 是合法文档搜索词）
    benign = Action(
        type="tool_call", tool_name="wolfram_context", tool_args={"query": "Import 怎么用"}
    )
    assert env._validate(benign) is None
    assert env.violations == []

    unsafe = Action(
        type="tool_call", tool_name="execute_wl", tool_args={"code": 'Import["http://x/a.csv"]'}
    )
    assert "安全红线" in (env._validate(unsafe) or "")
    assert env.violations, "执行类载荷必须过安全闸门"


def test_validate_scans_only_code_payloads():
    """只有将执行的代码过安全闸门；两个查参考的工具的 query 都不扫。"""
    env = enhancer_mod.WlEnv(TaskSpec(id="t", prompt="x"), 2, {})
    for tool in ("wolfram_context", "wolfram_alpha"):
        ok = Action(type="tool_call", tool_name=tool, tool_args={"query": "Import 怎么用"})
        assert env._validate(ok) is None, tool
    assert env.violations == [], "查参考资料不该污染安全状态"

    code = Action(
        type="tool_call", tool_name="execute_wl", tool_args={"code": 'Import["http://x/a.csv"]'}
    )
    assert "安全红线" in (env._validate(code) or "")

    # 提交答案（<final>）同样是要执行的代码，必须被扫
    env2 = enhancer_mod.WlEnv(TaskSpec(id="t", prompt="x"), 2, {})
    final = Action(type="answer", tool_args={"code": 'Run["whoami"]'})
    assert "安全红线" in (env2._validate(final) or "")


# ===================== 7. 会话记忆 =====================


def test_last_turns_keeps_three_rounds():
    trace = TraceStore()
    for i in range(5):
        traj = Trajectory(
            task_id=f"t{i}", task_prompt=f"第{i}轮问题", session_id="s-mem",
            enhanced_query=f"expr{i}", exec_ok=True,
        )
        trace.write_trajectory(traj, TaskSpec(id=f"t{i}", prompt=f"第{i}轮问题"))

    turns = trace.last_turns("s-mem")
    assert len(turns) == 6, f"3 轮 = 6 条消息，实际 {len(turns)}"
    assert turns[0]["content"] == "第2轮问题", "应保留最近 3 轮且按时间正序"
    assert turns[-1]["role"] == "assistant" and "expr4" in turns[-1]["content"]
    assert trace.last_turns("另一个会话") == []
    assert trace.last_turns("") == []


def test_last_turns_records_chat_reply():
    trace = TraceStore()
    trace.write_trajectory(
        Trajectory(task_id="c1", task_prompt="你好", session_id="s-chat", reply="你好呀"),
        TaskSpec(id="c1", prompt="你好"),
    )
    turns = trace.last_turns("s-chat")
    assert turns and turns[1]["content"] == "你好呀"


# ===================== 8. judge 安全红线 =====================


def test_judge_zeroes_reward_on_safety_violation():
    from backend.core.models import SafetyViolation

    traj = Trajectory(
        task_id="t", task_prompt="x", enhanced_query='DeleteFile["/tmp/a"]',
        safety_passed=False,
        safety_violations=[SafetyViolation(rule="symbol:DeleteFile", detail="禁止调用 DeleteFile")],
    )
    task = TaskSpec(id="t", prompt="x", expr_checks=["DeleteFile"])
    asyncio.run(Judge().score(traj, task))

    assert traj.total_reward == 0.0 and traj.success is False
    assert traj.judge_score == 0.0
    assert "安全红线" in traj.judge_rationale
    assert traj.judge_tokens == 0, "命中红线时不该调用 judge（离线可测的证据）"


# ===================== 9. 校验指纹 =====================


def test_checks_fingerprint_is_stable_and_sensitive():
    a = TaskSpec(id="a", prompt="p", expr_checks=["Plot", "Sin"])
    b = TaskSpec(id="b", prompt="完全不同的题面", expr_checks=["Plot", "Sin"])
    c = TaskSpec(id="c", prompt="p", expr_checks=["Plot"])
    assert checks_fingerprint(a) == checks_fingerprint(b), "指纹只认校验规则，不认题面"
    assert checks_fingerprint(a) != checks_fingerprint(c)


# ===================== 10. Wolfram 资源利用率 =====================


def test_reward_weights_favor_utilization():
    """用户口径：利用率必须是奖励里的多数项，且三项权重和为 1。"""
    assert settings.UTIL_WEIGHT > settings.VERIFY_WEIGHT > settings.JUDGE_WEIGHT
    assert settings.UTIL_WEIGHT > 0.5, "主信号必须过半"
    total = settings.UTIL_WEIGHT + settings.VERIFY_WEIGHT + settings.JUDGE_WEIGHT
    assert abs(total - 1.0) < 1e-9, f"权重和应为 1，实际 {total}"


def test_check_symbols_caches_mcp_results():
    """符号真实性按需批查 + 落盘缓存：同一批问第二次不该再打 MCP。"""
    from backend.core import mcp as mcp_mod

    cache_path = Path(settings.ASSET_DIR) / "system_names.json"
    cache_path.unlink(missing_ok=True)
    calls: list = []
    original = mcp_mod.call_tool

    def fake_call(name, args, attempts=3):
        calls.append(name)
        return mcp_mod.ToolOutput(text='Out[1]= "Plot|Solve"')

    mcp_mod.call_tool = fake_call  # type: ignore[assignment]
    try:
        first = mcp_mod.check_symbols(["Plot", "Solve", "FooBar"])
        second = mcp_mod.check_symbols(["Plot", "Solve", "FooBar"])
    finally:
        mcp_mod.call_tool = original  # type: ignore[assignment]

    assert first == {"Plot", "Solve"} and second == {"Plot", "Solve"}
    assert len(calls) == 1, f"第二次应全部命中缓存，实际打了 {len(calls)} 次 MCP"


def test_utilization_signals():
    """五信号等权：通道真实性 / 查过官方文档 / 拿到真实结果 / 函数名真实 / 答案是否丰富。"""
    from backend.core import judge as judge_mod
    from backend.core import mcp as mcp_mod

    original = mcp_mod.check_symbols
    mcp_mod.check_symbols = lambda names: {n for n in names if n in {"Solve", "N"}}  # type: ignore[assignment]
    try:
        good = Trajectory(
            task_id="t", task_prompt="解方程",
            enhanced_query="{Solve[x == 1, x], N[1/3, 10]}",
            exec_ok=True, exec_output="{{x -> 1}}", exec_strategy="mcp",
            docs_used=True, doc_hits=[{"query": "Solve", "chars": 120}],
        )
        util, detail = asyncio.run(judge_mod._utilization(good))

        bad = Trajectory(
            task_id="t", task_prompt="解方程", enhanced_query="FakeFunc[x]",
            exec_ok=False, exec_strategy="blocked",
        )
        util_bad, detail_bad = asyncio.run(judge_mod._utilization(bad))
    finally:
        mcp_mod.check_symbols = original  # type: ignore[assignment]

    assert [d["signal"] for d in detail] == ["channel", "docs", "result", "functions", "rich"]
    assert util == 1.0, detail
    assert util_bad == 0.0, detail_bad
    # 编造的函数名要被点出来，蒸馏时才知道该改什么
    assert "FakeFunc" in detail_bad[3]["detail"]


def test_is_multiview():
    """rich 信号用的形态代理：只认顶层 List / Association，区间列表不能被误判。"""
    from backend.core.judge import _is_multiview

    assert _is_multiview(
        "{Integrate[x^2 Sin[x], {x, 0, Pi}], N[Integrate[x^2 Sin[x], {x, 0, Pi}], 10]}"
    )
    assert _is_multiview('<|"a" -> 1, "b" -> 2|>')
    assert not _is_multiview("Integrate[x^2 Sin[x], {x, 0, Pi}]"), "区间列表不是多视图"
    assert not _is_multiview("Plot[x^2 Sin[x], {x, 0, Pi}]")
    assert not _is_multiview("{1}"), "只有一个元素不算多视图"
    assert not _is_multiview("")


def test_utilization_rewards_rich_answers():
    """同一 task：裸符号解 0.6，多视图或带图 1.0 —— 区分「用了」与「用透了」。"""
    from backend.core import judge as judge_mod
    from backend.core import mcp as mcp_mod

    original = mcp_mod.check_symbols
    mcp_mod.check_symbols = lambda names: set(names)  # type: ignore[assignment]
    try:
        plain = Trajectory(
            task_id="t", task_prompt="定积分",
            enhanced_query="Integrate[x^2 Sin[x], {x, 0, Pi}]",
            exec_ok=True, exec_output="-4 + Pi^2", exec_strategy="mcp",
        )
        util_plain, detail_plain = asyncio.run(judge_mod._utilization(plain))

        rich = Trajectory(
            task_id="t", task_prompt="定积分",
            enhanced_query="{Integrate[x^2 Sin[x], {x, 0, Pi}], Plot[x^2 Sin[x], {x, 0, Pi}]}",
            exec_ok=True, exec_output="{...}", exec_strategy="mcp", docs_used=True,
        )
        util_rich, _ = asyncio.run(judge_mod._utilization(rich))

        img = Trajectory(
            task_id="t", task_prompt="画图", enhanced_query="Plot[Sin[x], {x, 0, 2 Pi}]",
            exec_ok=True, exec_output="Out[1]= ", exec_strategy="mcp",
            docs_used=True, images=["AAAA"],
        )
        util_img, _ = asyncio.run(judge_mod._utilization(img))
    finally:
        mcp_mod.check_symbols = original  # type: ignore[assignment]

    # 裸解：channel1 + docs0 + result1 + functions1 + rich0
    assert util_plain == 0.6, detail_plain
    assert util_rich == 1.0 and util_img == 1.0


def test_judge_reward_weights_utilization_majority():
    """有 verifier 时 reward = 0.55*util + 0.30*verify + 0.15*judge，利用率是主导项。"""
    from backend.core import judge as judge_mod

    original_llm = judge_mod.Judge._llm_score
    original_syms = judge_mod.mcp.check_symbols

    async def fake_llm(self, t, spec):  # noqa: ANN001
        return 0.4, "stub", False, 0

    judge_mod.Judge._llm_score = fake_llm  # type: ignore[method-assign]
    judge_mod.mcp.check_symbols = lambda names: set(names)  # type: ignore[assignment]
    try:
        task = TaskSpec(id="t", prompt="解方程", expr_checks=["Solve"])
        traj = Trajectory(
            task_id="t", task_prompt="解方程",
            enhanced_query="{Solve[x == 1, x], N[1/3, 10]}",
            verify_score=1.0, exec_ok=True, exec_output="{{x -> 1}}",
            exec_strategy="mcp", docs_used=True,
        )
        asyncio.run(Judge().score(traj, task))
    finally:
        judge_mod.Judge._llm_score = original_llm  # type: ignore[method-assign]
        judge_mod.mcp.check_symbols = original_syms  # type: ignore[assignment]

    assert traj.utilization == 1.0
    assert abs(traj.total_reward - (0.55 + 0.30 + 0.15 * 0.4)) < 1e-9
    assert traj.success is True
    assert traj.total_reward - traj.judge_score > 0.5, "利用率是第一大项，judge 低分压不垮总分"


# ===================== 11. MCP content 异构块 =====================


def test_call_tool_collects_images():
    """图像块必须被收集：只拼 text 会把可视化整块丢掉（MAX_IMAGES 截断生效）。"""
    from backend.core import mcp as mcp_mod

    original_post = mcp_mod._post
    original_sid = mcp_mod._session_id
    mcp_mod._session_id = None

    def fake_post(method, params=None, notify=False):
        if method == "initialize":
            return {"result": {"serverInfo": {"name": "Wolfram"}}}
        blocks = [{"type": "text", "text": "Out[1]= "}]
        blocks += [{"type": "image", "data": f"img{i}"} for i in range(6)]
        return {"result": {"content": blocks}}

    mcp_mod._post = fake_post  # type: ignore[assignment]
    try:
        out = mcp_mod.call_tool("WolframLanguageEvaluator", {"code": "Plot[x, {x, 0, 1}]"})
    finally:
        mcp_mod._post = original_post  # type: ignore[assignment]
        mcp_mod._session_id = original_sid

    assert out is not None and out.text == "Out[1]=", out.text
    assert out.images == [f"img{i}" for i in range(settings.MAX_IMAGES)], out.images


# ===================== runner =====================


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        setup_function()
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
