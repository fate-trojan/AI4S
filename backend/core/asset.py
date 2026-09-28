"""策略资产 θ 的存储与版本管理。θ = base_prompt + 规则库，更新方式就是改写它再落盘。"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from backend.core.config import settings
from backend.core.models import PolicyAsset, Rule

DEFAULT_BASE_PROMPT = """# 角色
你是 AI4S Agent 的第一层：Wolfram Language 语义增强器。
你的唯一职责是把用户的自然语言科研需求，改写成 Wolfram Language 最容易正确执行的规范查询。

# 增强指令（必须全部执行）
1. 补全省略成分：用户省略的主语、对象、区间、单位、上下文一律补全。
   缺少区间时按该函数最常见的默认区间补（如折线图补 {x, -2 Pi, 2 Pi}）；
   缺少单位时补 Wolfram 的 Quantity 写法；引用实体一律补成 Entity["Type", "Name"]。
2. 口语转精确描述：把口语表达译成精确的英文 WL 表达式，不要保留中文口语。
3. 多轮指代补全：结合对话历史消解「它 / 这个 / 刚才那个」等指代；
   历史里出现过的对象、变量、结果要直接沿用，不要另起一个。
4. 富答案：一个结果值往往不足以回答需求。计算类需求除主结果外，应一并取回
   · 数值近似：N[..., 10]（符号解旁边给出可读的数字）；
   · 相关图像：能画就画，需要几种视角就画几张 —— 由问题本身决定，不要套模板。
     宁少勿滥：一张图能说清就不画第二张，重复或近似重复的图不要画，
     与结果无关的图不要画。凑数的图会拉低答案质量，不如不画。
     例如积分/求和问题可以是被积函数与积分区域（Plot[..., Filling -> Axis]）、
     逼近或累积过程、原函数/解曲线的形态；换一类问题就是另一套视角。
     每张都用 PlotLabel 写明它是什么，需要强调区域的用 Filling -> Axis 之类选项。
   把这几项用 {...} 打包成一次执行取回，不要分多次提交、也不要只给裸值。
5. 输出格式约束：只输出改写后的 WL 表达式，不要输出任何解释、注释或额外文本。

# 可用工具
  execute_wl(code)       —— 在第二层 Wolfram 执行器里执行你的表达式并返回结果。
                            绝大多数情况用它：所有计算都该走这条路。
  wolfram_context(query) —— 查 Wolfram 官方参考资料。不确定某个函数名怎么写、
                            或需要确认语法/参数时才用；查到就把文档内容用上，
                            不要凭印象编函数名。它同样消耗一步预算。
  wolfram_alpha(query)   —— 看 Wolfram|Alpha 对同一个问题给出了哪几种表示。
                            它返回的条目**类别与条数都随问题而变**，格式也不统一，
                            你得自己读完、自己判断该从中取哪几种、怎么取 ——
                            没有固定套路可套。拿不准该给用户哪些表示时才用它。
                            它的内容不会直接进入最终答案，你必须自己写成 WL
                            表达式再执行。

# 输出协议

<lookup>
函数名 或 语法问题
</lookup>
→ 查一次官方文档，我把结果反馈给你。不确定时才用。

<alpha>
用自然语言或 WL 语法描述的问题
</alpha>
→ 看一次 Wolfram|Alpha 的参考返回，我把原文反馈给你，供你读完自己判断该取哪几种表示。
  拿不准该给用户哪些表示时才用；最终答案仍必须由你的 WL 表达式算出来。

<attempt>
```wl
(* 你认为可能正确的 WL 表达式 *)
```
</attempt>
→ 提交一版表达式试执行，我会把执行结果反馈给你。

<final>
```wl
(* 最终确认的 WL 表达式 *)
```
</final>
→ 提交最终表达式。

要求：
- 代码围栏里只能放可直接执行的 WL 表达式，不要放 markdown、不要放解释。
- 只要一版就能跑对就直接给 <final>。
- 表达式里禁止出现文件读写、系统命令、网络请求类符号，命中会被安全闸门拒绝。
"""

#: 通用对话路径的收尾指令：追加在 base_prompt 之后，把它里面的 WL 输出协议压掉。
#: 到那时再把 base_prompt 拆成 persona / protocol 两段（要改资产 schema，现在不值得）。
CHAT_SUFFIX = """

# 本轮是通用对话
上面那段「WL 表达式改写」的输出协议本轮不适用。用户这一轮在闲聊或在问别的事，
直接用自然语言回答：不要输出 <lookup> / <alpha> / <attempt> / <final>，不要贴 WL 代码块，
也不要自己编一道题来算。"""


def _norm(text: str) -> str:
    """规则去重用的归一化键，中英文都保留。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text).lower()


class AssetStore:
    """一个 task_type 对应一份 θ。"""

    def __init__(self, task_type: str = "codegen"):
        self.task_type = task_type
        self.dir = Path(settings.ASSET_DIR)
        self.path = self.dir / f"{task_type}.json"
        self.asset = self._load()

    # ---------- 持久化 ----------

    def _load(self) -> PolicyAsset:
        if self.path.exists():
            try:
                return PolicyAsset.model_validate_json(self.path.read_text("utf-8"))
            except Exception:
                # 资产文件损坏时不要让训练直接崩，重新起一份
                pass
        return PolicyAsset(task_type=self.task_type, base_prompt=DEFAULT_BASE_PROMPT)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = self.asset.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)  # 原子替换，避免训练中断留下半截 JSON
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def reset(self) -> None:
        """清空规则库回到 θ₀，但保留 base_prompt —— 那是手写资产，不该被重置冲掉。"""
        self.asset = PolicyAsset(
            task_type=self.task_type, base_prompt=self.asset.base_prompt
        )
        self.save()

    # ---------- 读写 ----------

    @property
    def version(self) -> int:
        return self.asset.version

    def add_rules(self, draft: List[Rule]) -> Tuple[List[Rule], List[Rule]]:
        """写回 θ。返回 (新增的规则, 被剪掉的规则)。三道闸门：长度校验 → 去重 → 超限剪枝。"""
        existing = {_norm(r.text) for r in self.asset.rules}
        added: List[Rule] = []
        for r in draft:
            text = (r.text or "").strip()
            if not text or len(text) > 240:
                continue
            key = _norm(text)
            if not key or key in existing:
                continue
            existing.add(key)
            r.text = text
            r.asset_version = self.asset.version + 1
            self.asset.rules.append(r)
            added.append(r)

        pruned: List[Rule] = []
        if len(self.asset.rules) > settings.MAX_RULES:
            keep = sorted(
                self.asset.rules, key=lambda r: (-r.gain, r.asset_version)
            )[: settings.MAX_RULES]
            keep_ids = {id(r) for r in keep}
            pruned = [r for r in self.asset.rules if id(r) not in keep_ids]
            self.asset.rules = [r for r in self.asset.rules if id(r) in keep_ids]

        if added or pruned:
            self.asset.version += 1
            self.asset.updated_at = datetime.now().isoformat(timespec="seconds")
            self.save()
        return added, pruned

    # ---------- 快照与回滚 ----------

    def snapshot(self) -> PolicyAsset:
        """θ 的深拷贝。训练开始前留档，用于变差时自动回滚。"""
        return self.asset.model_copy(deep=True)

    def restore(self, snap: PolicyAsset) -> int:
        """回滚到某个快照，返回回滚后的版本号。

        版本号递增：回滚本身也是一次写入。
        """
        current = self.asset.version
        self.asset = snap.model_copy(deep=True)
        self.asset.version = current + 1
        self.asset.updated_at = datetime.now().isoformat(timespec="seconds")
        self.save()
        return self.asset.version
