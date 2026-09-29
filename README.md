# AI4S Agent

把自然语言需求，经**语义增强**变成规范的 Wolfram Language
表达式，交给 **Wolfram 计算层**执行，再用自然语言把结果讲清楚 —— 并让这条链路**越用越准**。

```
用户输入
  ↓
[Step1 路由探针 WolframContext]  查得到 → codegen；查不到 → chat
  ↓
[Step2 第一层语义增强 πθ] ⇄ 工具：<lookup> 查文档 / <alpha> 查 W|A 参考 / <attempt> 试执行
  ↕                                                        ⇅ 同一个 MCP
[Step3 结果后处理] ← [第二层 Wolfram 官方 MCP 执行] —— 失败或未通过就回 Step2 修正
  ↓
[Step4 自然语言总结] → [Step5 会话记忆]
  ↓
runs/*.jsonl 审计 → GRPO + Self-Judge 写回 θ
```

**路由不看关键词**：Step 1 先拿 `WolframContext` 探一次，查得到内容就走 codegen 闭环，
查不到（MCP 返回空，或只剩 `No Results Found` 占位）就走通用对话。判据来自 Wolfram
本身，所以换学科不用改代码 —— 关键词表要按学科无限扩张，那是维护不完的。
这一探还顺带把官方文档链接里的真实符号名沉淀进 `assets/system_names.json`，
查得越多，「函数真实性」信号越少需要现场打 MCP。

奖励由三部分构成，**Wolfram 资源利用率占多数（0.55）**，其次才是可验证信号（0.30）
与 LLM judge（0.15）——「是不是真在用 Wolfram、用对没有」是训练的第一目标。


## 快速开始

```bash
cd AI4S
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 配置：新建 .env，至少填 DEEPSEEK_API_KEY。
#    第二层的 Wolfram 官方 MCP 无需鉴权、无需 AppID、无需本机装引擎。

# 2) 启动服务（前端与 API 同进程）
python -m backend.main        # 或 uvicorn backend.main:app --port 8000
# 打开 http://localhost:8000

# 3) 离线自检（不需要 API Key，不联网）
python tests/test_pipeline.py
```
---

## 什么是 Wolfram 资源利用率

`backend/core/judge.py` 的 `_utilization()` 用五个**可离线复核**的信号等权求均值，不交给 LLM 打分：

| 信号 | 判定 | 为什么 |
|---|---|---|
| channel | 结果确实出自 Wolfram MCP（`exec_strategy == "mcp"`） | 网页抓来的文本、压根没执行都算 0 |
| docs | 主动调过 `wolfram_context` 查官方文档 | 官方文档检索是「用上资源」的直接证据 |
| result | 执行成功且拿回非空结果 | `$Failed` 在 executor 里已判成 `ok=False` |
| functions | 表达式里的函数名真的是 `System\`` 符号，按比例给分 | 编造函数名会被点名，蒸馏时才知道该改什么 |
| rich | 结果带回了图像（`image_count > 0`），或最终表达式是多视图形态 | 区分「用了」和「用透了」：裸符号解 0.6，富答案 1.0 |

---

## 目录

```
AI4S/
├── backend/
│   ├── main.py                 FastAPI 装配 / 前端托管 / /health
│   ├── core/
│   │   ├── config.py           全部配置（MCP 端点、权重、自治边界）
│   │   ├── models.py           六元组 + Trajectory/StepRecord + θ + API 契约
│   │   ├── llm.py              LLM 客户端唯一构造点
│   │   ├── utils.py            时间戳 / 耗时
│   │   ├── mcp.py              Wolfram 官方 MCP 客户端（执行 + 文档检索的唯一通道）
│   │   ├── enhancer.py         第一层：πθ 采样 + WlEnv 闭环 + 输入路由
│   │   ├── executor.py         第二层：Wolfram MCP 单通道执行
│   │   ├── safety.py           Wolfram 禁止操作清单闸门
│   │   ├── asset.py            θ 的存储、写入闸门、快照与回滚
│   │   ├── judge.py            三层奖励：利用率 + verifier + LLM judge
│   │   ├── grpo.py             组内优势 + 蒸馏写回 + 外部验证 + 自治边界
│   │   ├── trace.py            JSONL 审计与**会话记忆**
│   │   └── tasks.py            任务集（可验证奖励的来源）
│   └── routes/                 execute（Step1-5）/ training
├── frontend/index.html         极简单页（见下方「与 cheese 的关系」）
├── runs/                       traj-*.jsonl / run-*.jsonl / rule-*.jsonl
└── assets/                     codegen.json（θ）+ system_names.json（符号真实性缓存，
                                每次 WolframContext 查询都会把新学到的符号名追加进去）
```

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/execute` | 走完整 Step 1-5 流水线。**SSE**：先推 `{stage, label, detail, payload?}` 阶段进度（`nl` → `enhance` → `wolfram` → `summary`，前端用它点亮页头那条思维链，并把 `payload` 里的探针原文 / 改写出的 WL / Summary 思维链就地摊开），最后一条 `{stage:"result", data}` 带模式、增强表达式、执行结果（含图像）、利用率与自然语言总结 |
| GET | `/runs` | 最近的轨迹 / 训练运行 / 规则写入清单 |
| GET | `/runs/{trajectory_id}` | 取回一条完整轨迹（逐步的动作、环境状态、判定依据、计量） |
| POST | `/training/start` | 启动后台训练 |
| GET | `/training/status` | 进度、平均奖励、成功率、**平均利用率**、θ 版本、升级人工事件 |
| GET | `/training/result` | 训练前后 eval 对照、回滚原因、token 计量 |
| POST | `/training/stop` | 请求停止（当前 group 结束后退出） |
| GET | `/training/asset` | 当前 θ：base_prompt + 规则库（每条规则含来源轨迹与步号） |
| POST | `/training/asset/reset` | 清空规则库回到 θ₀（保留手写 base_prompt） |
| GET | `/health` | LLM 是否就绪、**Wolfram MCP 是否可达** |

训练是后台协程，`/training/status` 里 `needs_human=true` 表示有事件需要人工介入：
`safety_violation` / `regression` / `no_learning_signal` / `llm_failure` / `budget_exceeded` /
`unexpected_error`。