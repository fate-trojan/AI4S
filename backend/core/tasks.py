"""任务集：可验证奖励的来源。

每个任务给两类校验（见 models.TaskSpec）：

    expr_checks   必须出现在「增强后的 WL 表达式」里 —— 离线就能判，用于约束
                  「函数选对了吗、省略成分补了吗」
    result_checks 必须出现在第二层的执行结果里 —— 需要 Wolfram MCP 可达

result_checks 里的子串都是在真实 MCP 上跑出来的结果里截的（例如 100 个质数确实是 541）。
表达式类的任务（画图）拿不到文本结果，就只留 expr_checks。

split="eval" 的任务不参与训练，只在训练前后各评一次，且与 train 刻意不重叠。
"""

from typing import Dict, List

from backend.core.models import TaskSpec

_RAW = [
    # ---------------- 训练集 ----------------
    dict(
        id="plot_sin",
        split="train",
        difficulty="easy",
        prompt="画个正弦曲线",
        # {x, 强制补全区间；Plot/Sin 强制选对函数
        expr_checks=["Plot", "Sin", "{x,"],
        # 图形结果在 MCP 里回显成空的 `Out[1]=`，没有可匹配的文本，只留表达式校验
    ),
    dict(
        id="matrix_eigen",
        split="train",
        difficulty="easy",
        prompt="这个矩阵 {{1, 2}, {3, 4}} 的特征值是多少",
        expr_checks=["Eigenvalues"],
        result_checks=["Sqrt[33]"],  # → {(5 + Sqrt[33])/2, (5 - Sqrt[33])/2}
    ),
    dict(
        id="definite_integral",
        split="train",
        difficulty="medium",
        prompt="帮我算 x^2 Sin[x] 在 0 到 Pi 上的定积分",
        expr_checks=["Integrate", "Sin", "0", "Pi"],
        result_checks=["Pi^2"],  # → -4 + Pi^2
    ),
    dict(
        id="solve_quadratic",
        split="train",
        difficulty="easy",
        prompt="解方程 x^2 - 3 x + 2 == 0",
        expr_checks=["Solve"],
        result_checks=["x -> 1"],  # → {{x -> 1}, {x -> 2}}
    ),
    dict(
        id="dsolve_ode",
        split="train",
        difficulty="medium",
        prompt="求解微分方程 y''[x] + y[x] == 0",
        expr_checks=["DSolve", "y"],
        # 通解里的正弦项。这正是原网页通道拿不到、MCP 才给得出的符号解
        result_checks=["Cos[x]"],
    ),
    dict(
        id="fourier_transform",
        split="train",
        difficulty="medium",
        prompt="求 f(t) = t 的傅里叶变换",
        expr_checks=["FourierTransform"],
        result_checks=["DiracDelta"],
    ),
    dict(
        id="beijing_weather",
        split="train",
        difficulty="medium",
        prompt="北京今天天气怎么样",
        # 补全实体：必须落到 Entity["City", ...] 且带 Beijing
        expr_checks=["WeatherData", "Beijing"],
        result_checks=["DegreesCelsius"],  # 温度值每天都变，只校验量纲
    ),
    # ---------------- 评测集（不参与训练） ----------------
    dict(
        id="describe_stats",
        split="eval",
        difficulty="easy",
        prompt="算一下这组数据的均值和标准差 {3, 1, 4, 1, 5, 9, 2, 6}",
        expr_checks=["Mean", "StandardDeviation"],
        result_checks=["31/8"],  # 均值
    ),
    dict(
        id="nth_prime",
        split="eval",
        difficulty="easy",
        prompt="第 100 个质数是多少",
        expr_checks=["Prime"],
        result_checks=["541"],
    ),
    dict(
        id="country_capital",
        split="eval",
        difficulty="medium",
        prompt="法国的首都是哪里",
        expr_checks=["Entity", "France"],
        result_checks=["Paris"],
    ),
]

TASKS: List[TaskSpec] = [TaskSpec(**r) for r in _RAW]
_BY_ID: Dict[str, TaskSpec] = {t.id: t for t in TASKS}


def train_tasks() -> List[TaskSpec]:
    return [t for t in TASKS if t.split == "train"]


def eval_tasks() -> List[TaskSpec]:
    return [t for t in TASKS if t.split == "eval"]


def get_task(task_id: str) -> TaskSpec:
    if task_id not in _BY_ID:
        raise KeyError(f"未知 task_id: {task_id}（可用：{', '.join(_BY_ID)}）")
    return _BY_ID[task_id]
