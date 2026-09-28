"""跨模块复用的小工具：UTC 时间戳与耗时。"""

import time
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
