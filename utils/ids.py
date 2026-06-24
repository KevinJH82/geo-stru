"""任务编码生成。

task_code：单次分析任务的全局唯一编码，用于出问题后按编码回溯到具体分析
（磁盘产物目录 / metadata.json / 该任务的全部日志行）。

形如 GST-20260624-125430-a1f9：可读、可按时间排序、跨重启与并发唯一。
与 commons/trace/ids.py 的 trace_id 语义不同——trace_id 是跨服务请求血缘，
task_code 是单次分析任务编码。本地时间戳与 run_id 的 datetime.now() 保持一致。
"""

from __future__ import annotations

import secrets
from datetime import datetime


def new_task_code(prefix: str = "GST") -> str:
    """生成全局唯一任务编码：<prefix>-YYYYMMDD-HHMMSS-<4位hex>。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"
