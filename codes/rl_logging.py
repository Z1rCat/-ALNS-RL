"""
轻量级日志工具，用于追加 CSV 行到 codes/logs 目录。
"""
import csv
import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


def _ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_row(file_name, fieldnames, row_dict):
    """
    追加一行到指定 CSV 文件。缺失字段补空字符串。
    """
    _ensure_dir()
    file_path = LOG_DIR / file_name
    file_exists = file_path.exists()
    row = {k: row_dict.get(k, "") for k in fieldnames}
    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def now_ts():
    return time.time()
