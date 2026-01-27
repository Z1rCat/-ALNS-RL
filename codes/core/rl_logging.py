"""
轻量级日志工具，用于追加 CSV 行到 codes/logs 目录。
"""
import csv
import time
import json
import threading
from pathlib import Path

LOG_ROOT = Path(__file__).resolve().parent.parent / "logs"
RUN_DIR = LOG_ROOT
_LOCK = threading.Lock()


def set_run_dir(run_name: str):
    """
    配置本次运行的日志目录，如 logs/run_xxx
    """
    global RUN_DIR
    RUN_DIR = LOG_ROOT / run_name
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "data").mkdir(parents=True, exist_ok=True)


def get_run_dir():
    return RUN_DIR


def get_run_data_dir():
    data_dir = RUN_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def write_meta(meta: dict):
    """
    写入本次运行的元信息（分布、R 等）
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RUN_DIR / "meta.json"
    try:
        with _LOCK:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def append_row(file_name, fieldnames, row_dict):
    """
    追加一行到指定 CSV 文件。缺失字段补空字符串。
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RUN_DIR / file_name
    row = {k: row_dict.get(k, "") for k in fieldnames}
    with _LOCK:
        file_exists = file_path.exists()
        with file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def now_ts():
    return time.time()
