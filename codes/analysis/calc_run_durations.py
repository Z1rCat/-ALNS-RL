import os
import sys
import csv
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict

# 你的 ts 是 Unix epoch 秒（带小数）
# 默认按本地时区显示；如想固定中国时间可改为 timezone(timedelta(hours=8))

TRACE_CANDIDATES = [
    "rl_trace.csv",
    "rl_trace.CSV",
    "rl_trace",         # 有些系统会省略扩展名
]

def find_trace_file(run_dir: str) -> Optional[str]:
    # 优先找明确的文件名
    for name in TRACE_CANDIDATES:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p

    # 兜底：找名字里包含 rl_trace 且是 csv 的
    for fn in os.listdir(run_dir):
        low = fn.lower()
        if "rl_trace" in low and low.endswith(".csv"):
            p = os.path.join(run_dir, fn)
            if os.path.isfile(p):
                return p
    return None


def read_first_last_ts(csv_path: str) -> Tuple[Optional[float], Optional[float], int, Optional[str]]:
    """
    返回：first_ts, last_ts, row_count, err_msg
    """
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None, None, 0, "empty_header"

            # 兼容列名大小写
            field_map = {name.lower(): name for name in reader.fieldnames}
            if "ts" not in field_map:
                return None, None, 0, f"no_ts_column fields={reader.fieldnames}"

            ts_col = field_map["ts"]
            first_ts = None
            last_ts = None
            n = 0

            for row in reader:
                n += 1
                v = row.get(ts_col, "")
                if v is None:
                    continue
                v = str(v).strip()
                if not v:
                    continue
                try:
                    t = float(v)
                except ValueError:
                    continue

                if first_ts is None:
                    first_ts = t
                last_ts = t

            return first_ts, last_ts, n, None
    except Exception as e:
        return None, None, 0, f"read_error:{type(e).__name__}:{e}"


def fmt_dt(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    # 用本地时区显示
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def iter_run_dirs(root: str) -> List[str]:
    """
    认为 root 目录下：
    - 可能直接就是一个 run 目录（包含 rl_trace）
    - 也可能包含多个 run_xxx 子目录
    """
    root = os.path.abspath(root)
    # 若 root 本身就是 run_dir
    if any(os.path.isfile(os.path.join(root, x)) for x in TRACE_CANDIDATES) or any("rl_trace" in f.lower() for f in os.listdir(root)):
        return [root]

    # 否则扫描一级子目录
    run_dirs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            # 你这里的 run 命名一般以 run_ 开头
            if name.startswith("run_") or True:
                run_dirs.append(p)
    return sorted(run_dirs)


def main():
    if len(sys.argv) < 2:
        print("用法: python calc_run_durations.py <RUN_ROOT_DIR> [--out out.csv]")
        print(r"示例: python calc_run_durations.py A:\MYdata\logs\run_20260124_190710_107430_R5_S3_1_DQN_S123 --out durations.csv")
        sys.exit(1)

    root = sys.argv[1]
    out_csv = None
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        if idx + 1 < len(sys.argv):
            out_csv = sys.argv[idx + 1]

    rows: List[Dict[str, str]] = []
    run_dirs = iter_run_dirs(root)

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir.rstrip("\\/"))
        trace_path = find_trace_file(run_dir)
        if not trace_path:
            rows.append({
                "run": run_name,
                "run_dir": run_dir,
                "trace_file": "",
                "rows": "0",
                "start_ts": "",
                "end_ts": "",
                "start_time": "",
                "end_time": "",
                "duration_sec": "",
                "duration_hms": "",
                "status": "NO_TRACE",
            })
            continue

        first_ts, last_ts, n, err = read_first_last_ts(trace_path)
        if err or first_ts is None or last_ts is None:
            rows.append({
                "run": run_name,
                "run_dir": run_dir,
                "trace_file": os.path.basename(trace_path),
                "rows": str(n),
                "start_ts": "" if first_ts is None else str(first_ts),
                "end_ts": "" if last_ts is None else str(last_ts),
                "start_time": fmt_dt(first_ts),
                "end_time": fmt_dt(last_ts),
                "duration_sec": "",
                "duration_hms": "",
                "status": f"BAD_TRACE:{err or 'missing_ts'}",
            })
            continue

        dur = last_ts - first_ts
        # 异常保护：如果出现负数，说明 ts 不单调或文件混乱
        status = "OK"
        if dur < 0:
            status = "TS_NON_MONOTONIC"
        rows.append({
            "run": run_name,
            "run_dir": run_dir,
            "trace_file": os.path.basename(trace_path),
            "rows": str(n),
            "start_ts": f"{first_ts:.6f}",
            "end_ts": f"{last_ts:.6f}",
            "start_time": fmt_dt(first_ts),
            "end_time": fmt_dt(last_ts),
            "duration_sec": f"{dur:.3f}",
            "duration_hms": fmt_hms(dur),
            "status": status,
        })

    # 按 duration_sec 降序（空的排后）
    def key_fn(r: Dict[str, str]):
        try:
            return float(r["duration_sec"])
        except Exception:
            return -1.0

    rows_sorted = sorted(rows, key=key_fn, reverse=True)

    # 打印一个简洁的控制台汇总
    print(f"扫描目录: {os.path.abspath(root)}")
    print(f"发现 run 数: {len(rows_sorted)}")
    print("")
    print("Top 20 (按时长降序):")
    for r in rows_sorted[:20]:
        print(f"{r['duration_hms']:>10}  {r['status']:<18}  {r['run']}")

    # 输出 CSV
    if out_csv:
        out_csv = os.path.abspath(out_csv)
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(rows_sorted)
        print("")
        print(f"已写出: {out_csv}")

if __name__ == "__main__":
    main()
