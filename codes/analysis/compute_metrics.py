import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


INVALID_REWARD = -10000000

ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"
SUMMARY_DIR = LOG_ROOT / "summary"
SUMMARY_CSV = SUMMARY_DIR / "metrics_summary.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _coerce_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _extract_rewards_from_trace(df: pd.DataFrame, *, name: str) -> pd.Series:
    required = {"phase", "stage", "reward"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    phase = _coerce_lower(df["phase"])
    stage = _coerce_lower(df["stage"])
    filtered = df[(phase == "implement") & (stage == "receive_reward")].copy()
    if filtered.empty:
        raise ValueError(f"{name}: no rows where phase=='implement' and stage=='receive_reward'")

    if "source" in filtered.columns:
        src = filtered["source"].astype(str).str.strip().str.upper()
        filtered_rl = filtered[src == "RL"]
        if not filtered_rl.empty:
            filtered = filtered_rl

    rewards = pd.to_numeric(filtered["reward"], errors="coerce")
    if rewards.isna().any():
        raise ValueError(f"{name}: reward contains NaN after numeric coercion")
    if (rewards == INVALID_REWARD).any():
        raise ValueError(f"{name}: reward contains INVALID_REWARD={INVALID_REWARD}")
    return rewards.astype(float)


def _extract_rewards_from_training(df: pd.DataFrame, *, name: str) -> pd.Series:
    required = {"phase", "reward"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    phase = _coerce_lower(df["phase"])
    rewards = pd.to_numeric(df["reward"], errors="coerce")
    filtered = rewards[phase == "implement"].dropna()
    if filtered.empty:
        raise ValueError(f"{name}: no reward rows where phase=='implement'")
    if (filtered == INVALID_REWARD).any():
        raise ValueError(f"{name}: reward contains INVALID_REWARD={INVALID_REWARD}")
    return filtered.astype(float)


def _extract_rewards_from_baseline(df: pd.DataFrame, *, name: str) -> pd.Series:
    required = {"phase", "stage", "reward"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    data = df.copy()
    if "source" in data.columns and (data["source"].astype(str).str.strip().str.upper() == "BASELINE").any():
        data = data[data["source"].astype(str).str.strip().str.upper() == "BASELINE"]

    phase = _coerce_lower(data["phase"])
    stage = _coerce_lower(data["stage"])
    data["reward"] = pd.to_numeric(data["reward"], errors="coerce")

    implement = data[phase == "implement"]
    if implement.empty:
        raise ValueError(f"{name}: no rows where phase=='implement'")

    implement_stage = _coerce_lower(implement["stage"])
    if (implement_stage == "receive_reward").any():
        filtered = implement[implement_stage == "receive_reward"]
    else:
        filtered = implement[implement_stage == "finish_removal"]

    if filtered.empty:
        raise ValueError(f"{name}: no usable reward rows in implement phase (receive_reward/finish_removal)")

    rewards = pd.to_numeric(filtered["reward"], errors="coerce")
    rewards = rewards.dropna()
    if rewards.empty:
        raise ValueError(f"{name}: reward is empty after numeric coercion")
    if (rewards == INVALID_REWARD).any():
        raise ValueError(f"{name}: reward contains INVALID_REWARD={INVALID_REWARD}")
    return rewards.astype(float)


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(meta_path.read_text(encoding="utf-8-sig"))


def _acquire_lock(lock_path: Path, timeout_s: float = 120.0, poll_s: float = 0.2) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os_open_exclusive(lock_path)
            os_close_fd(fd)
            return
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
            time.sleep(poll_s)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def os_open_exclusive(path: Path) -> int:
    import os

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    return os.open(str(path), flags)


def os_close_fd(fd: int) -> None:
    import os

    os.close(fd)


def _write_summary_row(summary_csv: Path, row: Dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_csv.with_suffix(summary_csv.suffix + ".lock")

    desired_fieldnames = [
        "run_id",
        "scenario",
        "algorithm",
        "seed",
        "request_number",
        "n_test",
        "G",
        "G_prime",
        "Adv0",
        "Adv1",
        "AdvRand",
        "wall_time_sec",
        "cpu_percent_avg",
        "cpu_percent_peak",
        "ram_used_gb_peak",
        "gpu_util_percent_avg",
        "gpu_util_percent_peak",
        "gpu_mem_used_mb_peak",
    ]

    _acquire_lock(lock_path)
    try:
        existing_fieldnames: Optional[list] = None
        existing_rows: list = []
        if summary_csv.exists():
            with summary_csv.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames
                for r in reader:
                    existing_rows.append(r)

        if existing_fieldnames:
            fieldnames = list(existing_fieldnames)
            for name in desired_fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
        else:
            fieldnames = list(desired_fieldnames)

        needs_rewrite = bool(existing_fieldnames) and list(existing_fieldnames) != fieldnames
        if needs_rewrite:
            import os

            tmp_path = summary_csv.with_suffix(summary_csv.suffix + ".tmp")
            with tmp_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow({k: r.get(k) for k in fieldnames})
                writer.writerow({k: row.get(k) for k in fieldnames})
            os.replace(tmp_path, summary_csv)
        else:
            needs_header = (not summary_csv.exists()) or summary_csv.stat().st_size == 0
            with summary_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if needs_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k) for k in fieldnames})
    finally:
        _release_lock(lock_path)


def compute_metrics(run_dir: Path, *, summary_csv: Path = SUMMARY_CSV) -> Dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    meta = _load_meta(run_dir)
    run_id = str(meta.get("run_name") or run_dir.name)
    scenario = meta.get("distribution")
    algorithm = meta.get("algorithm")
    seed = meta.get("seed")
    request_number = meta.get("request_number")

    warnings = []

    trace_path = run_dir / "rl_trace.csv"
    training_path = run_dir / "rl_training.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"rl_trace.csv not found: {trace_path}")
    if not training_path.exists():
        raise FileNotFoundError(f"rl_training.csv not found: {training_path}")

    rl_reward_source = "rl_trace.csv (implement/receive_reward)"
    try:
        rewards_rl = _extract_rewards_from_trace(_read_csv(trace_path), name="RL")
    except Exception as exc:
        warnings.append(f"RL rewards fallback to rl_training.csv: {exc}")
        rl_reward_source = "rl_training.csv (phase=implement, reward)"
        rewards_rl = _extract_rewards_from_training(_read_csv(training_path), name="RL_training")

    strict = os.environ.get("METRICS_STRICT", "0").strip() == "1"
    n_test = int(rewards_rl.shape[0])

    def maybe_load(name: str, path: Path) -> Tuple[Optional[pd.Series], Optional[str]]:
        if not path.exists():
            return None, f"missing {name}: {path.name}"
        rewards = _extract_rewards_from_baseline(_read_csv(path), name=name)
        if strict and int(rewards.shape[0]) != n_test:
            raise ValueError(f"{name}: n_test mismatch (RL={n_test}, {name}={int(rewards.shape[0])})")
        if int(rewards.shape[0]) != n_test:
            warnings.append(f"{name}: n_test mismatch (RL={n_test}, {name}={int(rewards.shape[0])}) -> trim to min")
        return rewards, None

    r_wait_series, warn = maybe_load("Always_Wait", run_dir / "baseline_wait.csv")
    if warn:
        warnings.append(warn)
    r_reroute_series, warn = maybe_load("Always_Reroute", run_dir / "baseline_reroute.csv")
    if warn:
        warnings.append(warn)
    r_random_series, warn = maybe_load("Random", run_dir / "baseline_random.csv")
    if warn:
        warnings.append(warn)

    lengths = [int(rewards_rl.shape[0])]
    for series in (r_wait_series, r_reroute_series, r_random_series):
        if series is not None:
            lengths.append(int(series.shape[0]))
    n_test = int(min(lengths)) if lengths else 0
    rewards_rl = rewards_rl.iloc[:n_test]

    r_rl = float(rewards_rl.sum())
    r_wait = float(r_wait_series.iloc[:n_test].sum()) if r_wait_series is not None else None
    r_reroute = float(r_reroute_series.iloc[:n_test].sum()) if r_reroute_series is not None else None
    r_random = float(r_random_series.iloc[:n_test].sum()) if r_random_series is not None else None

    adv0 = (r_rl - r_wait) / n_test if r_wait is not None and n_test > 0 else None
    adv1 = (r_rl - r_reroute) / n_test if r_reroute is not None and n_test > 0 else None
    adv_rand = (r_random / n_test) if r_random is not None and n_test > 0 else None

    g_prime = (adv0 + adv1) if (adv0 is not None and adv1 is not None) else None
    g = (g_prime - adv_rand) if (g_prime is not None and adv_rand is not None) else None

    resource_usage = None
    resource_path = run_dir / "resource_usage.json"
    if resource_path.exists():
        try:
            resource_usage = json.loads(resource_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            resource_usage = json.loads(resource_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            warnings.append(f"failed to read resource_usage.json: {exc}")
            resource_usage = None

    metrics = {
        "run_id": run_id,
        "scenario": scenario,
        "algorithm": algorithm,
        "seed": seed,
        "request_number": request_number,
        "n_test": n_test,
        "reward_source_rl": rl_reward_source,
        "R_RL": r_rl,
        "R_wait": r_wait,
        "R_reroute": r_reroute,
        "R_random": r_random,
        "Adv0": adv0,
        "Adv1": adv1,
        "AdvRand": adv_rand,
        "G_prime": g_prime,
        "G": g,
        "meta": meta,
        "warnings": warnings,
        "resource_usage": resource_usage,
    }

    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_summary_row(
        summary_csv,
        {
            "run_id": run_id,
            "scenario": scenario,
            "algorithm": algorithm,
            "seed": seed,
            "request_number": request_number,
            "n_test": n_test,
            "G": g,
            "G_prime": g_prime,
            "Adv0": adv0,
            "Adv1": adv1,
            "AdvRand": adv_rand,
            "wall_time_sec": (resource_usage or {}).get("wall_time_sec") if isinstance(resource_usage, dict) else None,
            "cpu_percent_avg": (resource_usage or {}).get("cpu_percent_avg") if isinstance(resource_usage, dict) else None,
            "cpu_percent_peak": (resource_usage or {}).get("cpu_percent_peak")
            if isinstance(resource_usage, dict)
            else None,
            "ram_used_gb_peak": (resource_usage or {}).get("ram_used_gb_peak")
            if isinstance(resource_usage, dict)
            else None,
            "gpu_util_percent_avg": (resource_usage or {}).get("gpu_util_percent_avg")
            if isinstance(resource_usage, dict)
            else None,
            "gpu_util_percent_peak": (resource_usage or {}).get("gpu_util_percent_peak")
            if isinstance(resource_usage, dict)
            else None,
            "gpu_mem_used_mb_peak": (resource_usage or {}).get("gpu_mem_used_mb_peak")
            if isinstance(resource_usage, dict)
            else None,
        },
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute G/Adv metrics for a completed run_dir.")
    parser.add_argument("--run-dir", required=True, help="Run directory (contains rl_trace.csv / baseline_*.csv).")
    parser.add_argument(
        "--summary-csv",
        default=str(SUMMARY_CSV),
        help="Global summary CSV to append (default: codes/logs/summary/metrics_summary.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_metrics(Path(args.run_dir), summary_csv=Path(args.summary_csv))
    except Exception as exc:
        print(f"[metrics] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
