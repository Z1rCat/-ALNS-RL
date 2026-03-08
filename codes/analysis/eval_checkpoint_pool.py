from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from checkpoint_eval_common import summarize_run_dir


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT_DIR / "codes" / "analysis" / "outputs" / "checkpoint_eval"


def _default_python_exe() -> str:
    candidates = [
        ROOT_DIR / "codes" / "env" / "python.exe",
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(sys.executable)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints through Dynamic_master34959.py eval_only and/or summarize existing run directories."
    )
    parser.add_argument("--run-dir", action="append", default=[], help="existing run directory to summarize; repeatable")
    parser.add_argument("--checkpoint", action="append", default=[], help="checkpoint zip to evaluate; repeatable")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--summary-name", type=str, default="checkpoint_eval_summary.csv")
    parser.add_argument("--manifest-name", type=str, default="checkpoint_eval_manifest.json")
    parser.add_argument("--dist-name", type=str, default="")
    parser.add_argument("--request-number", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="PPO_NEW")
    parser.add_argument("--algo-version", type=str, default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python-exe", type=str, default=_default_python_exe())
    parser.add_argument("--run-prefix", type=str, default="ckpt_eval")
    parser.add_argument(
        "--external-data-root",
        type=str,
        default="",
        help="optional existing data root to reuse during eval; when set, generator is skipped",
    )
    parser.add_argument("--phase-name", type=str, default="implement")
    parser.add_argument("--hard-stage-family", type=str, default="removal")
    parser.add_argument("--hard-min-severity", type=int, default=5)
    parser.add_argument("--easy-max-severity", type=int, default=3)
    parser.add_argument("--lambda-fp", type=float, default=0.20)
    parser.add_argument(
        "--reuse-existing-run-dir",
        action="store_true",
        help="if a generated eval run directory already exists, reuse it instead of failing",
    )
    parser.add_argument(
        "--allow-partial-on-timeout",
        action="store_true",
        help="if checkpoint eval times out, keep the partial run directory and summarize whatever was written",
    )
    parser.add_argument("--timeout-sec", type=int, default=0, help="optional subprocess timeout; 0 disables timeout")
    return parser.parse_args()


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return token or "item"


def _extra_python_paths(python_exe: str) -> list[str]:
    python_path = Path(str(python_exe)).resolve()
    env_python = (ROOT_DIR / "codes" / "env" / "python.exe").resolve()
    candidates = [ROOT_DIR / "codes"]
    if python_path != env_python:
        candidates.append(ROOT_DIR / "codes" / "env" / "Lib" / "site-packages")
    return [str(path) for path in candidates if path.exists()]


def _summarize(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    run_label: str | None = None,
    source_kind: str,
    checkpoint_path: str = "",
) -> dict[str, Any]:
    row = summarize_run_dir(
        run_dir,
        run_label=run_label,
        phase_name=str(args.phase_name),
        hard_stage_family=str(args.hard_stage_family),
        hard_min_severity=int(args.hard_min_severity),
        easy_max_severity=int(args.easy_max_severity),
        lambda_fp=float(args.lambda_fp),
    )
    row["source_kind"] = str(source_kind)
    row["checkpoint_path"] = str(checkpoint_path)
    return row


def _run_eval_for_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    *,
    index: int,
    out_dir: Path,
) -> tuple[Path, str]:
    if not str(args.dist_name).strip() or int(args.request_number) <= 0:
        raise ValueError("--dist-name and --request-number are required when --checkpoint is used")

    run_token = _sanitize_token(checkpoint_path.stem)
    run_dir = out_dir / "runs" / f"{int(index):03d}_{run_token}"
    if run_dir.exists():
        if not bool(args.reuse_existing_run_dir):
            raise FileExistsError(
                f"Eval run directory already exists: {run_dir}. Use --reuse-existing-run-dir to reuse it."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python_exe),
        str(ROOT_DIR / "codes" / "Dynamic_master34959.py"),
        "--dist_name",
        str(args.dist_name),
        "--request_number",
        str(int(args.request_number)),
        "--workers",
        str(int(args.workers)),
        "--algorithm",
        str(args.algorithm),
        "--algo_version",
        str(args.algo_version),
        "--seed",
        str(int(args.seed)),
        "--stage-mode",
        "eval_only",
        "--init-model-path",
        str(checkpoint_path.resolve()),
        "--run-name",
        str(run_dir.resolve()),
    ]
    external_data_root = str(args.external_data_root or "").strip()
    if external_data_root:
        external_path = Path(external_data_root).resolve()
        if not external_path.exists():
            raise FileNotFoundError(f"external data root does not exist: {external_path}")
        cmd.extend(
            [
                "--skip-generator",
                "--external-data-root",
                str(external_path),
            ]
        )

    env = os.environ.copy()
    extra_paths = _extra_python_paths(str(args.python_exe))
    if extra_paths:
        existing = str(env.get("PYTHONPATH", "") or "").strip()
        merged = extra_paths + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(merged)
    timeout = None if int(args.timeout_sec) <= 0 else int(args.timeout_sec)
    try:
        subprocess.run(cmd, cwd=str(ROOT_DIR), env=env, check=True, timeout=timeout)
        return run_dir, "ok"
    except subprocess.TimeoutExpired:
        if not bool(args.allow_partial_on_timeout):
            raise
        return run_dir, f"timeout_{int(args.timeout_sec)}s"


def main() -> None:
    args = _parse_args()
    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for raw_run_dir in args.run_dir:
        run_dir = Path(str(raw_run_dir)).resolve()
        row = _summarize(run_dir, args, source_kind="run_dir")
        row["subprocess_status"] = "existing"
        rows.append(row)
        manifest_runs.append(
            {
                "source_kind": "run_dir",
                "run_dir": str(run_dir),
                "checkpoint_path": "",
                "subprocess_status": "existing",
                "status": row.get("status", ""),
            }
        )

    for idx, raw_checkpoint in enumerate(args.checkpoint, start=1):
        checkpoint_path = Path(str(raw_checkpoint)).resolve()
        run_dir, subprocess_status = _run_eval_for_checkpoint(checkpoint_path, args, index=idx, out_dir=out_dir)
        row = _summarize(
            run_dir,
            args,
            run_label=run_dir.name,
            source_kind="checkpoint",
            checkpoint_path=str(checkpoint_path),
        )
        row["subprocess_status"] = str(subprocess_status)
        rows.append(row)
        manifest_runs.append(
            {
                "source_kind": "checkpoint",
                "checkpoint_path": str(checkpoint_path),
                "run_dir": str(run_dir),
                "subprocess_status": str(subprocess_status),
                "status": row.get("status", ""),
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        sort_cols = [c for c in ["source_kind", "distribution", "run_label"] if c in summary_df.columns]
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
    summary_path = out_dir / str(args.summary_name)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    manifest = {
        "status": "ok",
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "phase_name": str(args.phase_name),
        "external_data_root": str(args.external_data_root or ""),
        "hard_stage_family": str(args.hard_stage_family),
        "hard_min_severity": int(args.hard_min_severity),
        "easy_max_severity": int(args.easy_max_severity),
        "lambda_fp": float(args.lambda_fp),
        "runs": manifest_runs,
    }
    (out_dir / str(args.manifest_name)).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
