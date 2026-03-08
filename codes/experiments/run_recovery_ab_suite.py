from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
ROOT_DIR = CODES_DIR.parent
LOGS_DIR = CODES_DIR / "logs"


CONTROLLED_FLAGS_WITH_VALUE = {
    "--run-id",
    "--phase1-history-every-tables",
    "--phase1-external-data-root",
    "--outer-base-ckpt",
    "--actor-rollback-manifest-path",
    "--rollback-manifest-path",
}

CONTROLLED_FLAGS_NO_VALUE = {
    "--phase1-skip",
    "--actor-rollback-enable",
    "--no-actor-rollback-enable",
    "--rollback-enable",
    "--no-rollback-enable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a comparable three-branch recovery suite over the current pipeline: "
            "main baseline, A (actor rollback), and B (checkpoint rollback)."
        )
    )
    parser.add_argument("--suite-root", type=str, default="recovery_ab_suite")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--phase1-history-every-tables", type=int, default=10)
    parser.add_argument("--skip-main", action="store_true", default=False)
    parser.add_argument("--skip-a", action="store_true", default=False)
    parser.add_argument("--skip-b", action="store_true", default=False)
    parser.add_argument("--skip-summary", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args, passthrough = parser.parse_known_args()
    args.pipeline_passthrough = passthrough
    return args


def _resolve_suite_root(raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if not path:
        raise ValueError("empty --suite-root")
    if path.is_absolute():
        return path.resolve()
    return (LOGS_DIR / path).resolve()


def _print_cmd(tag: str, cmd: Sequence[str]) -> None:
    pretty = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[recovery-ab][{tag}] {pretty}")


def _run_cmd(tag: str, cmd: Sequence[str], *, env: dict[str, str], dry_run: bool) -> int:
    _print_cmd(tag, cmd)
    if dry_run:
        return 0
    return int(subprocess.run(list(cmd), cwd=str(ROOT_DIR), env=env, check=False).returncode)


def _strip_controlled_passthrough(tokens: Iterable[str]) -> List[str]:
    raw = [str(token) for token in tokens]
    out: List[str] = []
    i = 0
    while i < len(raw):
        token = raw[i]
        if token in CONTROLLED_FLAGS_NO_VALUE:
            i += 1
            continue
        if token in CONTROLLED_FLAGS_WITH_VALUE:
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _phase1_ckpt_path(run_root: Path) -> Path:
    return (run_root / "post_stage" / "checkpoints" / "theta_phase1.zip").resolve()


def _phase1_manifest_path(run_root: Path) -> Path:
    return (run_root / "post_stage" / "phase1_ckpt_manifest.csv").resolve()


def _phase1_data_root(run_root: Path) -> Path:
    return (run_root / "data").resolve()


def _build_pipeline_cmd(
    *,
    python_bin: str,
    run_root: Path,
    passthrough: Sequence[str],
    extra_args: Sequence[str],
) -> List[str]:
    return [
        str(Path(python_bin).resolve()),
        str((CODES_DIR / "outer_rl" / "run_edrl_pipeline.py").resolve()),
        "--run-id",
        str(run_root),
        *list(passthrough),
        *list(extra_args),
    ]


def _build_summary_cmd(
    *,
    python_bin: str,
    suite_root: Path,
) -> List[str]:
    return [
        str(Path(python_bin).resolve()),
        str((CODES_DIR / "analysis" / "summarize_recovery_ab_suite.py").resolve()),
        "--suite-root",
        str(suite_root),
        "--out-dir",
        str((suite_root / "summary").resolve()),
    ]


def main() -> int:
    args = parse_args()
    suite_root = _resolve_suite_root(args.suite_root)
    suite_root.mkdir(parents=True, exist_ok=True)

    clean_passthrough = _strip_controlled_passthrough(args.pipeline_passthrough)
    python_bin = str(Path(args.python_bin).resolve())
    env = dict(os.environ)

    manifest_payload = {
        "suite_root": str(suite_root),
        "python_bin": python_bin,
        "phase1_history_every_tables": int(args.phase1_history_every_tables),
        "pipeline_passthrough": clean_passthrough,
        "runs": {
            "main": str((suite_root / "main").resolve()),
            "actor_rollback": str((suite_root / "actor_rollback").resolve()),
            "rollback": str((suite_root / "rollback").resolve()),
        },
    }
    (suite_root / "recovery_ab_suite_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    main_root = (suite_root / "main").resolve()
    actor_root = (suite_root / "actor_rollback").resolve()
    rollback_root = (suite_root / "rollback").resolve()

    main_cmd = _build_pipeline_cmd(
        python_bin=python_bin,
        run_root=main_root,
        passthrough=clean_passthrough,
        extra_args=[
            "--phase1-history-every-tables",
            str(int(args.phase1_history_every_tables)),
            "--no-actor-rollback-enable",
            "--no-rollback-enable",
        ],
    )

    phase1_ckpt = _phase1_ckpt_path(main_root)
    phase1_manifest = _phase1_manifest_path(main_root)
    phase1_data = _phase1_data_root(main_root)

    actor_cmd = _build_pipeline_cmd(
        python_bin=python_bin,
        run_root=actor_root,
        passthrough=clean_passthrough,
        extra_args=[
            "--phase1-skip",
            "--outer-base-ckpt",
            str(phase1_ckpt),
            "--phase1-external-data-root",
            str(phase1_data),
            "--phase1-history-every-tables",
            str(int(args.phase1_history_every_tables)),
            "--actor-rollback-enable",
            "--no-rollback-enable",
            "--actor-rollback-manifest-path",
            str(phase1_manifest),
        ],
    )

    rollback_cmd = _build_pipeline_cmd(
        python_bin=python_bin,
        run_root=rollback_root,
        passthrough=clean_passthrough,
        extra_args=[
            "--phase1-skip",
            "--outer-base-ckpt",
            str(phase1_ckpt),
            "--phase1-external-data-root",
            str(phase1_data),
            "--phase1-history-every-tables",
            str(int(args.phase1_history_every_tables)),
            "--no-actor-rollback-enable",
            "--rollback-enable",
            "--rollback-manifest-path",
            str(phase1_manifest),
        ],
    )

    summary_cmd = _build_summary_cmd(python_bin=python_bin, suite_root=suite_root)

    if not args.skip_main:
        code = _run_cmd("MAIN", main_cmd, env=env, dry_run=bool(args.dry_run))
        if code != 0:
            return int(code)

    if (not phase1_ckpt.exists() or not phase1_data.exists() or not phase1_manifest.exists()) and not bool(args.dry_run):
        missing = []
        if not phase1_ckpt.exists():
            missing.append(str(phase1_ckpt))
        if not phase1_data.exists():
            missing.append(str(phase1_data))
        if not phase1_manifest.exists():
            missing.append(str(phase1_manifest))
        raise FileNotFoundError(
            "main run phase1 artifacts are required before A/B suite branches can start: "
            + ", ".join(missing)
        )

    if not args.skip_a:
        code = _run_cmd("A", actor_cmd, env=env, dry_run=bool(args.dry_run))
        if code != 0:
            return int(code)

    if not args.skip_b:
        code = _run_cmd("B", rollback_cmd, env=env, dry_run=bool(args.dry_run))
        if code != 0:
            return int(code)

    if not args.skip_summary:
        code = _run_cmd("SUMMARY", summary_cmd, env=env, dry_run=bool(args.dry_run))
        if code != 0:
            return int(code)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
