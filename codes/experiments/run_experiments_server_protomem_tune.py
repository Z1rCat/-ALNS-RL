import argparse
import datetime
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
SERVER_OUTPUT_ROOT = CODES_DIR / "nexus"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import ExperimentConfig, run_experiments


TARGET_DISTRIBUTIONS = [
    "G_10_30_60",
    "O_10_60",
    "O_30_60",
    "F2_10_60",
]


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if not (sys.stdout and sys.stdout.isatty()):
        return text
    palette = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    codes: List[str] = []
    if bold:
        codes.append("1")
    if color and color in palette:
        codes.append(palette[color])
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


@dataclass(frozen=True)
class TunePreset:
    name: str
    note: str
    env: Dict[str, str]


PRESETS: List[TunePreset] = [
    TunePreset(
        name="p1_balanced",
        note="Default balanced routing; stage A/B first pass.",
        env={
            "PM_INPUT_MODE": "full",
            "PM_NUM_PROTOTYPES": "32",
            "PM_MEM_DIM": "64",
            "PM_HIDDEN_DIM": "64",
            "PM_TAU": "0.70",
            "PM_LAMBDA_SPARSE": "0.0002",
            "PM_LAMBDA_DIV": "0.0030",
            "PM_LAMBDA_STABLE": "0.0",
            "PM_LAMBDA_AUX": "0.0",
            "PM_MEM_LR_SCALE": "0.5",
            "PM_USE_SMOOTH": "1",
            "PM_SMOOTH_ALPHA": "0.10",
            "PM_SMOOTH_TRAIN_TEST_CONSISTENT": "1",
            "PM_N_STEPS": "10",
            "PM_BATCH_SIZE": "10",
            "PM_N_EPOCHS": "5",
            "PM_LR": "0.0003",
        },
    ),
    TunePreset(
        name="p2_soft_route",
        note="Softer routing to avoid early prototype collapse.",
        env={
            "PM_INPUT_MODE": "full",
            "PM_NUM_PROTOTYPES": "32",
            "PM_MEM_DIM": "64",
            "PM_HIDDEN_DIM": "64",
            "PM_TAU": "0.90",
            "PM_LAMBDA_SPARSE": "0.0001",
            "PM_LAMBDA_DIV": "0.0020",
            "PM_LAMBDA_STABLE": "0.0",
            "PM_LAMBDA_AUX": "0.0",
            "PM_MEM_LR_SCALE": "0.5",
            "PM_USE_SMOOTH": "1",
            "PM_SMOOTH_ALPHA": "0.10",
            "PM_SMOOTH_TRAIN_TEST_CONSISTENT": "1",
            "PM_N_STEPS": "10",
            "PM_BATCH_SIZE": "10",
            "PM_N_EPOCHS": "5",
            "PM_LR": "0.0003",
        },
    ),
    TunePreset(
        name="p3_sharp_route",
        note="Sharper routing + stronger sparsity; may improve phase split.",
        env={
            "PM_INPUT_MODE": "full",
            "PM_NUM_PROTOTYPES": "32",
            "PM_MEM_DIM": "64",
            "PM_HIDDEN_DIM": "64",
            "PM_TAU": "0.60",
            "PM_LAMBDA_SPARSE": "0.0005",
            "PM_LAMBDA_DIV": "0.0030",
            "PM_LAMBDA_STABLE": "0.0",
            "PM_LAMBDA_AUX": "0.0",
            "PM_MEM_LR_SCALE": "0.7",
            "PM_USE_SMOOTH": "1",
            "PM_SMOOTH_ALPHA": "0.10",
            "PM_SMOOTH_TRAIN_TEST_CONSISTENT": "1",
            "PM_N_STEPS": "10",
            "PM_BATCH_SIZE": "10",
            "PM_N_EPOCHS": "5",
            "PM_LR": "0.0003",
        },
    ),
    TunePreset(
        name="p4_with_stable",
        note="Enable stable replay regularization after warmup.",
        env={
            "PM_INPUT_MODE": "full",
            "PM_NUM_PROTOTYPES": "32",
            "PM_MEM_DIM": "64",
            "PM_HIDDEN_DIM": "64",
            "PM_TAU": "0.70",
            "PM_LAMBDA_SPARSE": "0.0002",
            "PM_LAMBDA_DIV": "0.0030",
            "PM_LAMBDA_STABLE": "0.05",
            "PM_LAMBDA_AUX": "0.0",
            "PM_STABLE_BUF_PER_PHASE": "400",
            "PM_STABLE_BATCH_RATIO": "0.30",
            "PM_STABLE_WARMUP_UPDATES": "5",
            "PM_MEM_LR_SCALE": "0.5",
            "PM_USE_SMOOTH": "1",
            "PM_SMOOTH_ALPHA": "0.10",
            "PM_SMOOTH_TRAIN_TEST_CONSISTENT": "1",
            "PM_N_STEPS": "10",
            "PM_BATCH_SIZE": "10",
            "PM_N_EPOCHS": "5",
            "PM_LR": "0.0003",
        },
    ),
]


def resolve_target_run_root(run_folder: str) -> Path:
    raw = str(run_folder or "").strip()
    if not raw:
        raise ValueError("--run-folder is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (SERVER_OUTPUT_ROOT / candidate).resolve()


def build_config(seeds: Iterable[int], generator_workers: int) -> ExperimentConfig:
    return ExperimentConfig(
        name="server_protomem_tune",
        distributions=list(TARGET_DISTRIBUTIONS),
        request_numbers=[30],
        algorithms=["PPO_PROTOMEM"],
        seeds=[int(s) for s in seeds],
        generator_workers=max(1, int(generator_workers)),
        run_baseline=False,
        baseline_include_random=False,
        run_metrics=False,
        run_plots=False,
        cleanup_after_run=False,
        log_subdir="server_runs",
    )


def parse_int_csv(raw: str) -> List[int]:
    values: List[int] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        values.append(int(text))
    if not values:
        raise ValueError("seed list is empty")
    return values


def parse_preset_names(raw: str) -> List[str]:
    values: List[str] = []
    for token in str(raw).split(","):
        text = token.strip()
        if text:
            values.append(text)
    if not values:
        raise ValueError("preset list is empty")
    return values


def select_presets(raw: str) -> List[TunePreset]:
    wanted = parse_preset_names(raw)
    by_name = {preset.name: preset for preset in PRESETS}
    selected: List[TunePreset] = []
    for name in wanted:
        preset = by_name.get(name)
        if preset is None:
            known = ", ".join(by_name.keys())
            raise ValueError(f"unknown preset '{name}', available: {known}")
        selected.append(preset)
    return selected


@contextmanager
def temporary_env(overrides: Dict[str, str]):
    backup: Dict[str, str] = {}
    missing: List[str] = []
    for key, value in overrides.items():
        if key in os.environ:
            backup[key] = os.environ[key]
        else:
            missing.append(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key in missing:
            os.environ.pop(key, None)
        for key, value in backup.items():
            os.environ[key] = value


def write_tuning_plan(run_root: Path, seeds: List[int], presets: List[TunePreset]) -> None:
    plan = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "workdir": str(Path.cwd()),
        "seeds": seeds,
        "distributions": TARGET_DISTRIBUTIONS,
        "algorithm": "PPO_PROTOMEM",
        "request_number": 30,
        "presets": [{"name": p.name, "note": p.note, "env": p.env} for p in presets],
    }
    (run_root / "tuning_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_precheck_if_needed(
    run_root: Path,
    distributions: List[str],
    algorithms: List[str],
    precheck_workers: int,
    dry_run: bool,
) -> int:
    has_existing_runs = any(run_root.glob("run_*"))
    if not has_existing_runs:
        print(
            f"{_c('[protomem_tune]', 'cyan', True)} "
            f"{_c('PRECHECK', 'yellow', True)} skipped (no existing run_* under {run_root})"
        )
        return 0
    cmd = [
        sys.executable,
        str(CODES_DIR / "tools" / "rerun_incomplete.py"),
        "--logs-root",
        str(run_root),
        "--no-clean",
    ]
    for algorithm in algorithms:
        cmd.extend(["--algorithm", algorithm])
    for dist_name in distributions:
        cmd.extend(["--dist-name", dist_name])
    if precheck_workers > 0:
        cmd.extend(["--workers", str(precheck_workers)])
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=str(CODES_DIR)).returncode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="target folder under codes/nexus (or absolute path)",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="parallel workers across distributions")
    parser.add_argument("--generator-workers", type=int, default=1, help="workers passed to Dynamic_master34959.py")
    parser.add_argument("--seeds", type=str, default="42", help="comma-separated seeds, e.g. 42,43")
    parser.add_argument(
        "--presets",
        type=str,
        default="p1_balanced,p2_soft_route,p3_sharp_route,p4_with_stable",
        help="comma-separated preset names",
    )
    parser.add_argument("--list-presets", action="store_true", help="print preset names and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--precheck",
        action="store_true",
        default=True,
        help="run rerun_incomplete before each preset (default: on)",
    )
    parser.add_argument(
        "--no-precheck",
        action="store_false",
        dest="precheck",
        help="disable rerun_incomplete precheck",
    )
    parser.add_argument("--precheck-workers", type=int, default=0, help="workers for rerun_incomplete (0=auto)")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        default=True,
        help="resume latest incomplete run for same (dist,R,algo,seed) (default: on)",
    )
    parser.add_argument(
        "--no-resume-existing",
        action="store_false",
        dest="resume_existing",
        help="always create a new run_* folder",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="skip task if an existing completed run is found (default: on)",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_false",
        dest="skip_completed",
        help="do not skip tasks even if completed runs exist",
    )
    parser.add_argument(
        "--notify-success",
        action="store_true",
        default=False,
        help="send summary notification on success when notification channels are configured",
    )
    parser.add_argument(
        "--no-notify-failure",
        action="store_false",
        dest="notify_failure",
        default=True,
        help="disable failure notifications",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_presets:
        for preset in PRESETS:
            print(f"{preset.name}: {preset.note}")
        return 0

    seeds = parse_int_csv(args.seeds)
    selected_presets = select_presets(args.presets)

    run_root = resolve_target_run_root(args.run_folder)
    run_root.mkdir(parents=True, exist_ok=True)
    write_tuning_plan(run_root, seeds=seeds, presets=selected_presets)

    total_failed = 0
    preset_results: Dict[str, int] = {}

    for index, preset in enumerate(selected_presets, start=1):
        preset_root = run_root / f"{index:02d}_{preset.name}"
        preset_root.mkdir(parents=True, exist_ok=True)

        print(
            f"{_c('[protomem_tune]', 'cyan', True)} "
            f"{_c(f'PRESET {index}/{len(selected_presets)}', 'magenta', True)} "
            f"{preset.name} -> {preset.note}"
        )
        print(f"{_c('[protomem_tune]', 'cyan', True)} {_c('OUTPUT', 'blue', True)} {preset_root}")

        config = build_config(seeds=seeds, generator_workers=args.generator_workers)
        config.name = f"protomem_tune:{preset.name}"
        config.log_subdir = str(preset_root)
        config.resume_existing = bool(args.resume_existing)
        config.skip_completed = bool(args.skip_completed)
        config.notify_on_success = bool(args.notify_success)
        config.notify_on_failure = bool(args.notify_failure)

        if args.precheck:
            precheck_code = run_precheck_if_needed(
                run_root=preset_root,
                distributions=config.distributions,
                algorithms=config.algorithms,
                precheck_workers=args.precheck_workers,
                dry_run=args.dry_run,
            )
            if precheck_code != 0:
                print(
                    f"{_c('[protomem_tune]', 'cyan', True)} "
                    f"{_c('PRECHECK', 'red', True)} failed for {preset.name} (exit={precheck_code})"
                )
                preset_results[preset.name] = 1
                total_failed += 1
                continue

        with temporary_env(preset.env):
            failed = run_experiments(config, args.max_workers, args.dry_run)

        preset_results[preset.name] = int(failed)
        total_failed += int(failed)

    summary = {
        "run_root": str(run_root),
        "total_failed": total_failed,
        "preset_results": preset_results,
    }
    print(f"{_c('[protomem_tune]', 'cyan', True)} {_c('SUMMARY', 'green' if total_failed == 0 else 'red', True)} {json.dumps(summary, ensure_ascii=False)}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
