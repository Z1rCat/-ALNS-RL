import argparse
import concurrent.futures
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
SERVER_OUTPUT_ROOT = CODES_DIR / "nexus"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))


DEFAULT_DISTS = [
    "O_10_30",
    "O_10_60",
    "O_30_60",
    "F1_10_60",
    "F2_10_60",
    "F1_10_30",
    "F2_10_30",
    "G_10_30_60",
    "G_10_60_30",
]


@dataclass(frozen=True)
class TaskSpec:
    variant: str
    dist: str
    request_number: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.variant}|{self.dist}|R{self.request_number}|S{self.seed}"


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def resolve_target_run_root(run_folder: str) -> Path:
    raw = str(run_folder or "").strip()
    if not raw:
        raise ValueError("--run-folder is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return (SERVER_OUTPUT_ROOT / candidate).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream scheduler over (variant, dist, R, seed): as soon as one task completes, "
            "the next task starts, keeping max-workers utilized."
        )
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="target folder under codes/nexus (or absolute path)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="variant spec (repeatable), e.g. PPO or PPO_NEW:v3.1",
    )
    parser.add_argument("--algo", type=str, default="PPO", help="fallback algorithm when --variant not set")
    parser.add_argument("--algo-version", type=str, default="v1", help="fallback version when --variant not set")
    parser.add_argument("--n-stack", type=int, default=None, help="optional global n_stack override for PPO_NEW")
    parser.add_argument("--dist-name", action="append", default=None, help="distribution name (repeatable)")
    parser.add_argument("--request-number", type=int, action="append", default=None, help="request number R (repeatable)")
    parser.add_argument("--seed", type=int, action="append", default=None, help="seed (repeatable)")
    parser.add_argument("--max-workers", type=int, default=8, help="max concurrent tasks in the stream queue")
    parser.add_argument("--generator-workers", type=int, default=1, help="generator workers for each task")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--run-baseline", action="store_true", default=False, help="run baseline stage")
    parser.add_argument("--no-run-baseline", action="store_false", dest="run_baseline")
    parser.add_argument("--run-plots", action="store_true", default=False, help="run plotting stage")
    parser.add_argument("--no-run-plots", action="store_false", dest="run_plots")
    parser.add_argument("--run-metrics", action="store_true", default=True, help="run metrics stage")
    parser.add_argument("--no-run-metrics", action="store_false", dest="run_metrics")

    parser.add_argument("--cleanup-after-run", action="store_true", default=False, help="cleanup transient files")
    parser.add_argument("--resume-existing", action="store_true", default=True, help="resume incomplete run")
    parser.add_argument("--no-resume-existing", action="store_false", dest="resume_existing")
    parser.add_argument("--skip-completed", action="store_true", default=True, help="skip completed runs")
    parser.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")

    parser.add_argument("--precheck", action="store_true", default=False, help="run precheck per task")
    parser.add_argument("--no-precheck", action="store_false", dest="precheck")
    parser.add_argument("--precheck-workers", type=int, default=0, help="workers for precheck")

    parser.add_argument("--notify-success", action="store_true", default=False)
    parser.add_argument("--no-notify-failure", action="store_false", dest="notify_failure", default=True)
    return parser.parse_args()


def parse_variants(args: argparse.Namespace) -> List[str]:
    raw = [str(v).strip() for v in (args.variant or []) if str(v).strip()]
    if not raw:
        raw = [f"{str(args.algo).strip()}:{str(args.algo_version).strip()}"]
    return _dedupe_keep_order(raw)


def build_tasks(args: argparse.Namespace) -> List[TaskSpec]:
    variants = parse_variants(args)
    dists = _dedupe_keep_order([str(x).strip() for x in (args.dist_name or DEFAULT_DISTS) if str(x).strip()])
    requests = [int(x) for x in (args.request_number or [30])]
    seeds = [int(x) for x in (args.seed or [42])]

    tasks: List[TaskSpec] = []
    for variant in variants:
        for dist in dists:
            for r in requests:
                for seed in seeds:
                    tasks.append(TaskSpec(variant=variant, dist=dist, request_number=int(r), seed=int(seed)))
    return tasks


def build_cmd(args: argparse.Namespace, run_root: Path, task: TaskSpec) -> List[str]:
    cmd = [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_experiments_server_unified.py"),
        "--run-folder",
        str(run_root),
        "--variant",
        task.variant,
        "--dist-name",
        task.dist,
        "--request-number",
        str(task.request_number),
        "--seed",
        str(task.seed),
        "--max-workers",
        "1",
        "--generator-workers",
        str(int(args.generator_workers)),
    ]

    if args.n_stack is not None:
        cmd.extend(["--n-stack", str(int(args.n_stack))])

    if args.run_baseline:
        cmd.append("--run-baseline")
    else:
        cmd.append("--no-run-baseline")

    if args.run_plots:
        cmd.append("--run-plots")
    else:
        cmd.append("--no-run-plots")

    if args.run_metrics:
        cmd.append("--run-metrics")
    else:
        cmd.append("--no-run-metrics")

    if args.cleanup_after_run:
        cmd.append("--cleanup-after-run")

    if args.resume_existing:
        cmd.append("--resume-existing")
    else:
        cmd.append("--no-resume-existing")

    if args.skip_completed:
        cmd.append("--skip-completed")
    else:
        cmd.append("--no-skip-completed")

    if args.precheck:
        cmd.append("--precheck")
    else:
        cmd.append("--no-precheck")
    cmd.extend(["--precheck-workers", str(int(args.precheck_workers))])

    if args.notify_success:
        cmd.append("--notify-success")
    if not bool(args.notify_failure):
        cmd.append("--no-notify-failure")

    if args.dry_run:
        cmd.append("--dry-run")

    return cmd


def _run_one(task: TaskSpec, cmd: List[str]) -> Tuple[TaskSpec, int, float]:
    started = time.monotonic()
    code = subprocess.run(cmd, cwd=str(CODES_DIR)).returncode
    elapsed_s = time.monotonic() - started
    return task, int(code), float(elapsed_s)


def main() -> int:
    args = parse_args()
    run_root = resolve_target_run_root(args.run_folder)
    run_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    if not tasks:
        print("[stream] no tasks")
        return 0

    max_workers = max(1, int(args.max_workers))
    print(f"[stream] run_root={run_root}")
    print(f"[stream] tasks={len(tasks)} max_workers={max_workers}")

    for idx, task in enumerate(tasks, start=1):
        print(f"[stream] queued {idx:03d}/{len(tasks)} {task.key}")

    failures: Dict[str, int] = {}
    completed = 0
    started_at = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task: Dict[concurrent.futures.Future, TaskSpec] = {}
        for task in tasks:
            cmd = build_cmd(args=args, run_root=run_root, task=task)
            fut = executor.submit(_run_one, task, cmd)
            future_to_task[fut] = task

        for fut in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[fut]
            completed += 1
            try:
                _task, code, elapsed_s = fut.result()
            except Exception as exc:
                failures[task.key] = -999
                print(f"[stream] FAIL {completed:03d}/{len(tasks)} {task.key} exception={type(exc).__name__}: {exc}")
                continue

            if code != 0:
                failures[task.key] = code
                print(f"[stream] FAIL {completed:03d}/{len(tasks)} {task.key} exit={code} elapsed={elapsed_s:.1f}s")
            else:
                print(f"[stream] DONE {completed:03d}/{len(tasks)} {task.key} elapsed={elapsed_s:.1f}s")

    total_s = time.monotonic() - started_at
    if failures:
        print(f"[stream] completed with failures={len(failures)} total={len(tasks)} elapsed={total_s/60.0:.1f}m")
        for key, code in sorted(failures.items()):
            print(f"[stream] failed_task {key} exit={code}")
        return 1

    print(f"[stream] all tasks done total={len(tasks)} elapsed={total_s/60.0:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
