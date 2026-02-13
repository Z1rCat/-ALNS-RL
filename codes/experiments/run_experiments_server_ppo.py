import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import ExperimentConfig, resolve_run_root, run_experiments


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="server_ppo",
        distributions=[
            "S1_1",
            "S2_1",
            "S3_1",
            "S4_1",
            "S5_1",
            "S6_1",
        ],
        request_numbers=[30],
        algorithms=["PPO"],
        seeds=[42],
        generator_workers=1,
        baseline_include_random=True,
        run_metrics=True,
        cleanup_after_run=True,
        log_subdir="server_runs",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=None, help="parallel workers across scenarios")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--precheck",
        action="store_true",
        default=True,
        help="run rerun_incomplete before experiments (default: on)",
    )
    parser.add_argument(
        "--no-precheck",
        action="store_false",
        dest="precheck",
        help="disable rerun_incomplete precheck",
    )
    parser.add_argument("--precheck-workers", type=int, default=0, help="workers for rerun_incomplete (0=auto)")
    parser.add_argument("--precheck-logs-root", type=str, default="", help="logs root for rerun_incomplete")
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
    config = build_config()
    config.resume_existing = bool(args.resume_existing)
    config.skip_completed = bool(args.skip_completed)
    config.notify_on_success = bool(args.notify_success)
    config.notify_on_failure = bool(args.notify_failure)
    if args.precheck:
        logs_root = args.precheck_logs_root or str(resolve_run_root(config))
        cmd = [
            sys.executable,
            str(CODES_DIR / "tools" / "rerun_incomplete.py"),
            "--logs-root",
            logs_root,
        ]
        if args.precheck_workers:
            cmd.extend(["--workers", str(args.precheck_workers)])
        if args.dry_run:
            cmd.append("--dry-run")
        code = subprocess.run(cmd, cwd=str(CODES_DIR)).returncode
        if code != 0:
            print(f"[server_ppo] precheck failed (exit={code})")
            return 1
    failed = run_experiments(config, args.max_workers, args.dry_run)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
