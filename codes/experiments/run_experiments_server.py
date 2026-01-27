import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import ExperimentConfig, run_experiments


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="server",
        distributions=[
            
            
            "S3_1",
            "S5_1",
            "S6_1",
            "V1_1",
            
            
            "T1_3",
            
            "W1_1",
            
            
        ],
        request_numbers=[30],
        algorithms=["PPO", "A2C", "PPO_HAT", "DQN"],
        seeds=[42],
        generator_workers=1,
        baseline_include_random=True,
        run_metrics=True,
        cleanup_after_run=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--precheck", action="store_true", help="run rerun_incomplete before experiments")
    parser.add_argument("--precheck-workers", type=int, default=0, help="workers for rerun_incomplete (0=auto)")
    parser.add_argument("--precheck-logs-root", type=str, default="", help="logs root for rerun_incomplete")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config()
    if args.precheck:
        logs_root = args.precheck_logs_root or str(CODES_DIR / "logs")
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
            print(f"[server] precheck failed (exit={code})")
            return 1
    failed = run_experiments(config, args.max_workers, args.dry_run)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
