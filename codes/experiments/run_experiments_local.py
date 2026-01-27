import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import ExperimentConfig, run_experiments


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="local",
        distributions=["S5_1", "S3_1", "V1_3"],
        request_numbers=[30],
        algorithms=["DQN", "A2C"],
        seeds=[42],
        generator_workers=1,
        max_workers=2,
        exclude_tasks=[("S5_1", 30, "A2C", 42)],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config()
    failed = run_experiments(config, args.max_workers, args.dry_run)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
