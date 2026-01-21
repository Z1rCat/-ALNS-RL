import argparse
import sys

from run_experiments_common import ExperimentConfig, run_experiments


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="server",
        distributions=["S1_1", "S5_1", "S2_1", "S3_1", "S6_1", "V1_3"],
        request_numbers=[5, 30],
        algorithms=["DQN", "PPO", "A2C"],
        seeds=[42, 123, 2024],
        generator_workers=1,
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
