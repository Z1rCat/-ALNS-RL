import argparse
import sys

from run_experiments_common import ExperimentConfig, run_experiments


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="server",
        distributions=[
            "S1_1",
            "S2_1",
            "S3_1",
            "S5_1",
            "S6_1",
            "V1_1",
            "V1_3",
            "M1_3",
            "T1_3",
            "C1_1",
            "W1_1",
            "G1_1",
            "G1_2",
            "S6_2",
            "S6_3",
            "S2_5",
            "S1_2",
            "S3_5",
            "S5_5",
            "S1_3",
            "S3_2",
            "S3_4",
            "S3_6",
            "S5_2",
            "S5_4",
            "S5_6",
            "V1_2",
            "M1_2",
            "S2_3",
            "S3_3",
            "S5_3",
        ],
        request_numbers=[5, 20, 30],
        algorithms=["DQN", "PPO", "A2C"],
        seeds=[42, 123, 2024],
        generator_workers=1,
        baseline_include_random=True,
        run_metrics=True,
        cleanup_after_run=True,
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
