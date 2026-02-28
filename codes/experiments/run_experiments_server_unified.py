import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
SERVER_OUTPUT_ROOT = CODES_DIR / "nexus"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from experiments.run_experiments_common import ExperimentConfig, run_experiments


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
class VariantSpec:
    raw: str
    algorithm: str
    algo_version: str
    ppo_new_window: Optional[int]


def _norm_algo_version(version: str) -> str:
    value = str(version or "v1").strip().lower()
    aliases = {
        "v31": "v3.1",
        "v3_1": "v3.1",
        "v32": "v3.2",
        "v3_2": "v3.2",
        "v41": "v4.1",
        "v4_1": "v4.1",
        "v42_phase": "v4.2_phase",
        "v4_2_phase": "v4.2_phase",
        "v42_mean": "v4.2_mean",
        "v4_2_mean": "v4.2_mean",
        "v43_ent": "v4.3_ent",
        "v4_3_ent": "v4.3_ent",
        "v43_logit_bias": "v4.3_logit_bias",
        "v4_3_logit_bias": "v4.3_logit_bias",
        "v51_abppo": "v5.1_abppo",
        "v5_1_abppo": "v5.1_abppo",
        "v52_qcritic": "v5.2_qcritic",
        "v5_2_qcritic": "v5.2_qcritic",
        "v53_auxweak": "v5.3_auxweak",
        "v5_3_auxweak": "v5.3_auxweak",
        "v61_cvarppo": "v6.1_cvarppo",
        "v6_1_cvarppo": "v6.1_cvarppo",
        "v62_v3cvar": "v6.2_v3cvar",
        "v6_2_v3cvar": "v6.2_v3cvar",
        "v71_poolppo": "v7.1_poolppo",
        "v7_1_poolppo": "v7.1_poolppo",
        "v72_poolv3": "v7.2_poolv3",
        "v7_2_poolv3": "v7.2_poolv3",
    }
    return aliases.get(value, value)


def _default_window_for_version(version: str) -> int:
    v = _norm_algo_version(version)
    if v == "v3.1":
        return 8
    if v in (
        "v2",
        "v3",
        "v3.2",
        "v4",
        "v4.1",
        "v4.2_phase",
        "v4.2_mean",
        "v4.3_ent",
        "v4.3_logit_bias",
        "v5.1_abppo",
        "v5.2_qcritic",
        "v5.3_auxweak",
        "v6.2_v3cvar",
        "v7.2_poolv3",
    ):
        return 4
    return 1


def _parse_one_variant(spec: str, n_stack_override: Optional[int]) -> VariantSpec:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("empty variant spec")

    if ":" in raw:
        algo_part, version_part = raw.split(":", 1)
    elif "@" in raw:
        algo_part, version_part = raw.split("@", 1)
    else:
        algo_part, version_part = raw, ""

    algorithm = str(algo_part or "").strip().upper()
    if not algorithm:
        raise ValueError(f"invalid variant spec: {raw}")

    if algorithm == "PPO_NEW":
        algo_version = _norm_algo_version(version_part or "v1")
        if algo_version == "v1":
            window_k = 1
        elif n_stack_override is not None:
            window_k = max(1, int(n_stack_override))
        else:
            window_k = _default_window_for_version(algo_version)
        return VariantSpec(
            raw=raw,
            algorithm=algorithm,
            algo_version=algo_version,
            ppo_new_window=int(window_k),
        )

    algo_version = _norm_algo_version(version_part or "v1")
    return VariantSpec(raw=raw, algorithm=algorithm, algo_version=algo_version, ppo_new_window=None)


def parse_variants(args: argparse.Namespace) -> List[VariantSpec]:
    specs = [str(v).strip() for v in (args.variant or []) if str(v).strip()]
    if not specs:
        specs = [f"{args.algo}:{args.algo_version}"]
    out: List[VariantSpec] = []
    seen = set()
    for spec in specs:
        item = _parse_one_variant(spec=spec, n_stack_override=args.n_stack)
        key = (item.algorithm, item.algo_version, item.ppo_new_window)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def resolve_target_run_root(run_folder: str) -> Path:
    raw = str(run_folder or "").strip()
    if not raw:
        raise ValueError("--run-folder is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return (SERVER_OUTPUT_ROOT / candidate).resolve()


def build_config(
    variant: VariantSpec,
    dists: List[str],
    requests: List[int],
    seeds: List[int],
    args: argparse.Namespace,
    run_root: Path,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"server_unified_{variant.algorithm}_{variant.algo_version}",
        distributions=dists,
        request_numbers=requests,
        algorithms=[variant.algorithm],
        seeds=seeds,
        generator_workers=int(args.generator_workers),
        run_baseline=bool(args.run_baseline),
        baseline_include_random=bool(args.baseline_include_random),
        run_plots=bool(args.run_plots),
        run_metrics=bool(args.run_metrics),
        cleanup_after_run=bool(args.cleanup_after_run),
        log_subdir=str(run_root),
        algo_version=str(variant.algo_version),
        ppo_new_window=variant.ppo_new_window,
        resume_existing=bool(args.resume_existing),
        skip_completed=bool(args.skip_completed),
        notify_on_failure=bool(args.notify_failure),
        notify_on_success=bool(args.notify_success),
        stage_mode=str(args.stage_mode or "train_eval"),
        init_model_path=str(args.init_model_path or "").strip() or None,
        save_model_path=str(args.save_model_path or "").strip() or None,
    )


def _run_precheck(
    run_root: Path,
    algorithms: List[str],
    dists: List[str],
    args: argparse.Namespace,
) -> int:
    has_existing_runs = any(run_root.glob("run_*"))
    if not args.precheck:
        return 0
    if not has_existing_runs:
        print(f"[unified] precheck skipped (no existing run_* under {run_root})")
        return 0

    cmd = [
        sys.executable,
        str(CODES_DIR / "tools" / "rerun_incomplete.py"),
        "--logs-root",
        str(run_root),
        "--no-clean",
    ]
    for algo in sorted(set(algorithms)):
        cmd.extend(["--algorithm", str(algo)])
    for dist_name in sorted(set(dists)):
        cmd.extend(["--dist-name", str(dist_name)])
    if int(args.precheck_workers or 0) > 0:
        cmd.extend(["--workers", str(int(args.precheck_workers))])
    if args.dry_run:
        cmd.append("--dry-run")

    return subprocess.run(cmd, cwd=str(CODES_DIR)).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified server runner: one script for PPO / PPO_NEW(v1..v7) with multicore support."
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
        help="variant spec (repeatable), format: ALGO[:version], e.g. PPO or PPO_NEW:v3.1",
    )
    parser.add_argument("--algo", type=str, default="PPO", help="fallback algorithm when --variant not set")
    parser.add_argument("--algo-version", type=str, default="v1", help="fallback algo version when --variant not set")
    parser.add_argument("--n-stack", type=int, default=None, help="override PPO_NEW window for stacked versions")
    parser.add_argument("--dist-name", action="append", default=None, help="distribution name (repeatable)")
    parser.add_argument("--request-number", type=int, action="append", default=None, help="request number R (repeatable)")
    parser.add_argument("--seed", type=int, action="append", default=None, help="seed (repeatable)")
    parser.add_argument("--max-workers", type=int, default=None, help="parallel workers across scenarios")
    parser.add_argument("--generator-workers", type=int, default=1, help="generator workers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-mode", type=str, default="train_eval", help="train_eval/train_only/eval_only")
    parser.add_argument("--init-model-path", type=str, default="", help="optional checkpoint to load")
    parser.add_argument("--save-model-path", type=str, default="", help="optional checkpoint to save")

    parser.add_argument("--run-baseline", action="store_true", default=False, help="run baseline stage")
    parser.add_argument("--no-run-baseline", action="store_false", dest="run_baseline")
    parser.add_argument("--baseline-include-random", action="store_true", default=False)

    parser.add_argument("--run-plots", action="store_true", default=False, help="run plotting stage")
    parser.add_argument("--no-run-plots", action="store_false", dest="run_plots")

    parser.add_argument("--run-metrics", action="store_true", default=True, help="run metrics stage (default: on)")
    parser.add_argument("--no-run-metrics", action="store_false", dest="run_metrics")

    parser.add_argument("--cleanup-after-run", action="store_true", default=False, help="cleanup transient files")
    parser.add_argument("--resume-existing", action="store_true", default=True, help="resume incomplete run (default: on)")
    parser.add_argument("--no-resume-existing", action="store_false", dest="resume_existing")
    parser.add_argument("--skip-completed", action="store_true", default=True, help="skip completed run (default: on)")
    parser.add_argument("--no-skip-completed", action="store_false", dest="skip_completed")

    parser.add_argument("--precheck", action="store_true", default=True, help="run rerun_incomplete before execute")
    parser.add_argument("--no-precheck", action="store_false", dest="precheck")
    parser.add_argument("--precheck-workers", type=int, default=0, help="workers for rerun_incomplete (0=auto)")

    parser.add_argument("--notify-success", action="store_true", default=False)
    parser.add_argument("--no-notify-failure", action="store_false", dest="notify_failure", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dists = [str(x).strip() for x in (args.dist_name or DEFAULT_DISTS) if str(x).strip()]
    requests = [int(x) for x in (args.request_number or [30])]
    seeds = [int(x) for x in (args.seed or [42])]
    variants = parse_variants(args)

    run_root = resolve_target_run_root(args.run_folder)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[unified] run_root={run_root}")
    print(f"[unified] variants={[v.raw for v in variants]}")
    print(f"[unified] distributions={dists}")
    print(f"[unified] requests={requests}")
    print(f"[unified] seeds={seeds}")
    print(f"[unified] max_workers={args.max_workers}")

    precheck_code = _run_precheck(
        run_root=run_root,
        algorithms=[v.algorithm for v in variants],
        dists=dists,
        args=args,
    )
    if precheck_code != 0:
        print(f"[unified] precheck failed (exit={precheck_code})")
        return 1

    failed_variants: List[str] = []
    for variant in variants:
        print(
            f"\n[unified] start variant={variant.raw} "
            f"-> algo={variant.algorithm}, version={variant.algo_version}, window={variant.ppo_new_window}"
        )
        config = build_config(
            variant=variant,
            dists=dists,
            requests=requests,
            seeds=seeds,
            args=args,
            run_root=run_root,
        )
        failed = run_experiments(config=config, max_workers=args.max_workers, dry_run=bool(args.dry_run))
        if failed:
            failed_variants.append(variant.raw)
            print(f"[unified] failed variant={variant.raw}")
        else:
            print(f"[unified] done variant={variant.raw}")

    if failed_variants:
        print(f"[unified] failed variants: {failed_variants}")
        return 1
    print("[unified] all variants completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
