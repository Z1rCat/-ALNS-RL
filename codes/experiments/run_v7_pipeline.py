import argparse
import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
ROOT_DIR = CODES_DIR.parent
DEFAULT_OUTPUT_ROOT = CODES_DIR / "nexus"
DIST_CONFIG_PATH = ROOT_DIR / "distribution_config.json"


@dataclass(frozen=True)
class VariantSpec:
    raw: str
    algorithm: str
    algo_version: str


V7_GROUPS: Dict[str, Dict[str, object]] = {
    "S1": {"pool_means": [10, 30, 60, 90, 120], "test_mean": 60, "type": "Seen-mean"},
    "S2": {"pool_means": [10, 30, 60, 90, 120], "test_mean": 90, "type": "Seen-mean"},
    "U1": {"pool_means": [10, 30, 90, 120], "test_mean": 60, "type": "Unseen interpolation"},
    "U2": {"pool_means": [10, 30, 60, 120], "test_mean": 90, "type": "Unseen interpolation"},
    "U3": {"pool_means": [10, 60, 90, 120], "test_mean": 30, "type": "Unseen interpolation"},
    "E1": {"pool_means": [10, 30, 60, 90], "test_mean": 120, "type": "Unseen extrapolation"},
}


def _norm_algo_version(version: str) -> str:
    value = str(version or "v1").strip().lower()
    aliases = {
        "v31": "v3.1",
        "v3_1": "v3.1",
        "v32": "v3.2",
        "v3_2": "v3.2",
        "v61_cvarppo": "v6.1_cvarppo",
        "v6_1_cvarppo": "v6.1_cvarppo",
        "v62_v3cvar": "v6.2_v3cvar",
        "v6_2_v3cvar": "v6.2_v3cvar",
        "v71_poolppo": "v7.1_poolppo",
        "v7_1_poolppo": "v7.1_poolppo",
        "v72_poolv3": "v7.2_poolv3",
        "v7_2_poolv3": "v7.2_poolv3",
        "v73_tcrppo": "v7.3_tcrppo",
        "v7_3_tcrppo": "v7.3_tcrppo",
        "v74_tcrv3": "v7.4_tcrv3",
        "v7_4_tcrv3": "v7.4_tcrv3",
    }
    return aliases.get(value, value or "v1")


def _parse_variant(spec: str) -> VariantSpec:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("empty --variant")
    if ":" in raw:
        algo, ver = raw.split(":", 1)
    elif "@" in raw:
        algo, ver = raw.split("@", 1)
    else:
        algo, ver = raw, "v1"
    algorithm = str(algo or "").strip().upper()
    if not algorithm:
        raise ValueError(f"invalid variant: {raw}")
    return VariantSpec(raw=raw, algorithm=algorithm, algo_version=_norm_algo_version(ver))


def _algorithm_tag(algorithm: str, algo_version: str) -> str:
    algo = str(algorithm or "").strip().upper()
    if algo == "PPO_NEW":
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(algo_version or "v1"))
        cleaned = cleaned.strip("_").upper() or "V1"
        return f"PPONEW{cleaned}"
    return algo


def _mean_to_dist(mean_val: int) -> str:
    return f"M_{int(mean_val)}"


def _pool_signature(pool_means: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in pool_means}))


def _pool_tag(pool_sig: Sequence[int]) -> str:
    return "POOL_" + "_".join(str(int(x)) for x in pool_sig)


def _build_run_name(
    stage: str,
    tag: str,
    request_number: int,
    dist_name: str,
    algorithm: str,
    algo_version: str,
    seed: int,
) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    algo_tag = _algorithm_tag(algorithm, algo_version)
    run_tag = str(tag).upper().replace(" ", "_")
    return f"run_{timestamp}_{run_tag}_{stage}_R{request_number}_{dist_name}_{algo_tag}_S{seed}"


def _load_distribution_map() -> Dict[str, Dict[str, object]]:
    if not DIST_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(DIST_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items: Dict[str, Dict[str, object]] = {}
    for item in data.get("distributions", []):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                items[name] = item
    return items


def _validate_group_dist_keys(groups: Sequence[str]) -> None:
    dist_map = _load_distribution_map()
    existing = set(dist_map.keys())
    missing: List[str] = []
    bad_pattern: List[str] = []
    for gid in groups:
        spec = V7_GROUPS[gid]
        pool = [int(x) for x in spec["pool_means"]]  # type: ignore[index]
        test_mean = int(spec["test_mean"])  # type: ignore[index]
        for mean_val in sorted(set(pool + [test_mean])):
            dist_name = _mean_to_dist(mean_val)
            if dist_name not in existing:
                missing.append(dist_name)
                continue
            pattern = str(dist_map[dist_name].get("pattern", "")).strip().lower()
            if pattern == "single_mean":
                bad_pattern.append(dist_name)
    if missing:
        uniq = sorted(set(missing))
        raise ValueError(
            "missing mean distributions in distribution_config.json: "
            + ", ".join(uniq)
            + ". Please add them using current uncertainty-event pattern semantics."
        )
    if bad_pattern:
        uniq = sorted(set(bad_pattern))
        raise ValueError(
            "V7 mean distributions must not use pattern=single_mean: "
            + ", ".join(uniq)
            + ". Use existing mechanism pattern (recommended: pattern=ab with A=B=mean)."
        )


def _resolve_eval_dist_names(
    eval_dist_names: Sequence[str],
    eval_patterns: Sequence[str],
) -> List[str]:
    dist_map = _load_distribution_map()
    if not dist_map:
        return []

    selected: List[str] = []
    seen = set()

    for raw in (eval_dist_names or []):
        name = str(raw or "").strip()
        if not name:
            continue
        if name not in dist_map:
            raise ValueError(f"unknown --eval-dist-name '{name}'")
        if name not in seen:
            seen.add(name)
            selected.append(name)

    pattern_set = {str(p or "").strip().lower() for p in (eval_patterns or []) if str(p or "").strip()}
    if pattern_set:
        for name, item in dist_map.items():
            pattern = str(item.get("pattern", "")).strip().lower()
            if pattern in pattern_set and name not in seen:
                seen.add(name)
                selected.append(name)

    return selected


def _run_one_master(
    run_root: Path,
    run_tag: str,
    dist_name: str,
    request_number: int,
    variant: VariantSpec,
    seed: int,
    stage_mode: str,
    init_model_path: str,
    save_model_path: str,
    workers: int,
    run_metrics: bool,
    dry_run: bool,
    train_only_early_stop: bool,
    train_only_min_table: int,
) -> Path:
    run_name = _build_run_name(
        stage="pretrain" if stage_mode == "train_only" else "eval",
        tag=run_tag,
        request_number=request_number,
        dist_name=dist_name,
        algorithm=variant.algorithm,
        algo_version=variant.algo_version,
        seed=seed,
    )
    run_dir = run_root / run_name
    cmd = [
        sys.executable,
        str(CODES_DIR / "Dynamic_master34959.py"),
        "--dist_name",
        str(dist_name),
        "--request_number",
        str(request_number),
        "--algorithm",
        str(variant.algorithm),
        "--algo_version",
        str(variant.algo_version),
        "--workers",
        str(max(1, int(workers))),
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--stage-mode",
        stage_mode,
    ]
    if init_model_path:
        cmd.extend(["--init-model-path", init_model_path])
    if save_model_path:
        cmd.extend(["--save-model-path", save_model_path])

    env = dict(os.environ)
    env["RL_LOG_ROOT"] = str(run_root)
    if str(stage_mode).strip().lower() == "train_only":
        env["RL_TRAIN_ONLY_EARLY_STOP"] = "1" if bool(train_only_early_stop) else "0"
        env["RL_TRAIN_ONLY_MIN_TABLE"] = str(max(0, int(train_only_min_table)))

    print(f"[v7] start tag={run_tag} stage={stage_mode} variant={variant.raw} seed={seed} dist={dist_name}")
    if dry_run:
        print("[v7] dry-run cmd:", " ".join(cmd))
        return run_dir
    subprocess.run(cmd, cwd=str(ROOT_DIR), env=env, check=True)

    if run_metrics:
        metrics_cmd = [
            sys.executable,
            str(CODES_DIR / "analysis" / "compute_metrics.py"),
            "--run-dir",
            str(run_dir),
        ]
        subprocess.run(metrics_cmd, cwd=str(ROOT_DIR), check=False)
    return run_dir


def _run_variant_seed_pool_pretrain(
    run_root: Path,
    checkpoint_dir: Path,
    pool_sig: Sequence[int],
    request_number: int,
    variant: VariantSpec,
    seed: int,
    generator_workers: int,
    run_metrics: bool,
    reset_checkpoint: bool,
    dry_run: bool,
    train_only_early_stop: bool,
    train_only_min_table: int,
) -> Path:
    pool_tag = _pool_tag(pool_sig)
    pool_dists = [_mean_to_dist(x) for x in pool_sig]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{_algorithm_tag(variant.algorithm, variant.algo_version)}_{pool_tag}_S{seed}.zip"
    checkpoint_path = checkpoint_dir / checkpoint_name
    if checkpoint_path.exists() and not reset_checkpoint:
        print(f"[v7] reuse checkpoint variant={variant.raw} seed={seed} pool={list(pool_sig)} -> {checkpoint_path}")
        return checkpoint_path
    if reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    init_path = str(checkpoint_path) if checkpoint_path.exists() else ""
    for dist_name in pool_dists:
        _run_one_master(
            run_root=run_root,
            run_tag=pool_tag,
            dist_name=dist_name,
            request_number=request_number,
            variant=variant,
            seed=seed,
            stage_mode="train_only",
            init_model_path=init_path,
            save_model_path=str(checkpoint_path),
            workers=generator_workers,
            run_metrics=run_metrics,
            dry_run=dry_run,
            train_only_early_stop=train_only_early_stop,
            train_only_min_table=train_only_min_table,
        )
        init_path = str(checkpoint_path)

    if dry_run:
        return checkpoint_path
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint missing after pretrain: {checkpoint_path}")
    return checkpoint_path


def _run_variant_seed_group_eval(
    run_root: Path,
    group: str,
    eval_dist_name: str,
    request_number: int,
    variant: VariantSpec,
    seed: int,
    generator_workers: int,
    run_metrics: bool,
    checkpoint_path: Path,
    dry_run: bool,
    train_only_early_stop: bool,
    train_only_min_table: int,
) -> None:
    if not dry_run and not checkpoint_path.exists():
        raise RuntimeError(f"missing checkpoint for eval: {checkpoint_path}")
    _run_one_master(
        run_root=run_root,
        run_tag=group,
        dist_name=eval_dist_name,
        request_number=request_number,
        variant=variant,
        seed=seed,
        stage_mode="eval_only",
        init_model_path=str(checkpoint_path),
        save_model_path="",
        workers=generator_workers,
        run_metrics=run_metrics,
        dry_run=dry_run,
        train_only_early_stop=train_only_early_stop,
        train_only_min_table=train_only_min_table,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V7 group protocol by mean pools (S1/S2/U1/U2/U3/E1), with optional multi-distribution evaluation."
    )
    parser.add_argument("--run-folder", type=str, default="v7_group_pipeline_run")
    parser.add_argument("--variant", action="append", default=None, help="e.g. PPO_NEW:v7.1_poolppo (repeatable)")
    parser.add_argument("--group", action="append", default=None, help="group id: S1/S2/U1/U2/U3/E1 (repeatable)")
    parser.add_argument("--request-number", type=int, default=30)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--eval-dist-name", action="append", default=None, help="optional eval distribution names (repeatable)")
    parser.add_argument("--eval-pattern", action="append", default=None, help="optional eval distribution patterns (repeatable), e.g. random_mix/ab/aba/abba/abc")
    parser.add_argument("--max-workers", type=int, default=2, help="parallel pipelines across (variant,seed,group)")
    parser.add_argument("--generator-workers", type=int, default=1, help="data generator workers inside one run")
    parser.add_argument("--run-metrics", action="store_true", default=True)
    parser.add_argument("--no-run-metrics", action="store_false", dest="run_metrics")
    parser.add_argument("--reset-checkpoint", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--train-only-early-stop", action="store_true", default=True, help="allow early-stop in train_only after min-table")
    parser.add_argument("--no-train-only-early-stop", action="store_false", dest="train_only_early_stop")
    parser.add_argument("--train-only-min-table", type=int, default=200, help="minimum table id before train_only early-stop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_folder)
    if not run_root.is_absolute():
        run_root = (DEFAULT_OUTPUT_ROOT / run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    variants_raw = args.variant or ["PPO_NEW:v7.1_poolppo", "PPO_NEW:v7.2_poolv3"]
    variants = [_parse_variant(v) for v in variants_raw]
    groups = [str(x).strip().upper() for x in (args.group or list(V7_GROUPS.keys())) if str(x).strip()]
    unknown = [g for g in groups if g not in V7_GROUPS]
    if unknown:
        raise ValueError(f"unknown group ids: {unknown}; allowed={sorted(V7_GROUPS.keys())}")
    _validate_group_dist_keys(groups)

    seeds = [int(x) for x in (args.seed or [42])]
    request_number = int(args.request_number)
    max_workers = max(1, int(args.max_workers))
    generator_workers = max(1, int(args.generator_workers))
    train_only_min_table = max(0, int(args.train_only_min_table))

    print(f"[v7] run_root={run_root}")
    print(f"[v7] variants={[v.raw for v in variants]}")
    print(f"[v7] groups={groups}")
    for gid in groups:
        spec = V7_GROUPS[gid]
        print(
            f"[v7]   {gid}: pool={spec['pool_means']} -> test={spec['test_mean']} "
            f"({spec['type']})"
        )
    print(f"[v7] request_number={request_number} seeds={seeds} max_workers={max_workers}")
    print(f"[v7] train_only_early_stop={bool(args.train_only_early_stop)} train_only_min_table={train_only_min_table}")

    eval_override = _resolve_eval_dist_names(args.eval_dist_name or [], args.eval_pattern or [])
    if eval_override:
        print(f"[v7] eval_override_dists={eval_override}")
    else:
        print("[v7] eval_override_dists=<none>; using group default test mean")

    group_pool_map: Dict[str, Tuple[int, ...]] = {}
    for gid in groups:
        spec = V7_GROUPS[gid]
        group_pool_map[gid] = _pool_signature([int(x) for x in spec["pool_means"]])  # type: ignore[index]

    pool_jobs = []
    seen_pool_jobs = set()
    for variant in variants:
        for seed in seeds:
            for gid in groups:
                sig = group_pool_map[gid]
                key = (variant.raw, int(seed), sig)
                if key in seen_pool_jobs:
                    continue
                seen_pool_jobs.add(key)
                pool_jobs.append((variant, int(seed), sig))

    eval_jobs = []
    for variant in variants:
        for seed in seeds:
            for gid in groups:
                if eval_override:
                    eval_dist_names = eval_override
                else:
                    spec = V7_GROUPS[gid]
                    eval_dist_names = [_mean_to_dist(int(spec["test_mean"]))]  # type: ignore[index]
                for eval_dist in eval_dist_names:
                    eval_jobs.append((variant, int(seed), gid, group_pool_map[gid], eval_dist))
    checkpoint_dir = run_root / "_checkpoints"

    failures: List[str] = []
    pool_checkpoint_map: Dict[Tuple[str, int, Tuple[int, ...]], Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for variant, seed, pool_sig in pool_jobs:
            future = executor.submit(
                _run_variant_seed_pool_pretrain,
                run_root,
                checkpoint_dir,
                pool_sig,
                request_number,
                variant,
                seed,
                generator_workers,
                bool(args.run_metrics),
                bool(args.reset_checkpoint),
                bool(args.dry_run),
                bool(args.train_only_early_stop),
                train_only_min_table,
            )
            future_map[future] = (variant, seed, pool_sig)
        for future in concurrent.futures.as_completed(future_map):
            variant, seed, pool_sig = future_map[future]
            key_text = f"{variant.raw}|S{seed}|pool={list(pool_sig)}"
            try:
                ckpt = future.result()
                pool_checkpoint_map[(variant.raw, seed, pool_sig)] = ckpt
                print(f"[v7] done pretrain {key_text}")
            except Exception as exc:
                failures.append(f"{key_text}: {exc}")
                print(f"[v7] failed pretrain {key_text}: {exc}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for variant, seed, gid, pool_sig, eval_dist in eval_jobs:
            ckpt = pool_checkpoint_map.get((variant.raw, seed, pool_sig), checkpoint_dir / f"missing_{variant.raw}_{seed}.zip")
            future = executor.submit(
                _run_variant_seed_group_eval,
                run_root,
                gid,
                eval_dist,
                request_number,
                variant,
                seed,
                generator_workers,
                bool(args.run_metrics),
                ckpt,
                bool(args.dry_run),
                bool(args.train_only_early_stop),
                train_only_min_table,
            )
            future_map[future] = (variant, seed, gid, eval_dist)
        for future in concurrent.futures.as_completed(future_map):
            variant, seed, gid, eval_dist = future_map[future]
            key_text = f"{variant.raw}|S{seed}|{gid}|{eval_dist}"
            try:
                future.result()
                print(f"[v7] done eval {key_text}")
            except Exception as exc:
                failures.append(f"{key_text}: {exc}")
                print(f"[v7] failed eval {key_text}: {exc}")

    if failures:
        print("[v7] failures:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("[v7] all pipelines completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
