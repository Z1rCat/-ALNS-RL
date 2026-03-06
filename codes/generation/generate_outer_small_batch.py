from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from tqdm import tqdm

    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    from . import generate_mixed_parallel as base_gen
    from .outer_batch_schema import (
        OuterBatchAction,
        action_to_dict,
        build_phase_labels,
        phase_counts,
        validate_action,
    )
except Exception:
    import generate_mixed_parallel as base_gen
    from outer_batch_schema import (
        OuterBatchAction,
        action_to_dict,
        build_phase_labels,
        phase_counts,
        validate_action,
    )


def _c(text: str, color: str = "", bold: bool = False) -> str:
    try:
        return base_gen._c(text, color=color, bold=bold)
    except Exception:
        return text


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _update_global_manifest(global_manifest_path: Path, iter_manifest_path: Path, payload: Dict[str, object]) -> None:
    data: Dict[str, object] = {}
    if global_manifest_path.exists():
        try:
            data = json.loads(global_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    iterations = data.get("iterations")
    if not isinstance(iterations, dict):
        iterations = {}
    iter_key = str(payload.get("iter_id", ""))
    iterations[iter_key] = {
        "iter_manifest": str(iter_manifest_path.resolve()),
        "updated_at": datetime.now().isoformat(),
        "request_number": payload.get("request_number", ""),
        "num_files": payload.get("num_files", ""),
        "pattern": payload.get("pattern", ""),
    }
    data["schema_version"] = 1
    data["updated_at"] = datetime.now().isoformat()
    data["iterations"] = iterations
    global_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    global_manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_duration_matrix(action: OuterBatchAction, labels: List[str], max_events: int) -> np.ndarray:
    matrix = np.zeros((len(labels), max_events), dtype=int)
    for idx, label in enumerate(labels):
        if label == "A":
            mean_val = action.mu_a
            std_val = action.std_a
        else:
            mean_val = action.mu_b
            std_val = action.std_b
        matrix[idx] = base_gen.sample_durations(mean_val=float(mean_val), max_events=max_events, std=std_val, dist=action.dist)
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--iter-id", type=int, required=True)
    parser.add_argument("--request-number", type=int, required=True, choices=sorted(base_gen.EXP_NUMBERS.keys()))
    parser.add_argument("--mu-a", type=float, required=True)
    parser.add_argument("--mu-b", type=float, required=True)
    parser.add_argument("--ratio-a", type=float, required=True)
    parser.add_argument("--num-files", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default="ab", choices=["ab", "aba", "abba", "random_mix"])
    parser.add_argument("--dist", type=str, default="normal", choices=["normal", "lognormal"])
    parser.add_argument("--std-a", type=float, default=None)
    parser.add_argument("--std-b", type=float, default=None)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--objective-score", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or os.environ.get("RUN_ID", "").strip()
    action = OuterBatchAction(
        mu_a=args.mu_a,
        mu_b=args.mu_b,
        ratio_a=args.ratio_a,
        num_files=args.num_files,
        seed=args.seed,
        pattern=args.pattern,
        std_a=args.std_a,
        std_b=args.std_b,
        dist=args.dist,
    )
    validate_action(action)

    out_root = Path(args.out_root).resolve()
    iter_dir = out_root / f"iter_{int(args.iter_id):03d}"
    r_dir = iter_dir / f"R{int(args.request_number)}"
    post_stage_dir = out_root.parent
    iter_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(action.seed))

    n_a, n_b = phase_counts(action.num_files, action.ratio_a)
    labels = build_phase_labels(
        num_files=action.num_files,
        ratio_a=action.ratio_a,
        pattern=action.pattern,
        seed=action.seed,
    )
    if len(labels) != action.num_files:
        raise RuntimeError("internal error: phase label count mismatch")

    print(_c("[OUTER][GEN] start", "cyan", True))
    print(
        f"[OUTER][ITER] iter={int(args.iter_id):03d} action="
        f"(muA={action.mu_a},muB={action.mu_b},p={action.ratio_a:.4f},n={action.num_files},seed={action.seed},pattern={action.pattern})"
    )
    print(f"[OUTER][GEN] out_root={out_root}")
    print(f"[OUTER][GEN] iter_dir={iter_dir}")
    print(f"[OUTER][GEN] request_number={int(args.request_number)} workers={int(args.workers)}")

    max_events = 60
    matrix = _build_duration_matrix(action=action, labels=labels, max_events=max_events)
    meta_rows: List[Dict[str, object]] = []
    row_means: List[float] = []
    for i, label in enumerate(labels):
        gt_mean = float(action.mu_a if label == "A" else action.mu_b)
        meta_rows.append({"gt_mean": gt_mean, "phase_label": label})
        row_means.append(float(np.mean(matrix[i])))

    start_gen = time.time()
    tasks = []
    for i in range(action.num_files):
        tasks.append((i, int(args.request_number), matrix[i], str(r_dir), meta_rows[i], int(action.seed)))

    failures = 0
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=base_gen.init_worker,
        initargs=(base_gen.DATA_FILE, base_gen.EXP_NUMBERS, base_gen.FIGURES_DIR),
    ) as executor:
        futures = [executor.submit(base_gen.generate_single_file, t) for t in tasks]
        iterator = tqdm(as_completed(futures), total=len(futures), unit="file", ncols=88) if HAS_TQDM else as_completed(futures)
        done = 0
        for future in iterator:
            done += 1
            ok = False
            try:
                ok = bool(future.result())
            except Exception:
                ok = False
            if not ok:
                failures += 1
            if (not HAS_TQDM) and done % 10 == 0:
                print(f"[OUTER][GEN] progress {done}/{len(futures)}")
    elapsed = time.time() - start_gen
    if failures > 0:
        raise RuntimeError(f"generation failed: {failures} files")

    if args.verify:
        verify_failures = base_gen.verify_output_dir(str(r_dir), action.num_files)
        if verify_failures:
            raise RuntimeError(f"verification failed on {verify_failures} files")

    generation_rows: List[Dict[str, object]] = []
    mean_a_values = [m for m, lbl in zip(row_means, labels) if lbl == "A"]
    mean_b_values = [m for m, lbl in zip(row_means, labels) if lbl == "B"]
    for i in range(action.num_files):
        file_path = r_dir / f"Intermodal_EGS_data_dynamic_congestion{i}.xlsx"
        file_hash = _sha256_file(file_path)
        generation_rows.append(
            {
                "iter_id": int(args.iter_id),
                "logical_idx": int(i),
                "file_path": str(file_path.resolve()),
                "gt_mean": float(meta_rows[i]["gt_mean"]),
                "phase_label": str(meta_rows[i]["phase_label"]),
                "hash": file_hash,
                "ok": 1,
            }
        )

    iter_manifest = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(),
        "run_id": run_id,
        "iter_id": int(args.iter_id),
        "request_number": int(args.request_number),
        "pattern": str(action.pattern),
        "num_files": int(action.num_files),
        "action": action_to_dict(action),
        "counts": {"n_a": int(n_a), "n_b": int(n_b)},
        "stats": {
            "ratio_a_target": float(action.ratio_a),
            "ratio_a_real": float(sum(1 for x in labels if x == "A") / max(1, len(labels))),
            "gt_mean_a_avg": float(np.mean(mean_a_values)) if mean_a_values else "",
            "gt_mean_b_avg": float(np.mean(mean_b_values)) if mean_b_values else "",
            "elapsed_s": float(elapsed),
        },
        "paths": {
            "out_root": str(out_root),
            "iter_dir": str(iter_dir),
            "r_dir": str(r_dir),
        },
        "files": generation_rows,
    }
    iter_manifest_path = iter_dir / "manifest.json"
    iter_manifest_path.write_text(json.dumps(iter_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    actions_csv = post_stage_dir / "outer_actions.csv"
    generation_csv = post_stage_dir / "outer_generation.csv"
    global_manifest = post_stage_dir / "manifest.json"
    action_row = {
        "iter_id": int(args.iter_id),
        "mu_a": float(action.mu_a),
        "mu_b": float(action.mu_b),
        "p": float(action.ratio_a),
        "n_files": int(action.num_files),
        "seed": int(action.seed),
        "pattern": str(action.pattern),
        "objective_score": "" if args.objective_score is None else float(args.objective_score),
    }
    _append_csv_row(
        actions_csv,
        ["iter_id", "mu_a", "mu_b", "p", "n_files", "seed", "pattern", "objective_score"],
        action_row,
    )
    for row in generation_rows:
        _append_csv_row(
            generation_csv,
            ["iter_id", "logical_idx", "file_path", "gt_mean", "phase_label", "hash", "ok"],
            row,
        )
    _update_global_manifest(global_manifest, iter_manifest_path, iter_manifest)

    print(
        f"[OUTER][GEN] done iter={int(args.iter_id):03d} files={action.num_files} ok={action.num_files} "
        f"fail=0 elapsed={elapsed:.2f}s"
    )
    print(
        f"[OUTER][GEN] stats ratio_A={sum(1 for x in labels if x == 'A')/len(labels):.4f} "
        f"ratio_B={sum(1 for x in labels if x == 'B')/len(labels):.4f}"
    )
    print(f"[OUTER][GEN] iter_manifest={iter_manifest_path}")


if __name__ == "__main__":
    main()

