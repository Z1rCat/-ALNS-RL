from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"


def _safe_float(value, default: float = float("-inf")) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _resolve_run_root(run_id: str) -> Path:
    p = Path(run_id)
    if p.is_absolute():
        return p
    return (CODES_DIR / "logs" / run_id).resolve()


def _print_cmd(tag: str, cmd: List[str]) -> None:
    pretty = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[PIPELINE][{tag}] {pretty}")


def _run_or_fail(tag: str, cmd: List[str]) -> None:
    _print_cmd(tag, cmd)
    subprocess.run(cmd, check=True)


def _run_with_code(tag: str, cmd: List[str]) -> int:
    _print_cmd(tag, cmd)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _build_phase1_cmd(
    args: argparse.Namespace,
    python_bin: str,
    run_name: str,
    phase1_ckpt: Path,
) -> List[str]:
    cmd = [
        python_bin,
        str(CODES_DIR / "Dynamic_master34959.py"),
        "--dist_name",
        str(args.phase1_dist_name),
        "--request_number",
        str(int(args.phase1_request_number)),
        "--algorithm",
        str(args.phase1_algorithm),
        "--algo_version",
        str(args.phase1_algo_version),
        "--stage-mode",
        "train_only",
        "--run-name",
        str(run_name),
        "--seed",
        str(int(args.phase1_seed)),
        "--save-model-path",
        str(phase1_ckpt),
    ]
    if int(args.phase1_workers) > 0:
        cmd.extend(["--workers", str(int(args.phase1_workers))])
    if str(args.phase1_init_model_path).strip():
        cmd.extend(["--init-model-path", str(Path(args.phase1_init_model_path).resolve())])
    if bool(args.phase1_skip_generator):
        cmd.append("--skip-generator")
    if str(args.phase1_external_data_root).strip():
        cmd.extend(["--external-data-root", str(Path(args.phase1_external_data_root).resolve())])
    return cmd


def _build_outer_cmd(
    args: argparse.Namespace,
    python_bin: str,
    run_name: str,
    base_ckpt: Optional[Path],
    phase1_data_root: Path,
    passthrough: List[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(CODES_DIR / "outer_rl" / "run_edrl_phase2.py"),
        "--run-id",
        str(run_name),
        "--outer-phase",
        str(args.outer_phase),
        "--dist-name",
        str(args.outer_dist_name),
        "--request-number",
        str(int(args.outer_request_number)),
        "--algorithm",
        str(args.outer_algorithm),
        "--algo-version",
        str(args.outer_algo_version),
        "--iterations",
        str(int(args.outer_iterations)),
        "--seed",
        str(int(args.outer_seed)),
        "--policy-mode",
        str(args.outer_policy_mode),
        "--warmup-iters",
        str(int(args.outer_warmup_iters)),
        "--policy-decay",
        str(float(args.outer_policy_decay)),
        "--ts-prior-mean",
        str(float(args.outer_ts_prior_mean)),
        "--ts-prior-std",
        str(float(args.outer_ts_prior_std)),
        "--ts-obs-std",
        str(float(args.outer_ts_obs_std)),
        "--edrl-version",
        str(args.outer_edrl_version),
        "--mu-choices",
        str(args.outer_mu_choices),
        "--ratio-choices",
        str(args.outer_ratio_choices),
        "--num-file-choices",
        str(args.outer_num_file_choices),
        "--pattern-choices",
        str(args.outer_pattern_choices),
        "--action-space-version",
        str(args.outer_action_space_version),
        "--v2-fixed-ratio-a",
        str(float(args.outer_v2_fixed_ratio_a)),
        "--v2-fixed-pattern",
        str(args.outer_v2_fixed_pattern),
        "--v2-fixed-num-files",
        str(int(args.outer_v2_fixed_num_files)),
        "--inner-stop-mode",
        str(args.outer_inner_stop_mode),
        "--inner-fixed-n",
        str(int(args.outer_inner_fixed_n)),
        "--phase2-fixed-num-files",
        str(int(args.phase2_fixed_num_files)),
        "--phase3-num-file-choices",
        str(args.phase3_num_file_choices),
        "--workers",
        str(int(args.outer_workers)),
        "--gen-retry-max",
        str(int(args.outer_gen_retry_max)),
        "--phase2-min-iters",
        str(int(args.phase2_min_iters)),
        "--phase2-max-iters",
        str(int(args.phase2_max_iters)),
        "--phase3-min-iters",
        str(int(args.phase3_min_iters)),
        "--phase3-max-iters",
        str(int(args.phase3_max_iters)),
        "--converge-patience",
        str(int(args.converge_patience)),
        "--phase3-converge-patience",
        str(int(args.phase3_converge_patience)),
        "--converge-max-abs-dj",
        str(float(args.converge_max_abs_dj)),
        "--converge-max-obj-range",
        str(float(args.converge_max_obj_range)),
        "--phase2-converge-max-abs-dj",
        str(float(args.phase2_converge_max_abs_dj)),
        "--phase2-converge-max-obj-range",
        str(float(args.phase2_converge_max_obj_range)),
        "--converge-minority-floor",
        str(float(args.converge_minority_floor)),
        "--phase3-topk-k",
        str(int(args.phase3_topk_k)),
        "--phase3-topk-warmup-iters",
        str(int(args.phase3_topk_warmup_iters)),
        "--phase3-topk-prior-count",
        str(float(args.phase3_topk_prior_count)),
        "--rho-target",
        str(float(args.rho_target)),
        "--rho-floor",
        str(float(args.rho_floor)),
        "--eta-collapse",
        str(float(args.eta_collapse)),
        "--rho-floor-weight",
        str(float(args.rho_floor_weight)),
        "--rho-floor-hard-weight",
        str(float(args.rho_floor_hard_weight)),
        "--collapse-gap-power",
        str(float(args.collapse_gap_power)),
    ]
    if str(args.outer_edrl_version).strip().lower() == "v3":
        cmd.extend(
            [
                "--edrl-v3-dj-weight",
                str(float(args.outer_edrl_v3_dj_weight)),
                "--edrl-v3-j-weight",
                str(float(args.outer_edrl_v3_j_weight)),
                "--edrl-v3-minority-abs-weight",
                str(float(args.outer_edrl_v3_minority_abs_weight)),
            ]
        )
        cmd.append("--edrl-v3-level-replay" if bool(args.outer_edrl_v3_level_replay) else "--no-edrl-v3-level-replay")
        cmd.append(
            "--edrl-v3-replay-phase3-only"
            if bool(args.outer_edrl_v3_replay_phase3_only)
            else "--no-edrl-v3-replay-phase3-only"
        )
    elif str(args.outer_edrl_version).strip().lower() == "v4":
        cmd.extend(
            [
                "--edrl-v4-challenge-weight",
                str(float(args.outer_edrl_v4_challenge_weight)),
                "--edrl-v4-lp-weight",
                str(float(args.outer_edrl_v4_lp_weight)),
                "--edrl-v4-j-weight",
                str(float(args.outer_edrl_v4_j_weight)),
                "--edrl-v4-entropy-weight",
                str(float(args.outer_edrl_v4_entropy_weight)),
                "--edrl-v4-minority-weight",
                str(float(args.outer_edrl_v4_minority_weight)),
                "--edrl-v4-minority-abs-weight",
                str(float(args.outer_edrl_v4_minority_abs_weight)),
                "--edrl-v4-novelty-weight",
                str(float(args.outer_edrl_v4_novelty_weight)),
                "--edrl-v4-j-center",
                str(float(args.outer_edrl_v4_j_center)),
                "--edrl-v4-j-sigma",
                str(float(args.outer_edrl_v4_j_sigma)),
                "--edrl-v4-p-new-k",
                str(float(args.outer_edrl_v4_p_new_k)),
                "--edrl-v4-p-new-min",
                str(float(args.outer_edrl_v4_p_new_min)),
                "--edrl-v4-p-new-max",
                str(float(args.outer_edrl_v4_p_new_max)),
                "--edrl-v4-entropy-target",
                str(float(args.outer_edrl_v4_entropy_target)),
                "--plr-p-new",
                str(float(args.outer_plr_p_new)),
                "--plr-buffer-size",
                str(int(args.outer_plr_buffer_size)),
                "--plr-priority-ema-alpha",
                str(float(args.outer_plr_priority_ema_alpha)),
                "--plr-min-weight",
                str(float(args.outer_plr_min_weight)),
            ]
        )
        cmd.append("--edrl-v4-level-replay" if bool(args.outer_edrl_v4_level_replay) else "--no-edrl-v4-level-replay")
        cmd.append(
            "--edrl-v4-replay-phase3-only"
            if bool(args.outer_edrl_v4_replay_phase3_only)
            else "--no-edrl-v4-replay-phase3-only"
        )
    if bool(args.outer_auto_stop):
        cmd.append("--outer-auto-stop")
    if bool(args.outer_disable_fixed_n_sync):
        cmd.append("--disable-fixed-n-sync")
    if bool(args.outer_verify_batch):
        cmd.append("--verify-batch")
    if bool(args.outer_validate_path_map):
        cmd.append("--validate-path-map")
    phase_requires_curriculum = str(args.outer_phase).strip().lower() in {"phase3", "auto"}
    curriculum_on = bool(args.outer_curriculum_enable) or phase_requires_curriculum
    curriculum_base_root = str(args.outer_curriculum_base_root).strip()
    if curriculum_on and (not curriculum_base_root):
        curriculum_base_root = str(phase1_data_root)
    if curriculum_on:
        cmd.append("--curriculum-enable")
        if not curriculum_base_root:
            raise ValueError(
                "curriculum is enabled (phase3/auto or --outer-curriculum-enable), "
                "but --outer-curriculum-base-root is empty"
            )
        cmd.extend(["--curriculum-base-root", str(Path(curriculum_base_root).resolve())])
        cmd.extend(
            [
                "--curriculum-alpha-start",
                str(float(args.outer_curriculum_alpha_start)),
                "--curriculum-alpha-end",
                str(float(args.outer_curriculum_alpha_end)),
                "--curriculum-alpha-horizon",
                str(int(args.outer_curriculum_alpha_horizon)),
                "--curriculum-replay-ratio",
                str(float(args.outer_curriculum_replay_ratio)),
                "--curriculum-replay-max-iters",
                str(int(args.outer_curriculum_replay_max_iters)),
            ]
        )
    if base_ckpt is not None and base_ckpt.exists():
        cmd.extend(["--base-ckpt", str(base_ckpt)])
    if passthrough:
        cmd.extend(passthrough)
    return cmd


def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_phase4_ckpt(post_stage: Path, policy: str = "latest_phase3") -> Tuple[Path, Dict[str, object]]:
    train_round_csv = post_stage / "outer_train_round.csv"
    rows = _read_csv_rows(train_round_csv)
    policy_name = str(policy or "latest_phase3").strip().lower()
    chosen: Optional[Path] = None
    chosen_meta: Dict[str, object] = {
        "policy": policy_name,
        "source": "",
        "phase": "",
        "iter_id": "",
        "objective_score": "",
    }

    valid_rows: List[Tuple[Path, Dict[str, str]]] = []
    phase3_rows: List[Tuple[Path, Dict[str, str]]] = []
    for row in rows:
        ckpt_out = str(row.get("ckpt_out", "")).strip()
        if not ckpt_out:
            continue
        p = Path(ckpt_out).resolve()
        if not p.exists():
            continue
        item = (p, row)
        valid_rows.append(item)
        if str(row.get("phase", "")).strip().lower() == "phase3":
            phase3_rows.append(item)

    if policy_name == "best_phase3_objective":
        ranked = sorted(
            phase3_rows,
            key=lambda item: (
                _safe_float(item[1].get("objective_score", ""), default=float("-inf")),
                _safe_float(item[1].get("objective", ""), default=float("-inf")),
                _safe_float(item[1].get("iter_id", ""), default=float("-inf")),
            ),
            reverse=True,
        )
        if ranked:
            chosen, row = ranked[0]
            chosen_meta.update(
                {
                    "source": "outer_train_round.csv",
                    "phase": str(row.get("phase", "")),
                    "iter_id": str(row.get("iter_id", "")),
                    "objective_score": _safe_float(
                        row.get("objective_score", row.get("objective", "")),
                        default=float("nan"),
                    ),
                }
            )
    elif policy_name == "latest_any":
        if valid_rows:
            chosen, row = valid_rows[-1]
            chosen_meta.update(
                {
                    "source": "outer_train_round.csv",
                    "phase": str(row.get("phase", "")),
                    "iter_id": str(row.get("iter_id", "")),
                    "objective_score": _safe_float(
                        row.get("objective_score", row.get("objective", "")),
                        default=float("nan"),
                    ),
                }
            )
    else:
        if phase3_rows:
            chosen, row = phase3_rows[-1]
            chosen_meta.update(
                {
                    "source": "outer_train_round.csv",
                    "phase": str(row.get("phase", "")),
                    "iter_id": str(row.get("iter_id", "")),
                    "objective_score": _safe_float(
                        row.get("objective_score", row.get("objective", "")),
                        default=float("nan"),
                    ),
                }
            )
        elif valid_rows:
            chosen, row = valid_rows[-1]
            chosen_meta.update(
                {
                    "source": "outer_train_round.csv_fallback",
                    "phase": str(row.get("phase", "")),
                    "iter_id": str(row.get("iter_id", "")),
                    "objective_score": _safe_float(
                        row.get("objective_score", row.get("objective", "")),
                        default=float("nan"),
                    ),
                }
            )

    if chosen is None:
        ckpt_dir = post_stage / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("theta_iter*.zip"))
        if ckpts:
            chosen = ckpts[-1].resolve()
            chosen_meta.update(
                {
                    "source": "checkpoints_dir_fallback",
                    "phase": "",
                    "iter_id": "",
                    "objective_score": "",
                }
            )
    if chosen is None:
        raise FileNotFoundError("phase4 init checkpoint not found from outer stage")
    return chosen, chosen_meta


def _build_phase4_cmd(
    args: argparse.Namespace,
    python_bin: str,
    run_name: str,
    phase4_ckpt: Path,
    phase1_data_root: Path,
) -> List[str]:
    phase4_dist = str(args.phase4_dist_name).strip() or str(args.phase1_dist_name)
    phase4_req = int(args.phase4_request_number) if int(args.phase4_request_number) > 0 else int(args.phase1_request_number)
    phase4_algo = str(args.phase4_algorithm).strip() or str(args.phase1_algorithm)
    if str(phase4_algo).strip().upper() == "NOVA_EDRL":
        # phase4 must reuse master implement path, not recursively invoke the NOVA pipeline.
        phase4_algo = str(args.phase1_algorithm)
    phase4_ver = str(args.phase4_algo_version).strip() or str(args.phase1_algo_version)
    phase4_seed = int(args.phase4_seed) if int(args.phase4_seed) >= 0 else int(args.phase1_seed)

    cmd = [
        python_bin,
        str(CODES_DIR / "Dynamic_master34959.py"),
        "--dist_name",
        str(phase4_dist),
        "--request_number",
        str(int(phase4_req)),
        "--algorithm",
        str(phase4_algo),
        "--algo_version",
        str(phase4_ver),
        "--stage-mode",
        "eval_only",
        "--run-name",
        str(run_name),
        "--seed",
        str(int(phase4_seed)),
        "--skip-generator",
        "--external-data-root",
        str(phase1_data_root.resolve()),
        "--init-model-path",
        str(phase4_ckpt.resolve()),
    ]
    if int(args.phase4_workers) > 0:
        cmd.extend(["--workers", str(int(args.phase4_workers))])
    return cmd


def _write_pipeline_summary(
    post_stage: Path,
    run_root: Path,
    run_name: str,
    phase1_ckpt: Optional[Path],
    phase4_ckpt: Optional[Path],
    phase4_ckpt_meta: Optional[Dict[str, object]],
    phase1_data_root: Path,
    phase4_ran: bool,
) -> None:
    payload = {
        "run_id": str(run_name),
        "run_root": str(run_root),
        "generated_at_ts": float(time.time()),
        "phase1_ckpt": "" if phase1_ckpt is None else str(phase1_ckpt),
        "phase4_init_ckpt": "" if phase4_ckpt is None else str(phase4_ckpt),
        "phase4_ckpt_policy": "" if not phase4_ckpt_meta else str(phase4_ckpt_meta.get("policy", "")),
        "phase4_ckpt_source": "" if not phase4_ckpt_meta else str(phase4_ckpt_meta.get("source", "")),
        "phase4_ckpt_phase": "" if not phase4_ckpt_meta else str(phase4_ckpt_meta.get("phase", "")),
        "phase4_ckpt_iter_id": "" if not phase4_ckpt_meta else str(phase4_ckpt_meta.get("iter_id", "")),
        "phase4_ckpt_objective_score": "" if not phase4_ckpt_meta else phase4_ckpt_meta.get("objective_score", ""),
        "phase1_data_root": str(phase1_data_root),
        "phase4_ran": int(bool(phase4_ran)),
    }
    out_path = post_stage / "pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_phase1_data_root(args: argparse.Namespace, run_root: Path) -> Path:
    raw = str(args.phase1_external_data_root).strip()
    if raw:
        return Path(raw).resolve()
    run_id_raw = str(getattr(args, "run_id", "") or "").strip()
    if run_id_raw:
        run_id_path = Path(run_id_raw)
        if run_id_path.is_absolute():
            return (run_id_path / "data").resolve()
        rl_log_root = str(os.environ.get("RL_LOG_ROOT", "") or "").strip()
        if rl_log_root:
            return (Path(rl_log_root).resolve() / run_id_raw / "data").resolve()
    return (run_root / "data").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end V3+EDRL pipeline: phase1(inner train_only with 1000-file generation) "
            "then phase2/phase3 outer RL orchestration, and phase4(master eval_only implement)."
        )
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--python-bin", type=str, default=sys.executable)

    parser.add_argument("--phase1-skip", action="store_true", help="skip phase1 warmup")
    parser.add_argument("--phase1-dist-name", type=str, default="O_10_90")
    parser.add_argument("--phase1-request-number", type=int, default=30)
    parser.add_argument("--phase1-algorithm", type=str, default="PPO_NEW")
    parser.add_argument("--phase1-algo-version", type=str, default="v3")
    parser.add_argument("--phase1-seed", type=int, default=42)
    parser.add_argument("--phase1-workers", type=int, default=2)
    parser.add_argument("--phase1-init-model-path", type=str, default="")
    parser.add_argument("--phase1-save-model-path", type=str, default="")
    parser.add_argument("--phase1-skip-generator", action="store_true")
    parser.add_argument("--phase1-external-data-root", type=str, default="")

    parser.add_argument("--outer-phase", type=str, default="auto", choices=["phase2", "phase3", "auto"])
    parser.add_argument("--outer-auto-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-dist-name", type=str, default="O_10_90")
    parser.add_argument("--outer-request-number", type=int, default=30)
    parser.add_argument("--outer-algorithm", type=str, default="PPO_NEW")
    parser.add_argument("--outer-algo-version", type=str, default="v3")
    parser.add_argument("--outer-iterations", type=int, default=200)
    parser.add_argument("--outer-seed", type=int, default=42)
    parser.add_argument("--outer-policy-mode", type=str, default="ts")
    parser.add_argument("--outer-warmup-iters", type=int, default=1)
    parser.add_argument("--outer-policy-decay", type=float, default=1.0)
    parser.add_argument("--outer-ts-prior-mean", type=float, default=0.0)
    parser.add_argument("--outer-ts-prior-std", type=float, default=0.5)
    parser.add_argument("--outer-ts-obs-std", type=float, default=0.2)
    parser.add_argument("--outer-edrl-version", type=str, default="v1", choices=["v1", "v3", "v4"])
    parser.add_argument("--outer-edrl-v3-dj-weight", type=float, default=0.6)
    parser.add_argument("--outer-edrl-v3-j-weight", type=float, default=0.1)
    parser.add_argument("--outer-edrl-v3-minority-abs-weight", type=float, default=0.5)
    parser.add_argument("--outer-edrl-v3-level-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-edrl-v3-replay-phase3-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-edrl-v4-challenge-weight", type=float, default=0.40)
    parser.add_argument("--outer-edrl-v4-lp-weight", type=float, default=1.00)
    parser.add_argument("--outer-edrl-v4-j-weight", type=float, default=0.10)
    parser.add_argument("--outer-edrl-v4-entropy-weight", type=float, default=0.35)
    parser.add_argument("--outer-edrl-v4-minority-weight", type=float, default=1.20)
    parser.add_argument("--outer-edrl-v4-minority-abs-weight", type=float, default=0.80)
    parser.add_argument("--outer-edrl-v4-novelty-weight", type=float, default=0.20)
    parser.add_argument("--outer-edrl-v4-j-center", type=float, default=0.55)
    parser.add_argument("--outer-edrl-v4-j-sigma", type=float, default=0.20)
    parser.add_argument("--outer-edrl-v4-p-new-k", type=float, default=0.80)
    parser.add_argument("--outer-edrl-v4-p-new-min", type=float, default=0.20)
    parser.add_argument("--outer-edrl-v4-p-new-max", type=float, default=0.90)
    parser.add_argument("--outer-edrl-v4-entropy-target", type=float, default=0.25)
    parser.add_argument("--outer-edrl-v4-level-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-edrl-v4-replay-phase3-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outer-plr-p-new", type=float, default=0.55)
    parser.add_argument("--outer-plr-buffer-size", type=int, default=300)
    parser.add_argument("--outer-plr-priority-ema-alpha", type=float, default=0.50)
    parser.add_argument("--outer-plr-min-weight", type=float, default=0.02)
    parser.add_argument("--outer-mu-choices", type=str, default="10,30,60,90")
    parser.add_argument("--outer-ratio-choices", type=str, default="0.2,0.3,0.5,0.7,0.8")
    parser.add_argument("--outer-num-file-choices", type=str, default="5,10,15")
    parser.add_argument("--outer-pattern-choices", type=str, default="ab,random_mix")
    parser.add_argument(
        "--outer-action-space-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="v1: full outer action; v2: mu-only outer action",
    )
    parser.add_argument("--outer-v2-fixed-ratio-a", type=float, default=0.5)
    parser.add_argument("--outer-v2-fixed-pattern", type=str, default="ab")
    parser.add_argument("--outer-v2-fixed-num-files", type=int, default=0)
    parser.add_argument("--outer-inner-stop-mode", type=str, default="fixed_n")
    parser.add_argument("--outer-inner-fixed-n", type=int, default=0)
    parser.add_argument("--phase2-fixed-num-files", type=int, default=5)
    parser.add_argument("--phase3-num-file-choices", type=str, default="5,10,15")
    parser.add_argument("--outer-disable-fixed-n-sync", action="store_true")
    parser.add_argument("--outer-workers", type=int, default=2)
    parser.add_argument("--outer-gen-retry-max", type=int, default=2)
    parser.add_argument("--outer-verify-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-validate-path-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outer-base-ckpt", type=str, default="")

    parser.add_argument("--outer-curriculum-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outer-curriculum-base-root", type=str, default="")
    parser.add_argument("--outer-curriculum-alpha-start", type=float, default=0.7)
    parser.add_argument("--outer-curriculum-alpha-end", type=float, default=0.35)
    parser.add_argument("--outer-curriculum-alpha-horizon", type=int, default=25)
    parser.add_argument("--outer-curriculum-replay-ratio", type=float, default=0.2)
    parser.add_argument("--outer-curriculum-replay-max-iters", type=int, default=5)

    parser.add_argument("--phase2-min-iters", type=int, default=5)
    parser.add_argument("--phase2-max-iters", type=int, default=40)
    parser.add_argument("--phase3-min-iters", type=int, default=10)
    parser.add_argument("--phase3-max-iters", type=int, default=50)
    parser.add_argument("--converge-patience", type=int, default=2)
    parser.add_argument("--phase3-converge-patience", type=int, default=1)
    parser.add_argument("--converge-max-abs-dj", type=float, default=0.20)
    parser.add_argument("--converge-max-obj-range", type=float, default=0.50)
    parser.add_argument("--phase2-converge-max-abs-dj", type=float, default=0.80)
    parser.add_argument("--phase2-converge-max-obj-range", type=float, default=1.00)
    parser.add_argument("--converge-minority-floor", type=float, default=0.01)
    parser.add_argument("--phase3-topk-k", type=int, default=0)
    parser.add_argument("--phase3-topk-warmup-iters", type=int, default=0)
    parser.add_argument("--phase3-topk-prior-count", type=float, default=0.0)

    parser.add_argument("--rho-target", type=float, default=0.22)
    parser.add_argument("--rho-floor", type=float, default=0.10)
    parser.add_argument("--eta-collapse", type=float, default=1.4)
    parser.add_argument("--rho-floor-weight", type=float, default=4.0)
    parser.add_argument("--rho-floor-hard-weight", type=float, default=12.0)
    parser.add_argument("--collapse-gap-power", type=float, default=2.0)

    parser.add_argument("--phase4-skip", action="store_true", help="skip phase4 eval_only implement stage")
    parser.add_argument("--phase4-dist-name", type=str, default="")
    parser.add_argument("--phase4-request-number", type=int, default=-1)
    parser.add_argument("--phase4-algorithm", type=str, default="")
    parser.add_argument("--phase4-algo-version", type=str, default="")
    parser.add_argument("--phase4-seed", type=int, default=-1)
    parser.add_argument("--phase4-workers", type=int, default=2)
    parser.add_argument(
        "--phase4-ckpt-policy",
        type=str,
        default="latest_phase3",
        choices=["latest_phase3", "best_phase3_objective", "latest_any"],
        help="phase4 init checkpoint selection policy; default preserves current latest-phase3 behavior",
    )

    args, passthrough = parser.parse_known_args()
    args.outer_passthrough = passthrough
    return args


def main() -> None:
    args = parse_args()
    python_bin = str(Path(args.python_bin).resolve())
    run_root = _resolve_run_root(args.run_id)
    post_stage = run_root / "post_stage"
    ckpt_dir = post_stage / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(args.run_id)

    if str(args.phase1_save_model_path).strip():
        phase1_ckpt = Path(args.phase1_save_model_path).resolve()
    else:
        phase1_ckpt = (ckpt_dir / "theta_phase1.zip").resolve()
    phase1_data_root = _resolve_phase1_data_root(args=args, run_root=run_root)

    print(f"[PIPELINE][INIT] run_root={run_root}")
    print(f"[PIPELINE][INIT] phase1_ckpt={phase1_ckpt}")
    print(f"[PIPELINE][INIT] phase1_data_root={phase1_data_root}")

    if not bool(args.phase1_skip):
        phase1_cmd = _build_phase1_cmd(
            args=args,
            python_bin=python_bin,
            run_name=run_name,
            phase1_ckpt=phase1_ckpt,
        )
        phase1_code = _run_with_code("PHASE1", phase1_cmd)
        if phase1_code != 0:
            if phase1_ckpt.exists():
                print(
                    f"[PIPELINE][PHASE1][WARN] non-zero exit_code={phase1_code} but checkpoint exists; "
                    "continue to OUTER with produced checkpoint."
                )
            else:
                raise subprocess.CalledProcessError(phase1_code, phase1_cmd)
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"phase1 checkpoint not found: {phase1_ckpt}")
        print(f"[PIPELINE][PHASE1] done ckpt={phase1_ckpt}")
    else:
        print("[PIPELINE][PHASE1] skipped by --phase1-skip")

    base_ckpt: Optional[Path] = None
    if str(args.outer_base_ckpt).strip():
        base_ckpt = Path(args.outer_base_ckpt).resolve()
    elif phase1_ckpt.exists():
        base_ckpt = phase1_ckpt

    outer_cmd = _build_outer_cmd(
        args=args,
        python_bin=python_bin,
        run_name=run_name,
        base_ckpt=base_ckpt,
        phase1_data_root=phase1_data_root,
        passthrough=list(args.outer_passthrough),
    )
    _run_or_fail("OUTER", outer_cmd)

    phase4_ckpt: Optional[Path] = None
    phase4_ckpt_meta: Optional[Dict[str, object]] = None
    phase4_ran = False
    if bool(args.phase4_skip):
        print("[PIPELINE][PHASE4] skipped by --phase4-skip")
    else:
        if not phase1_data_root.exists():
            raise FileNotFoundError(
                f"phase4 requires phase1 data root for eval_only implement, but it does not exist: {phase1_data_root}"
            )
        phase4_ckpt, phase4_ckpt_meta = _pick_phase4_ckpt(
            post_stage=post_stage,
            policy=str(args.phase4_ckpt_policy),
        )
        print(
            "[PIPELINE][PHASE4] "
            f"ckpt_policy={str(args.phase4_ckpt_policy)} "
            f"init_ckpt={phase4_ckpt} "
            f"source={'' if not phase4_ckpt_meta else phase4_ckpt_meta.get('source', '')} "
            f"iter={'' if not phase4_ckpt_meta else phase4_ckpt_meta.get('iter_id', '')} "
            f"objective={'' if not phase4_ckpt_meta else phase4_ckpt_meta.get('objective_score', '')}"
        )
        phase4_cmd = _build_phase4_cmd(
            args=args,
            python_bin=python_bin,
            run_name=run_name,
            phase4_ckpt=phase4_ckpt,
            phase1_data_root=phase1_data_root,
        )
        _run_or_fail("PHASE4", phase4_cmd)
        phase4_ran = True
        print("[PIPELINE][PHASE4] done (master eval_only implement reused)")

    phase1_ckpt_for_summary = phase1_ckpt if phase1_ckpt.exists() else (base_ckpt if base_ckpt and base_ckpt.exists() else None)
    _write_pipeline_summary(
        post_stage=post_stage,
        run_root=run_root,
        run_name=run_name,
        phase1_ckpt=phase1_ckpt_for_summary,
        phase4_ckpt=phase4_ckpt,
        phase4_ckpt_meta=phase4_ckpt_meta,
        phase1_data_root=phase1_data_root,
        phase4_ran=phase4_ran,
    )
    print("[PIPELINE][DONE] phase1->phase2/phase3->phase4 pipeline finished")


if __name__ == "__main__":
    main()
