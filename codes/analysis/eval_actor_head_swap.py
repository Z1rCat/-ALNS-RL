from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ENV_PYTHON = ROOT_DIR / "codes" / "env" / "python.exe"


def _maybe_reexec_into_repo_env() -> None:
    if not REPO_ENV_PYTHON.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        target = REPO_ENV_PYTHON.resolve()
    except Exception:
        return
    if current == target:
        return
    os.execv(str(target), [str(target), __file__, *sys.argv[1:]])


_maybe_reexec_into_repo_env()

if str(ROOT_DIR / "codes") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "codes"))

from stable_baselines3 import PPO  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a temporary checkpoint by swapping actor-side parameters between two PPO_NEW v3 checkpoints."
    )
    parser.add_argument("--late-ckpt", type=str, required=True, help="late checkpoint that provides the base model")
    parser.add_argument("--early-ckpt", type=str, required=True, help="early checkpoint that provides rollback params")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["action_head_only", "actor_branch_and_head", "action_head_interp"],
        default="action_head_only",
    )
    parser.add_argument(
        "--interp-alpha",
        type=float,
        default=0.50,
        help="interpolation weight for early params in action_head_interp mode",
    )
    parser.add_argument("--out-ckpt", type=str, required=True, help="output path for the mixed checkpoint")
    parser.add_argument("--manifest-path", type=str, default="", help="optional path for swap manifest json")
    parser.add_argument("--eval", action="store_true", help="run eval_checkpoint_pool.py on the output checkpoint")
    parser.add_argument("--eval-out-dir", type=str, default="")
    parser.add_argument("--dist-name", type=str, default="")
    parser.add_argument("--request-number", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="PPO_NEW")
    parser.add_argument("--algo-version", type=str, default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--external-data-root", type=str, default="")
    parser.add_argument("--timeout-sec", type=int, default=0)
    return parser.parse_args()


def _selected_keys(mode: str, policy_state: dict[str, Any]) -> list[str]:
    if mode == "action_head_only":
        prefixes = ("action_net.",)
    elif mode == "actor_branch_and_head":
        prefixes = ("mlp_extractor.policy_net.", "action_net.")
    elif mode == "action_head_interp":
        prefixes = ("action_net.",)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return [key for key in policy_state.keys() if key.startswith(prefixes)]


def _load_models(late_ckpt: Path, early_ckpt: Path) -> tuple[PPO, PPO]:
    late_model = PPO.load(str(late_ckpt), device="cpu")
    early_model = PPO.load(str(early_ckpt), device="cpu")
    return late_model, early_model


def _blend_tensor(late_tensor, early_tensor, alpha: float):
    return late_tensor * float(1.0 - alpha) + early_tensor * float(alpha)


def _apply_swap(
    late_model: PPO,
    early_model: PPO,
    *,
    mode: str,
    interp_alpha: float,
) -> dict[str, Any]:
    late_state = late_model.policy.state_dict()
    early_state = early_model.policy.state_dict()
    keys = _selected_keys(mode, late_state)
    if not keys:
        raise RuntimeError(f"no parameter keys matched mode={mode}")

    applied: list[str] = []
    for key in keys:
        if key not in early_state:
            raise KeyError(f"early checkpoint missing key: {key}")
        if tuple(late_state[key].shape) != tuple(early_state[key].shape):
            raise ValueError(
                f"shape mismatch for key={key}: late={tuple(late_state[key].shape)} early={tuple(early_state[key].shape)}"
            )
        if mode == "action_head_interp":
            late_state[key] = _blend_tensor(late_state[key], early_state[key], alpha=float(interp_alpha))
        else:
            late_state[key] = early_state[key]
        applied.append(key)

    late_model.policy.load_state_dict(late_state, strict=True)
    return {
        "mode": str(mode),
        "interp_alpha": float(interp_alpha),
        "applied_keys": applied,
        "applied_key_count": int(len(applied)),
    }


def _write_manifest(
    manifest_path: Path,
    *,
    late_ckpt: Path,
    early_ckpt: Path,
    out_ckpt: Path,
    swap_meta: dict[str, Any],
) -> None:
    payload = {
        "late_ckpt": str(late_ckpt),
        "early_ckpt": str(early_ckpt),
        "out_ckpt": str(out_ckpt),
        **swap_meta,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_eval(args: argparse.Namespace, out_ckpt: Path) -> None:
    if not str(args.dist_name).strip() or int(args.request_number) <= 0:
        raise ValueError("--dist-name and --request-number are required when --eval is used")
    if not str(args.eval_out_dir).strip():
        raise ValueError("--eval-out-dir is required when --eval is used")

    cmd = [
        sys.executable,
        str(ROOT_DIR / "codes" / "analysis" / "eval_checkpoint_pool.py"),
        "--checkpoint",
        str(out_ckpt.resolve()),
        "--dist-name",
        str(args.dist_name),
        "--request-number",
        str(int(args.request_number)),
        "--algorithm",
        str(args.algorithm),
        "--algo-version",
        str(args.algo_version),
        "--seed",
        str(int(args.seed)),
        "--workers",
        str(int(args.workers)),
        "--out-dir",
        str(Path(args.eval_out_dir).resolve()),
    ]
    if str(args.external_data_root).strip():
        cmd.extend(["--external-data-root", str(Path(args.external_data_root).resolve())])
    if int(args.timeout_sec) > 0:
        cmd.extend(["--timeout-sec", str(int(args.timeout_sec))])
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def main() -> None:
    args = _parse_args()
    late_ckpt = Path(str(args.late_ckpt)).resolve()
    early_ckpt = Path(str(args.early_ckpt)).resolve()
    out_ckpt = Path(str(args.out_ckpt)).resolve()
    manifest_path = (
        Path(str(args.manifest_path)).resolve()
        if str(args.manifest_path).strip()
        else out_ckpt.with_suffix(out_ckpt.suffix + ".swap_manifest.json")
    )

    late_model, early_model = _load_models(late_ckpt, early_ckpt)
    swap_meta = _apply_swap(
        late_model,
        early_model,
        mode=str(args.mode),
        interp_alpha=float(args.interp_alpha),
    )

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    late_model.save(str(out_ckpt))
    _write_manifest(
        manifest_path,
        late_ckpt=late_ckpt,
        early_ckpt=early_ckpt,
        out_ckpt=out_ckpt,
        swap_meta=swap_meta,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "out_ckpt": str(out_ckpt),
                "manifest_path": str(manifest_path),
                "mode": str(args.mode),
                "applied_key_count": int(swap_meta["applied_key_count"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if bool(args.eval):
        _run_eval(args, out_ckpt)


if __name__ == "__main__":
    main()
