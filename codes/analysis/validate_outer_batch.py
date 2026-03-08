from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter-dir", type=str, required=True, help="Path like .../outer_batches/iter_001")
    parser.add_argument("--request-number", type=int, required=True)
    parser.add_argument("--expect-files", type=int, required=True)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--max-ratio-error", type=float, default=0.10)
    return parser.parse_args()


def _read_meta(path: Path) -> Dict[str, str]:
    meta_df = pd.read_excel(path, sheet_name="__meta__")
    out: Dict[str, str] = {}
    for _, row in meta_df.iterrows():
        key = str(row.get("Property", "")).strip()
        val = row.get("Value", "")
        if key:
            out[key] = "" if pd.isna(val) else str(val)
    return out


def main() -> None:
    args = parse_args()
    iter_dir = Path(args.iter_dir).resolve()
    r_dir = iter_dir / f"R{int(args.request_number)}"
    if not r_dir.exists():
        raise FileNotFoundError(f"missing directory: {r_dir}")

    missing: List[str] = []
    bad_meta: List[str] = []
    label_counts = {"A": 0, "B": 0}
    mean_values = {"A": [], "B": []}
    for i in range(int(args.expect_files)):
        path = r_dir / f"Intermodal_EGS_data_dynamic_congestion{i}.xlsx"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            meta = _read_meta(path)
        except Exception as exc:
            bad_meta.append(f"{path} ({exc})")
            continue
        label = str(meta.get("phase_label", "")).strip()
        gt_mean = str(meta.get("gt_mean", "")).strip()
        if label not in {"A", "B"}:
            bad_meta.append(f"{path} (invalid phase_label={label!r})")
            continue
        label_counts[label] += 1
        try:
            mean_values[label].append(float(gt_mean))
        except Exception:
            bad_meta.append(f"{path} (invalid gt_mean={gt_mean!r})")

    total_present = int(sum(label_counts.values()))
    if missing:
        print(f"[OUTER][CHECK] missing_files={len(missing)}")
        for item in missing[:10]:
            print(f"  - {item}")
    if bad_meta:
        print(f"[OUTER][CHECK] bad_meta={len(bad_meta)}")
        for item in bad_meta[:10]:
            print(f"  - {item}")

    if args.check_manifest:
        manifest_path = iter_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = payload.get("files", [])
        if not isinstance(files, list):
            raise ValueError("manifest.files is not a list")
        if len(files) != int(args.expect_files):
            raise ValueError(f"manifest file count mismatch: {len(files)} != {int(args.expect_files)}")
        action_obj = payload.get("action", {})
        ratio_target = None
        if isinstance(action_obj, dict):
            try:
                ratio_target = float(action_obj.get("ratio_a", "nan"))
                if ratio_target != ratio_target:
                    ratio_target = None
            except Exception:
                ratio_target = None
    else:
        ratio_target = None

    ratio_a = label_counts["A"] / max(1, total_present)
    ratio_b = label_counts["B"] / max(1, total_present)
    print(
        f"[OUTER][CHECK] total={total_present}/{int(args.expect_files)} "
        f"ratio_A={ratio_a:.4f} ratio_B={ratio_b:.4f} "
        f"mean_A={sum(mean_values['A'])/max(1,len(mean_values['A'])):.3f} "
        f"mean_B={sum(mean_values['B'])/max(1,len(mean_values['B'])):.3f}"
    )

    ratio_violation = False
    if ratio_target is not None:
        ratio_err = abs(float(ratio_a) - float(ratio_target))
        print(
            f"[OUTER][CHECK] ratio_target={ratio_target:.4f} "
            f"ratio_error={ratio_err:.4f} max_ratio_error={float(args.max_ratio_error):.4f}"
        )
        if ratio_err > float(args.max_ratio_error):
            ratio_violation = True

    if missing or bad_meta or total_present != int(args.expect_files) or ratio_violation:
        raise RuntimeError("outer batch validation failed")
    print("[OUTER][CHECK] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[OUTER][ERROR] {exc}")
        sys.exit(1)
