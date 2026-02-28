import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return int(float(text))
    except Exception:
        return None


def _build_group_metrics(df):
    work = df.copy()
    work["_action"] = pd.to_numeric(work.get("action"), errors="coerce")
    work["_reward"] = pd.to_numeric(work.get("reward"), errors="coerce")
    work["_group"] = (
        work.get("phase_label", pd.Series(["unknown"] * len(work)))
        .fillna("unknown")
        .astype(str)
        .replace("", "unknown")
    )
    rows = []
    for group_name, g in work.groupby("_group", dropna=False):
        n_total = int(len(g))
        a1_mask = g["_action"] == 1
        a0_mask = g["_action"] == 0
        n_a1 = int(a1_mask.sum())
        n_a0 = int(a0_mask.sum())
        action1_rate = _ratio(n_a1, n_total)
        r_a1 = float(g.loc[a1_mask, "_reward"].mean()) if n_a1 > 0 else np.nan
        r_a0 = float(g.loc[a0_mask, "_reward"].mean()) if n_a0 > 0 else np.nan
        rows.append(
            {
                "group": str(group_name),
                "n_total": n_total,
                "action1_rate": action1_rate,
                "E[r|a=1]": r_a1,
                "E[r|a=0]": r_a0,
                "N(a=1)": n_a1,
                "N(a=0)": n_a0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["group", "n_total", "action1_rate", "E[r|a=1]", "E[r|a=0]", "N(a=1)", "N(a=0)"])
    return pd.DataFrame(rows).sort_values(by="n_total", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Validate rl_decision.csv + h_dump.npz artifacts.")
    parser.add_argument("--run-dir", type=str, required=True, help="run directory that contains rl_decision.csv and h_dump.npz")
    parser.add_argument("--decision-csv", type=str, default="", help="override rl_decision.csv path")
    parser.add_argument("--h-dump", type=str, default="", help="override h_dump.npz path")
    parser.add_argument("--key-groups", nargs="*", default=["O_10_90", "O_10_60"], help="key groups with minimum sample check")
    parser.add_argument("--min-pairing-rate", type=float, default=0.98)
    parser.add_argument("--min-h-align-rate", type=float, default=0.90)
    parser.add_argument("--min-group-samples", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    decision_csv = Path(args.decision_csv).resolve() if args.decision_csv else (run_dir / "rl_decision.csv")
    h_dump = Path(args.h_dump).resolve() if args.h_dump else (run_dir / "h_dump.npz")

    if not decision_csv.exists():
        print(f"[FAIL] decision csv not found: {decision_csv}")
        sys.exit(2)

    df = pd.read_csv(decision_csv)
    if len(df) == 0:
        print(f"[FAIL] decision csv is empty: {decision_csv}")
        sys.exit(2)

    if "matched" in df.columns:
        matched_mask = pd.to_numeric(df["matched"], errors="coerce").fillna(0).astype(int) == 1
    else:
        reward_col = df.get("reward", pd.Series([""] * len(df)))
        matched_mask = reward_col.fillna("").astype(str).str.strip() != ""
    pairing_rate = _ratio(int(matched_mask.sum()), int(len(df)))

    h_align_rate = 0.0
    h_rows_in_csv = 0
    h_rows_aligned = 0
    h_rows_oob = 0
    h_rows_id_mismatch = 0
    h_file_rows = 0
    if h_dump.exists():
        try:
            npz = np.load(h_dump, allow_pickle=True)
            h_ids = np.asarray(npz["decision_id"])
            h_file_rows = int(len(h_ids))
            h_idx_series = df.get("h_index", pd.Series([-1] * len(df)))
            for _, row in df.iterrows():
                h_idx = _to_int_or_none(row.get("h_index", None))
                if h_idx is None or h_idx < 0:
                    continue
                h_rows_in_csv += 1
                if h_idx >= len(h_ids):
                    h_rows_oob += 1
                    continue
                if str(h_ids[h_idx]) == str(row.get("decision_id", "")):
                    h_rows_aligned += 1
                else:
                    h_rows_id_mismatch += 1
            h_align_rate = _ratio(h_rows_aligned, int(len(df)))
        except Exception as exc:
            print(f"[WARN] failed to parse h_dump: {h_dump} ({exc})")
    else:
        print(f"[WARN] h_dump not found: {h_dump}")

    group_metrics = _build_group_metrics(df)
    group_count_map = {str(row["group"]): int(row["n_total"]) for _, row in group_metrics.iterrows()}
    key_group_rows = []
    for group in args.key_groups:
        count = int(group_count_map.get(str(group), 0))
        key_group_rows.append({"group": str(group), "n_total": count, "ok": int(count >= int(args.min_group_samples))})
    key_group_df = pd.DataFrame(key_group_rows)

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "total_decisions": int(len(df)),
                "pairing_rate_matched": pairing_rate,
                "h_align_rate": h_align_rate,
                "csv_rows_with_h_index": h_rows_in_csv,
                "h_rows_aligned": h_rows_aligned,
                "h_rows_oob": h_rows_oob,
                "h_rows_id_mismatch": h_rows_id_mismatch,
                "h_file_rows": h_file_rows,
                "min_pairing_rate_req": float(args.min_pairing_rate),
                "min_h_align_rate_req": float(args.min_h_align_rate),
                "min_group_samples_req": int(args.min_group_samples),
            }
        ]
    )

    print("=== summary ===")
    print(summary.to_string(index=False))
    print("")
    print("=== key_groups ===")
    print(key_group_df.to_string(index=False))
    print("")
    print("=== group_metrics ===")
    print(group_metrics.to_string(index=False))

    pass_pairing = pairing_rate >= float(args.min_pairing_rate)
    pass_h_align = h_align_rate >= float(args.min_h_align_rate)
    pass_groups = bool((key_group_df["ok"] == 1).all()) if len(key_group_df) > 0 else True

    if pass_pairing and pass_h_align and pass_groups:
        print("")
        print("[PASS] decision artifacts validated.")
        sys.exit(0)

    print("")
    print("[FAIL] decision artifacts validation failed.")
    print(
        f"pairing_ok={int(pass_pairing)} h_align_ok={int(pass_h_align)} key_groups_ok={int(pass_groups)}"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
