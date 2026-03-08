from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


def _to_bool(value: str) -> bool:
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-map-csv", type=str, required=True)
    parser.add_argument("--iter-id", type=str, default="", help="optional iter filter")
    parser.add_argument("--min-read-rate", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_map_csv = Path(args.path_map_csv).resolve()
    rows = _read_rows(path_map_csv)
    if not rows:
        raise RuntimeError(f"no rows found in {path_map_csv}")
    if str(args.iter_id).strip():
        rows = [r for r in rows if str(r.get("iter_id", "")).strip() == str(args.iter_id).strip()]
        if not rows:
            raise RuntimeError(f"no rows found for iter_id={args.iter_id}")

    total = len(rows)
    exists_ok = sum(1 for r in rows if _to_bool(r.get("exists", "")))
    read_ok = sum(1 for r in rows if _to_bool(r.get("read_ok", "")))
    exists_rate = float(exists_ok) / float(max(1, total))
    read_rate = float(read_ok) / float(max(1, total))

    print(
        f"[OUTER][PATH-CHECK] rows={total} exists_ok={exists_ok} read_ok={read_ok} "
        f"exists_rate={exists_rate:.6f} read_rate={read_rate:.6f} "
        f"min_read_rate={float(args.min_read_rate):.6f}"
    )
    if read_rate < float(args.min_read_rate):
        raise RuntimeError(f"path read rate too low: {read_rate:.6f} < {float(args.min_read_rate):.6f}")
    print("[OUTER][PATH-CHECK] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[OUTER][ERROR] {exc}")
        sys.exit(1)

