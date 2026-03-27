#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional


def parse_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    succ_rows = [r for r in rows if parse_bool(r.get("success")) is True]
    crash_rows = [r for r in rows if parse_bool(r.get("crashed")) is True]

    succ_times = [parse_float(r.get("time_s")) for r in succ_rows]
    succ_times = [x for x in succ_times if x is not None]

    succ_jerk = [parse_float(r.get("max_jerk_mps3")) for r in succ_rows]
    succ_jerk = [x for x in succ_jerk if x is not None]

    return {
        "episodes": n,
        "successes": len(succ_rows),
        "crashes": len(crash_rows),
        "success_rate": (len(succ_rows) / n) if n else None,
        "crash_rate": (len(crash_rows) / n) if n else None,
        "time_p90_success": quantile(succ_times, 0.90),
        "max_jerk_p90_success": quantile(succ_jerk, 0.90),
    }


def fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser(
        description="Compare highway baseline and RL evaluation CSVs on success rate, P90 completion time, and P90 max jerk."
    )
    ap.add_argument("files", nargs="+", help="CSV files to compare")
    ap.add_argument("--out", default="results/compare/highway_rl_vs_baselines.csv")
    args = ap.parse_args()

    rows_out: List[Dict[str, Any]] = []
    for path in args.files:
        rows = read_rows(path)
        metrics = compute_metrics(rows)
        metrics["file"] = path
        rows_out.append(metrics)

    rows_out.sort(key=lambda r: (r["success_rate"] is not None, r["success_rate"]), reverse=True)

    print("file | episodes | successes | crashes | success_rate | p90_time_success | p90_max_jerk_success")
    print("-" * 120)
    for r in rows_out:
        print(
            " | ".join(
                [
                    os.path.basename(r["file"]),
                    fmt(r["episodes"]),
                    fmt(r["successes"]),
                    fmt(r["crashes"]),
                    fmt(r["success_rate"]),
                    fmt(r["time_p90_success"]),
                    fmt(r["max_jerk_p90_success"]),
                ]
            )
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file",
                "episodes",
                "successes",
                "crashes",
                "success_rate",
                "crash_rate",
                "time_p90_success",
                "max_jerk_p90_success",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
