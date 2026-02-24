#!/usr/bin/env python3
"""
Analyze CARLA baseline-run CSVs (e.g., behavior_normal_20.csv) and compare performance.

Usage:
  python tools/analyze_results.py results/left_turn/behavior_normal_20.csv
  python tools/analyze_results.py results/left_turn/*.csv
  python tools/analyze_results.py --dir results --recursive
  python tools/analyze_results.py --dir results/left_turn --glob "*.csv" --out results/summary.csv
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import statistics as stats
from typing import Any, Dict, List, Tuple, Optional


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
    """Simple percentile/quantile without numpy. q in [0,1]."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    # linear interpolation between closest ranks
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def safe_mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def safe_median(values: List[float]) -> Optional[float]:
    return stats.median(values) if values else None


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Standard columns you likely have
    # success, crashed, time_s, max_jerk_mps3, rms_jerk_mps3, max_a_long_mps2, max_a_lat_mps2, max_speed_kmh

    succ = []
    crash = []
    time_all: List[float] = []
    time_succ: List[float] = []
    jerk_all: List[float] = []
    jerk_succ: List[float] = []
    rmsjerk_all: List[float] = []

    max_a_long: List[float] = []
    max_a_lat: List[float] = []
    max_speed: List[float] = []

    for r in rows:
        b_succ = parse_bool(r.get("success"))
        b_crash = parse_bool(r.get("crashed"))
        if b_succ is not None:
            succ.append(b_succ)
        if b_crash is not None:
            crash.append(b_crash)

        t = parse_float(r.get("time_s"))
        if t is not None:
            time_all.append(t)
            if b_succ is True:
                time_succ.append(t)

        mj = parse_float(r.get("max_jerk_mps3"))
        if mj is not None:
            jerk_all.append(mj)
            if b_succ is True:
                jerk_succ.append(mj)

        rj = parse_float(r.get("rms_jerk_mps3"))
        if rj is not None:
            rmsjerk_all.append(rj)

        al = parse_float(r.get("max_a_long_mps2"))
        if al is not None:
            max_a_long.append(al)

        at = parse_float(r.get("max_a_lat_mps2"))
        if at is not None:
            max_a_lat.append(at)

        ms = parse_float(r.get("max_speed_kmh"))
        if ms is not None:
            max_speed.append(ms)

    n = len(rows)
    n_succ = sum(1 for x in succ if x) if succ else 0
    n_crash = sum(1 for x in crash if x) if crash else 0

    summary = {
        "episodes": n,
        "successes": n_succ,
        "crashes": n_crash,
        "success_rate": (n_succ / n) if n else None,
        "crash_rate": (n_crash / n) if n else None,

        "time_mean_all": safe_mean(time_all),
        "time_median_all": safe_median(time_all),
        "time_mean_success": safe_mean(time_succ),
        "time_median_success": safe_median(time_succ),
        "time_p90_success": quantile(time_succ, 0.90),

        "max_jerk_mean_all": safe_mean(jerk_all),
        "max_jerk_p90_success": quantile(jerk_succ, 0.90),
        "max_jerk_max_all": (max(jerk_all) if jerk_all else None),

        "rms_jerk_mean_all": safe_mean(rmsjerk_all),

        "max_a_long_mean": safe_mean(max_a_long),
        "max_a_lat_mean": safe_mean(max_a_lat),
        "max_speed_mean": safe_mean(max_speed),
    }
    return summary


def fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def gather_paths(files: List[str], directory: Optional[str], recursive: bool, pattern: str) -> List[str]:
    out: List[str] = []
    for f in files:
        out.extend(glob.glob(f))
    if directory:
        if recursive:
            out.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
        else:
            out.extend(glob.glob(os.path.join(directory, pattern)))
    # unique + stable
    out = sorted(set(out))
    out = [p for p in out if os.path.isfile(p) and p.lower().endswith(".csv")]
    return out


def write_summary_csv(out_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    keys = ["file"] + [k for k in rows[0].keys() if k != "file"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="CSV files or globs (e.g., results/*.csv)")
    ap.add_argument("--dir", dest="directory", default=None, help="Directory to scan for CSVs")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when using --dir")
    ap.add_argument("--glob", dest="pattern", default="*.csv", help="Glob pattern under --dir (default: *.csv)")
    ap.add_argument("--out", default="results/compare/summary.csv",
                help="Write summary CSV to this path (default: results/compare/summary.csv)")
    ap.add_argument("--sort", choices=["success", "time", "crash"], default="success",
                    help="Sort table by success rate (default), mean success time, or crash rate")
    args = ap.parse_args()

    paths = gather_paths(args.files, args.directory, args.recursive, args.pattern)
    if not paths:
        print("No CSV files found.")
        return

    summaries: List[Dict[str, Any]] = []
    for p in paths:
        rows = read_csv_rows(p)
        s = compute_summary(rows)
        s["file"] = p
        summaries.append(s)

    # sorting
    if args.sort == "success":
        summaries.sort(key=lambda r: (r["success_rate"] is not None, r["success_rate"]), reverse=True)
    elif args.sort == "crash":
        summaries.sort(key=lambda r: (r["crash_rate"] is not None, r["crash_rate"]), reverse=False)
    else:  # time
        # smaller is better
        summaries.sort(key=lambda r: (r["time_mean_success"] is None, r["time_mean_success"]))

    # print table
    header = [
        "file",
        "episodes", "succ", "crash",
        "succ_rate", "crash_rate",
        "t_mean_succ", "t_p90_succ",
        "jerk_p90_succ", "v_mean"
    ]
    print("\n=== Summary ===")
    print(" | ".join(header))
    print("-" * 140)
    for s in summaries:
        row = [
            os.path.basename(s["file"]),
            fmt(s["episodes"], 0),
            fmt(s["successes"], 0),
            fmt(s["crashes"], 0),
            fmt(s["success_rate"]),
            fmt(s["crash_rate"]),
            fmt(s["time_mean_success"]),
            fmt(s["time_p90_success"]),
            fmt(s["max_jerk_p90_success"]),
            fmt(s["max_speed_mean"]),
        ]
        print(" | ".join(row))

    write_summary_csv(args.out, summaries)
    print(f"\nWrote summary CSV: {args.out}")


if __name__ == "__main__":
    main()
