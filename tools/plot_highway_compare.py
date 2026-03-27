#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def label_from_path(path: str) -> str:
    stem = Path(path).stem
    if "ppo" in stem.lower() or "eval" in stem.lower():
        return "RL PPO"
    name = stem.replace("behavior_", "").replace("_100", "")
    return name.replace("_", " ").title()


def plot_success_rate(rows: List[Dict[str, Any]], out_path: str) -> None:
    labels = [label_from_path(r["file"]) for r in rows]
    values = [100.0 * parse_float(r["success_rate"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4" if "RL" in label else "#ff7f0e" for label in labels]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Highway Scenario Success Rate")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_p90_lines(rows: List[Dict[str, Any]], out_path: str) -> None:
    labels = [label_from_path(r["file"]) for r in rows]
    x = list(range(len(rows)))
    p90_time = [parse_float(r["time_p90_success"]) for r in rows]
    p90_jerk = [parse_float(r["max_jerk_p90_success"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(x, p90_time, marker="o", linewidth=2.2, color="#1f77b4", label="P90 Completion Time")
    ax1.set_ylabel("P90 Completion Time (s)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, p90_jerk, marker="s", linewidth=2.2, color="#d62728", label="P90 Max Jerk")
    ax2.set_ylabel("P90 Max Jerk (m/s^3)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    for idx, val in enumerate(p90_time):
        ax1.annotate(f"{val:.1f}", (x[idx], val), textcoords="offset points", xytext=(0, 8), ha="center", color="#1f77b4")
    for idx, val in enumerate(p90_jerk):
        ax2.annotate(f"{val:.1f}", (x[idx], val), textcoords="offset points", xytext=(0, -14), ha="center", color="#d62728")

    ax1.set_title("Highway Scenario P90 Time and Jerk")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot highway RL vs baseline comparison charts.")
    parser.add_argument("csv_path", help="Comparison CSV from compare_highway_results.py")
    parser.add_argument("--out-dir", default="results/compare/plots")
    args = parser.parse_args()

    rows = read_rows(args.csv_path)
    os.makedirs(args.out_dir, exist_ok=True)

    success_path = os.path.join(args.out_dir, "highway_success_rate.png")
    p90_path = os.path.join(args.out_dir, "highway_p90_time_jerk.png")

    plot_success_rate(rows, success_path)
    plot_p90_lines(rows, p90_path)

    print(f"Wrote: {success_path}")
    print(f"Wrote: {p90_path}")


if __name__ == "__main__":
    main()
