#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from unprotected_left_turn_env_touch_phase import CarlaPPOEnvTouchPhase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dest-radius-m", type=float, default=5.0)
    parser.add_argument("--out", default="results/ppo_left_turn_touch/eval.csv")
    parser.add_argument("--fail-dir", default="results/ppo_left_turn_touch_failures")
    parser.add_argument("--all-summary-out", default="")
    parser.add_argument("--metrics-out", default="")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--draw-interval", type=int, default=10)
    parser.add_argument("--draw-lifetime-s", type=float, default=1.0)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--ego-half-width", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(path: str) -> str:
    return Path(path).stem.replace(" ", "_")


def _fail_run_dir(base_dir: str, model_path: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{_safe_name(model_path)}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _default_all_summary_out(args) -> str:
    if args.all_summary_out:
        return args.all_summary_out
    out_path = Path(args.out)
    return str(out_path.with_name(f"{out_path.stem}_all_summary.csv"))


def _default_metrics_out(args) -> str:
    if args.metrics_out:
        return args.metrics_out
    out_path = Path(args.out)
    return str(out_path.with_name(f"{out_path.stem}_metrics.csv"))


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q = max(0.0, min(100.0, q))
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def build_eval_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    success_rows = [r for r in rows if bool(r.get("success", False))]
    crash_rows = [r for r in rows if bool(r.get("crashed", False))]
    timeout_rows = [r for r in rows if bool(r.get("timed_out", False))]

    success_times = sorted(float(r["episode_time_s"]) for r in success_rows)
    success_rewards = sorted(float(r["reward"]) for r in success_rows)
    final_distances = sorted(float(r["goal_distance"]) for r in rows)

    clear_successes = [r for r in success_rows if r.get("final_phase", "") == "CLEAR"]

    return {
        "episodes": total,
        "successes": len(success_rows),
        "crashes": len(crash_rows),
        "timeouts": len(timeout_rows),
        "success_rate": (len(success_rows) / total) if total else math.nan,
        "clear_success_rate": (len(clear_successes) / total) if total else math.nan,
        "p90_time_success_s": _percentile(success_times, 90.0),
        "p50_time_success_s": _percentile(success_times, 50.0),
        "mean_time_success_s": (sum(success_times) / len(success_times)) if success_times else math.nan,
        "p90_reward_success": _percentile(success_rewards, 90.0),
        "p50_final_goal_distance_m": _percentile(final_distances, 50.0),
    }


def classify_failure(trace_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trace_rows:
        return {"failure_type": "unknown", "notes": "empty trace"}

    final = trace_rows[-1]
    final_goal_distance = float(final.get("goal_distance", 0.0))
    low_speed_ratio = sum(float(r.get("speed_kmh", 0.0)) < 5.0 for r in trace_rows) / max(len(trace_rows), 1)
    in_clear_ratio = sum(r.get("phase", "") == "CLEAR" for r in trace_rows) / max(len(trace_rows), 1)
    near_conflict_ratio = sum(float(r.get("dist_to_conflict", 999.0)) < 3.0 for r in trace_rows) / max(len(trace_rows), 1)

    if bool(final.get("crashed", False)) and near_conflict_ratio > 0.2:
        failure_type = "collision_near_conflict_zone"
    elif bool(final.get("timed_out", False)) and in_clear_ratio < 0.1 and low_speed_ratio > 0.4:
        failure_type = "approach_hesitation_timeout"
    elif bool(final.get("timed_out", False)) and in_clear_ratio > 0.1:
        failure_type = "entered_but_failed_to_clear"
    elif final_goal_distance < 12.0 and bool(final.get("crashed", False)):
        failure_type = "late_collision_near_goal"
    elif low_speed_ratio > 0.6:
        failure_type = "persistent_low_speed_failure"
    else:
        failure_type = "mixed_failure"

    notes = (
        f"final_goal_distance={final_goal_distance:.3f}; "
        f"low_speed_ratio={low_speed_ratio:.3f}; "
        f"in_clear_ratio={in_clear_ratio:.3f}; "
        f"near_conflict_ratio={near_conflict_ratio:.3f}"
    )
    return {
        "failure_type": failure_type,
        "final_goal_distance": final_goal_distance,
        "low_speed_ratio": low_speed_ratio,
        "in_clear_ratio": in_clear_ratio,
        "near_conflict_ratio": near_conflict_ratio,
        "notes": notes,
    }


def main():
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 is required for PPO evaluation.") from exc

    env = CarlaPPOEnvTouchPhase(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        seed=args.seed,
        timeout_s=args.timeout_s,
        dest_radius_m=args.dest_radius_m,
        no_rendering=args.no_rendering,
        debug_draw=args.draw,
        debug_draw_interval_steps=args.draw_interval,
        debug_draw_lifetime_s=args.draw_lifetime_s,
        width_scale=args.width_scale,
        ego_half_width_m=args.ego_half_width,
    )
    model = PPO.load(args.model)

    rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []
    fail_run_dir: Optional[str] = None
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            terminated = False
            truncated = False
            last_info: Dict[str, Any] = dict(info)
            total_reward = 0.0
            step_rows: List[Dict[str, Any]] = []

            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                last_info = step_info
                step_rows.append(
                    {
                        "episode": ep,
                        "step": len(step_rows),
                        "reward": reward,
                        "phase": step_info.get("phase", ""),
                        "goal_distance": step_info.get("goal_distance", 0.0),
                        "speed_kmh": step_info.get("speed_kmh", 0.0),
                        "dist_to_conflict": step_info.get("dist_to_conflict", 0.0),
                        "phase_touch_idx": step_info.get("phase_touch_idx", -1),
                        "progress_reward": step_info.get("progress_reward", 0.0),
                        "comfort_penalty": step_info.get("comfort_penalty", 0.0),
                        "phase_bonus": step_info.get("phase_bonus", 0.0),
                        "phase_penalty": step_info.get("phase_penalty", 0.0),
                        "success": step_info.get("success", False),
                        "crashed": step_info.get("crashed", False),
                        "timed_out": step_info.get("timed_out", False),
                    }
                )

            steps = len(step_rows)
            episode_time_s = float(steps) * float(getattr(env, "dt", 0.05))
            row = {
                "episode": ep,
                "reward": total_reward,
                "success": last_info.get("success", False),
                "crashed": last_info.get("crashed", False),
                "timed_out": last_info.get("timed_out", False),
                "steps": steps,
                "episode_time_s": episode_time_s,
                "goal_distance": last_info.get("goal_distance", 0.0),
                "speed_kmh": last_info.get("speed_kmh", 0.0),
                "dist_to_conflict": last_info.get("dist_to_conflict", 0.0),
                "final_phase": last_info.get("phase", ""),
                "phase_touch_idx": last_info.get("phase_touch_idx", -1),
            }
            rows.append(row)

            if not row["success"]:
                if fail_run_dir is None:
                    fail_run_dir = _fail_run_dir(args.fail_dir, args.model)
                classification = classify_failure(step_rows)
                fail_rows.append({
                    "episode": ep,
                    **row,
                    **classification,
                })
                episode_dir = os.path.join(fail_run_dir, f"episode_{ep:03d}")
                os.makedirs(episode_dir, exist_ok=True)
                write_csv(os.path.join(episode_dir, "trace.csv"), step_rows)
                write_csv(os.path.join(episode_dir, "summary.csv"), [{**row, **classification}])

            print(
                f"[ep {ep}] success={rows[-1]['success']} crashed={rows[-1]['crashed']} "
                f"timed_out={rows[-1]['timed_out']} reward={total_reward:.2f} "
                f"time={episode_time_s:.2f}s final_phase={row['final_phase']}"
            )
    finally:
        env.close()

    write_csv(args.out, rows)
    print(f"Wrote per-episode summary: {args.out}")

    all_summary_out = _default_all_summary_out(args)
    write_csv(all_summary_out, rows)
    print(f"Wrote all-episode summary: {all_summary_out}")

    metrics_row = build_eval_metrics(rows)
    metrics_out = _default_metrics_out(args)
    write_csv(metrics_out, [metrics_row])
    print(f"Wrote aggregate metrics: {metrics_out}")
    print(
        "Aggregate: "
        f"episodes={metrics_row['episodes']} successes={metrics_row['successes']} "
        f"success_rate={metrics_row['success_rate']:.3f} "
        f"p90_time_success_s={metrics_row['p90_time_success_s']:.3f}"
    )

    if fail_run_dir is not None and fail_rows:
        write_csv(os.path.join(fail_run_dir, "failure_summary.csv"), fail_rows)
        print(f"Wrote failed episode traces under: {fail_run_dir}")


if __name__ == "__main__":
    main()
