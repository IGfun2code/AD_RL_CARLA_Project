#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to PPO .zip model")
    parser.add_argument("--env-module", default="ppo_env_left")
    parser.add_argument("--env-class", default="CarlaPPOEnvTouchPhase")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dest-radius-m", type=float, default=5.0)
    parser.add_argument("--out", default="results/eval_aggr/eval_episode_metrics.csv")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--draw-interval", type=int, default=10)
    parser.add_argument("--draw-lifetime-s", type=float, default=1.0)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--ego-half-width", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--scenario-name", default="left_turn")
    parser.add_argument("--tag", default="")
    parser.add_argument("--commit-speed-kmh", type=float, default=3.0)
    return parser.parse_args()


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _get_env_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' has no class '{class_name}'")
    return getattr(module, class_name)


def _summary_path(args) -> str:
    if args.summary_out:
        return args.summary_out
    out_path = Path(args.out)
    return str(out_path.with_name(f"{out_path.stem}_summary.csv"))


def _mean(vals: List[float]) -> float:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else math.nan


def _quantile(vals: List[float], q: float) -> float:
    vals = sorted(float(v) for v in vals if v is not None and not math.isnan(float(v)))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def detect_commit(step_rows: List[Dict[str, Any]], speed_thres_kmh: float):
    run = 0
    for i, r in enumerate(step_rows):
        v = float(r.get("speed_kmh", math.nan))
        if not math.isnan(v) and v >= speed_thres_kmh:
            run += 1
            if run >= 3:
                j = i - 2
                return j, float(step_rows[j].get("sim_time_s", math.nan))
        else:
            run = 0
    return -1, math.nan


def episode_metrics(step_rows: List[Dict[str, Any]], dt_s: float, speed_thres_kmh: float) -> Dict[str, Any]:
    if not step_rows:
        return {
            "steps": 0,
            "completion_time_s": 0.0,
            "min_ttc": math.nan,
            "commit_step": -1,
            "commit_time_s": math.nan,
            "wait_time_before_commit_s": math.nan,
            "max_speed_kmh": math.nan,
            "mean_speed_kmh": math.nan,
            "max_accel_mps2": math.nan,
            "min_accel_mps2": math.nan,
            "mean_abs_accel_mps2": math.nan,
            "max_abs_jerk_mps3": math.nan,
            "p90_abs_jerk_mps3": math.nan,
            "mean_abs_jerk_mps3": math.nan,
        }

    speeds = [float(r.get("speed_kmh", math.nan)) for r in step_rows if not math.isnan(float(r.get("speed_kmh", math.nan)))]
    accels = [float(r.get("ego_accel_mps2", math.nan)) for r in step_rows if not math.isnan(float(r.get("ego_accel_mps2", math.nan)))]
    ttcs = [float(r.get("ttc", math.nan)) for r in step_rows if not math.isnan(float(r.get("ttc", math.nan))) and float(r.get("ttc", math.nan)) > 0]
    jerks = []
    for i in range(1, len(step_rows)):
        a0 = float(step_rows[i-1].get("ego_accel_mps2", math.nan))
        a1 = float(step_rows[i].get("ego_accel_mps2", math.nan))
        if not math.isnan(a0) and not math.isnan(a1):
            jerks.append((a1 - a0) / dt_s)
    abs_jerks = [abs(j) for j in jerks if not math.isnan(j)]
    commit_step, commit_time_s = detect_commit(step_rows, speed_thres_kmh)
    return {
        "steps": len(step_rows),
        "completion_time_s": len(step_rows) * dt_s,
        "min_ttc": min(ttcs) if ttcs else math.nan,
        "commit_step": commit_step,
        "commit_time_s": commit_time_s,
        "wait_time_before_commit_s": commit_time_s,
        "max_speed_kmh": max(speeds) if speeds else math.nan,
        "mean_speed_kmh": _mean(speeds),
        "max_accel_mps2": max(accels) if accels else math.nan,
        "min_accel_mps2": min(accels) if accels else math.nan,
        "mean_abs_accel_mps2": _mean([abs(a) for a in accels]) if accels else math.nan,
        "max_abs_jerk_mps3": max(abs_jerks) if abs_jerks else math.nan,
        "p90_abs_jerk_mps3": _quantile(abs_jerks, 0.9) if abs_jerks else math.nan,
        "mean_abs_jerk_mps3": _mean(abs_jerks) if abs_jerks else math.nan,
    }


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    success_rows = [r for r in rows if bool(r.get("success", False))]
    non_collision_rows = [r for r in rows if not bool(r.get("crashed", False))]
    return {
        "episodes": n,
        "success_rate": _mean([1.0 if r.get("success", False) else 0.0 for r in rows]),
        "collision_rate": _mean([1.0 if r.get("crashed", False) else 0.0 for r in rows]),
        "timeout_rate": _mean([1.0 if r.get("timed_out", False) else 0.0 for r in rows]),
        "avg_completion_time_s": _mean([r.get("completion_time_s", math.nan) for r in rows]),
        "median_completion_time_s": _quantile([r.get("completion_time_s", math.nan) for r in rows], 0.5),
        "p90_completion_time_s": _quantile([r.get("completion_time_s", math.nan) for r in rows], 0.9),
        "mean_min_ttc": _mean([r.get("min_ttc", math.nan) for r in rows]),
        "median_min_ttc": _quantile([r.get("min_ttc", math.nan) for r in rows], 0.5),
        "p10_min_ttc": _quantile([r.get("min_ttc", math.nan) for r in rows], 0.1),
        "avg_wait_time_before_commit_s": _mean([r.get("wait_time_before_commit_s", math.nan) for r in rows]),
        "median_wait_time_before_commit_s": _quantile([r.get("wait_time_before_commit_s", math.nan) for r in rows], 0.5),
        "non_collision_mean_p90_abs_jerk_mps3": _mean([r.get("p90_abs_jerk_mps3", math.nan) for r in non_collision_rows]),
        "non_collision_mean_max_abs_jerk_mps3": _mean([r.get("max_abs_jerk_mps3", math.nan) for r in non_collision_rows]),
        "non_collision_mean_mean_abs_jerk_mps3": _mean([r.get("mean_abs_jerk_mps3", math.nan) for r in non_collision_rows]),
        "success_only_avg_completion_time_s": _mean([r.get("completion_time_s", math.nan) for r in success_rows]),
    }


def main():
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for PPO evaluation.") from exc

    EnvClass = _get_env_class(args.env_module, args.env_class)
    env = EnvClass(
        scenario_name=args.scenario_name,
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

    dt_s = float(getattr(env, "dt", 0.05))
    rows: List[Dict[str, Any]] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            terminated = False
            truncated = False
            step_rows: List[Dict[str, Any]] = []
            last_info: Dict[str, Any] = dict(info)
            total_reward = 0.0
            prev_speed_mps: Optional[float] = None

            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += float(reward)
                last_info = step_info

                speed_kmh = float(step_info.get("speed_kmh", math.nan))
                speed_mps = speed_kmh / 3.6 if not math.isnan(speed_kmh) else math.nan
                ego_accel = math.nan
                if prev_speed_mps is not None and not math.isnan(speed_mps):
                    ego_accel = (speed_mps - prev_speed_mps) / dt_s
                if not math.isnan(speed_mps):
                    prev_speed_mps = speed_mps

                ttc = step_info.get("ttc", step_info.get("min_ttc", math.nan))
                sim_time_s = (len(step_rows) + 1) * dt_s

                row = {
                    "episode": ep,
                    "step": len(step_rows),
                    "sim_time_s": sim_time_s,
                    "reward": float(reward),
                    "phase": step_info.get("phase", ""),
                    "goal_distance": step_info.get("goal_distance", math.nan),
                    "speed_kmh": speed_kmh,
                    "dist_to_conflict": step_info.get("dist_to_conflict", math.nan),
                    "phase_touch_idx": step_info.get("phase_touch_idx", -1),
                    "progress_reward": step_info.get("progress_reward", math.nan),
                    "comfort_penalty": step_info.get("comfort_penalty", math.nan),
                    "phase_bonus": step_info.get("phase_bonus", math.nan),
                    "phase_penalty": step_info.get("phase_penalty", math.nan),
                    "success": step_info.get("success", False),
                    "crashed": step_info.get("crashed", False),
                    "timed_out": step_info.get("timed_out", False),
                    "ttc": ttc,
                    "ego_accel_mps2": ego_accel,
                }
                for k, v in step_info.items():
                    if k not in row:
                        row[k] = v
                step_rows.append(row)

            epm = episode_metrics(step_rows, dt_s, args.commit_speed_kmh)
            episode_row = {
                "episode": ep,
                "tag": args.tag,
                "env_module": args.env_module,
                "env_class": args.env_class,
                "model": args.model,
                "success": bool(last_info.get("success", False)),
                "crashed": bool(last_info.get("crashed", False)),
                "timed_out": bool(last_info.get("timed_out", False)),
                "reward": total_reward,
                "goal_distance": float(last_info.get("goal_distance", math.nan)),
                "final_phase": last_info.get("phase", ""),
            }
            episode_row.update(epm)
            rows.append(episode_row)

            print(
                f"[ep {ep}] success={episode_row['success']} crashed={episode_row['crashed']} "
                f"timed_out={episode_row['timed_out']} time={episode_row['completion_time_s']:.2f}s "
                f"min_ttc={episode_row['min_ttc']:.3f} wait={episode_row['wait_time_before_commit_s']:.2f}s "
                f"p90jerk={episode_row['p90_abs_jerk_mps3']:.3f}"
            )
    finally:
        env.close()

    write_csv(args.out, rows)
    summary = build_summary(rows)
    summary.update({
        "tag": args.tag,
        "env_module": args.env_module,
        "env_class": args.env_class,
        "model": args.model,
    })
    write_csv(_summary_path(args), [summary])

    print(f"Wrote per-episode metrics: {args.out}")
    print(f"Wrote summary metrics: {_summary_path(args)}")


if __name__ == "__main__":
    main()
