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

from carla_ppo_env import CarlaPPOEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["left_turn", "highway_merge"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dest-radius-m", type=float, default=5.0)
    parser.add_argument("--out", default="results/ppo/eval.csv")
    parser.add_argument("--fail-dir", default="results/ppo_failures")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--draw-trajectory", action="store_true")
    parser.add_argument("--draw-interval", type=int, default=10)
    parser.add_argument("--draw-lifetime-s", type=float, default=1.0)
    parser.add_argument("--draw-route", action="store_true")
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


def _fail_run_dir(base_dir: str, scenario: str, model_path: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, scenario, f"{_safe_name(model_path)}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def vec_mag(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def speed_kmh(vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def accel_components(vehicle) -> Dict[str, float]:
    a = vehicle.get_acceleration()
    tf = vehicle.get_transform()
    f = tf.get_forward_vector()
    fn = vec_mag(f.x, f.y, f.z)
    fx, fy, fz = (f.x / fn, f.y / fn, f.z / fn) if fn > 1e-6 else (1.0, 0.0, 0.0)
    along = a.x * fx + a.y * fy + a.z * fz
    amag2 = a.x * a.x + a.y * a.y + a.z * a.z
    lat2 = max(amag2 - along * along, 0.0)
    lat = math.sqrt(lat2)
    return {"a_long": along, "a_lat": lat, "a_mag": math.sqrt(amag2)}


def classify_failure(trace_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trace_rows:
        return {"failure_type": "unknown", "notes": "empty trace"}

    final = trace_rows[-1]
    max_lane_offset = max(float(r["lane_offset_m"]) for r in trace_rows)
    final_goal_distance = float(final["goal_distance"])
    low_speed_ratio = sum(float(r["speed_kmh"]) < 10.0 for r in trace_rows) / max(len(trace_rows), 1)
    high_traffic_ratio = sum(float(r["traffic_penalty"]) > 0.25 for r in trace_rows) / max(len(trace_rows), 1)

    if max_lane_offset >= 1.0 and final_goal_distance >= 30.0:
        failure_type = "lane_departure_early_collision"
    elif final_goal_distance < 12.0 and max_lane_offset >= 0.75:
        failure_type = "terminal_oscillation_collision"
    elif final_goal_distance < 12.0:
        failure_type = "late_collision"
    elif low_speed_ratio >= 0.35:
        failure_type = "hesitation_then_collision"
    elif high_traffic_ratio >= 0.15:
        failure_type = "closing_speed_or_gap_failure"
    else:
        failure_type = "mixed_collision"

    notes = (
        f"max_lane_offset={max_lane_offset:.3f}; "
        f"final_goal_distance={final_goal_distance:.3f}; "
        f"low_speed_ratio={low_speed_ratio:.3f}; "
        f"high_traffic_ratio={high_traffic_ratio:.3f}"
    )
    return {
        "failure_type": failure_type,
        "max_lane_offset": max_lane_offset,
        "final_goal_distance": final_goal_distance,
        "low_speed_ratio": low_speed_ratio,
        "high_traffic_ratio": high_traffic_ratio,
        "notes": notes,
    }


def main():
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 is required for PPO evaluation.") from exc

    env = CarlaPPOEnv(
        scenario_name=args.scenario,
        host=args.host,
        port=args.port,
        seed=args.seed,
        timeout_s=args.timeout_s,
        dest_radius_m=args.dest_radius_m,
        no_rendering=args.no_rendering,
        debug_draw_trajectory=args.draw_trajectory,
        debug_draw_interval_steps=args.draw_interval,
        debug_draw_lifetime_s=args.draw_lifetime_s,
        debug_draw_route=args.draw_route,
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
            start_t = env.world.get_snapshot().timestamp.elapsed_seconds
            end_t = start_t
            max_speed = 0.0
            max_a_long = 0.0
            max_a_lat = 0.0
            max_jerk = 0.0
            jerk_sq_sum = 0.0
            jerk_n = 0
            jerk_ignore_until = start_t + 0.5
            prev_a = None

            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                last_info = step_info
                snap = env.world.get_snapshot()
                sim_t = snap.timestamp.elapsed_seconds
                end_t = sim_t
                max_speed = max(max_speed, speed_kmh(env.handles.ego))
                ac = accel_components(env.handles.ego)
                max_a_long = max(max_a_long, ac["a_long"])
                max_a_lat = max(max_a_lat, ac["a_lat"])
                a = env.handles.ego.get_acceleration()
                if prev_a is not None and sim_t >= jerk_ignore_until:
                    dt_s = max(env.dt, 1e-3)
                    jx = (a.x - prev_a[0]) / dt_s
                    jy = (a.y - prev_a[1]) / dt_s
                    jz = (a.z - prev_a[2]) / dt_s
                    j = vec_mag(jx, jy, jz)
                    max_jerk = max(max_jerk, j)
                    jerk_sq_sum += j * j
                    jerk_n += 1
                prev_a = (a.x, a.y, a.z)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": len(step_rows),
                        "reward": reward,
                        "goal_distance": step_info.get("goal_distance", 0.0),
                        "speed_kmh": step_info.get("speed_kmh", 0.0),
                        "lane_offset_m": step_info.get("lane_offset_m", 0.0),
                        "target_speed_kmh": step_info.get("target_speed_kmh", 0.0),
                        "trajectory_points": step_info.get("trajectory_points", 0),
                        "progress_reward": step_info.get("progress_reward", 0.0),
                        "comfort_penalty": step_info.get("comfort_penalty", 0.0),
                        "traffic_penalty": step_info.get("traffic_penalty", 0.0),
                        "path_overlap_penalty": step_info.get("path_overlap_penalty", 0.0),
                        "early_lane_crash_penalty": step_info.get("early_lane_crash_penalty", 0.0),
                        "headway_penalty": step_info.get("headway_penalty", 0.0),
                        "lane_penalty": step_info.get("lane_penalty", 0.0),
                        "idle_penalty": step_info.get("idle_penalty", 0.0),
                        "anchor_shape_penalty": step_info.get("anchor_shape_penalty", 0.0),
                        "action_delta_penalty": step_info.get("action_delta_penalty", 0.0),
                        "ttc_action_reward": step_info.get("ttc_action_reward", 0.0),
                        "forward_headway_m": step_info.get("forward_headway_m", 0.0),
                        "forward_ttc_s": step_info.get("forward_ttc_s", 0.0),
                        "invalid_creep_penalty": step_info.get("invalid_creep_penalty", 0.0),
                        "unsafe_commit_penalty": step_info.get("unsafe_commit_penalty", 0.0),
                        "lane_end_urgency_penalty": step_info.get("lane_end_urgency_penalty", 0.0),
                        "longitudinal_smooth_penalty": step_info.get("longitudinal_smooth_penalty", 0.0),
                        "wait_reward": step_info.get("wait_reward", 0.0),
                        "build_speed_reward": step_info.get("build_speed_reward", 0.0),
                        "commit_reward": step_info.get("commit_reward", 0.0),
                        "lead_gap_m": step_info.get("lead_gap_m", 0.0),
                        "lead_ttc_s": step_info.get("lead_ttc_s", 0.0),
                        "rear_gap_m": step_info.get("rear_gap_m", 0.0),
                        "rear_ttc_s": step_info.get("rear_ttc_s", 0.0),
                        "merge_time_current_s": step_info.get("merge_time_current_s", 0.0),
                        "merge_time_target_s": step_info.get("merge_time_target_s", 0.0),
                        "path_valid": step_info.get("path_valid", True),
                        "justified_waiting": step_info.get("justified_waiting", False),
                        "justified_wait_steps": step_info.get("justified_wait_steps", 0),
                        "success": step_info.get("success", False),
                        "crashed": step_info.get("crashed", False),
                        "timed_out": step_info.get("timed_out", False),
                        "stuck": step_info.get("stuck", False),
                    }
                )

            episode_info = last_info.get("episode", {})
            rms_jerk = math.sqrt(jerk_sq_sum / jerk_n) if jerk_n > 0 else 0.0
            row = {
                "episode": ep,
                "scenario": args.scenario,
                "reward": total_reward,
                "success": episode_info.get("success", False),
                "crashed": episode_info.get("crashed", False),
                "timed_out": episode_info.get("timed_out", False),
                "stuck": episode_info.get("stuck", False),
                "steps": episode_info.get("l", 0),
                "time_s": (end_t - start_t),
                "goal_distance": last_info.get("goal_distance", 0.0),
                "speed_kmh": last_info.get("speed_kmh", 0.0),
                "lane_offset_m": last_info.get("lane_offset_m", 0.0),
                "target_speed_kmh": last_info.get("target_speed_kmh", 0.0),
                "trajectory_points": last_info.get("trajectory_points", 0),
                "max_speed_kmh": max_speed,
                "max_a_long_mps2": max_a_long,
                "max_a_lat_mps2": max_a_lat,
                "max_jerk_mps3": max_jerk,
                "rms_jerk_mps3": rms_jerk,
            }
            rows.append(row)
            if not row["success"]:
                if fail_run_dir is None:
                    fail_run_dir = _fail_run_dir(args.fail_dir, args.scenario, args.model)
                classification = classify_failure(step_rows)
                fail_rows.append(
                    {
                        "episode": ep,
                        "scenario": args.scenario,
                        "reward": total_reward,
                        "success": row["success"],
                        "crashed": row["crashed"],
                        "timed_out": row["timed_out"],
                        "stuck": row["stuck"],
                        "steps": row["steps"],
                        "goal_distance": row["goal_distance"],
                        "lane_offset_m": row["lane_offset_m"],
                        "target_speed_kmh": row["target_speed_kmh"],
                        **classification,
                    }
                )
                episode_dir = os.path.join(fail_run_dir, f"episode_{ep:03d}")
                os.makedirs(episode_dir, exist_ok=True)
                write_csv(os.path.join(episode_dir, "trace.csv"), step_rows)
                write_csv(os.path.join(episode_dir, "summary.csv"), [{**row, **classification}])
            print(
                f"[ep {ep}] success={rows[-1]['success']} crashed={rows[-1]['crashed']} "
                f"timed_out={rows[-1]['timed_out']} stuck={rows[-1]['stuck']} reward={total_reward:.2f}"
            )
    finally:
        env.close()

    write_csv(args.out, rows)
    print(f"Wrote: {args.out}")
    if fail_run_dir is not None and fail_rows:
        write_csv(os.path.join(fail_run_dir, "failure_summary.csv"), fail_rows)
        print(f"Wrote failed episode traces under: {fail_run_dir}")


if __name__ == "__main__":
    main()
