#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

from rl.carla_ppo_env import CarlaPPOEnv


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
    return parser.parse_args()


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    )
    model = PPO.load(args.model)

    rows: List[Dict[str, Any]] = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            terminated = False
            truncated = False
            last_info: Dict[str, Any] = dict(info)
            total_reward = 0.0

            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                last_info = step_info

            episode_info = last_info.get("episode", {})
            rows.append(
                {
                    "episode": ep,
                    "scenario": args.scenario,
                    "reward": total_reward,
                    "success": episode_info.get("success", False),
                    "crashed": episode_info.get("crashed", False),
                    "timed_out": episode_info.get("timed_out", False),
                    "steps": episode_info.get("l", 0),
                    "goal_distance": last_info.get("goal_distance", 0.0),
                    "speed_kmh": last_info.get("speed_kmh", 0.0),
                }
            )
            print(
                f"[ep {ep}] success={rows[-1]['success']} crashed={rows[-1]['crashed']} "
                f"timed_out={rows[-1]['timed_out']} reward={total_reward:.2f}"
            )
    finally:
        env.close()

    write_csv(args.out, rows)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
