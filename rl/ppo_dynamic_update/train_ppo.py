#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from carla_ppo_env import CarlaPPOEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["left_turn", "highway_merge"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dest-radius-m", type=float, default=5.0)
    parser.add_argument("--checkpoint-dir", default="results/ppo")
    parser.add_argument("--log-dir", default="results/ppo_tb")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--draw-trajectory", action="store_true")
    parser.add_argument("--draw-interval", type=int, default=10)
    parser.add_argument("--draw-lifetime-s", type=float, default=1.0)
    parser.add_argument("--draw-route", action="store_true")
    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument('--suffix', type=str, default= '')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "stable-baselines3 is required for PPO training. Install stable-baselines3, gymnasium, and torch."
        ) from exc

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = Monitor(
        CarlaPPOEnv(
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
            obs_noise_std=args.obs_noise_std,
            action_noise_std=args.action_noise_std,
        )
    )

    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.log_dir,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            seed=args.seed,
            device=args.device,
        )

        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #can use the time in the file name if wanted
        out_path = os.path.join(args.checkpoint_dir, f"ppo_{args.scenario}_{args.suffix}.zip")
        model.save(out_path)
        print(f"Saved PPO model to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
