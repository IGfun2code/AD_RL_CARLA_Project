#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from unprotected_left_turn_env_touch_phase import CarlaPPOEnvTouchPhase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dest-radius-m", type=float, default=5.0)
    parser.add_argument("--checkpoint-dir", default="results/ppo_left_turn_touch")
    parser.add_argument("--log-dir", default="results/ppo_tb_left_turn_touch")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--draw-interval", type=int, default=10)
    parser.add_argument("--draw-lifetime-s", type=float, default=1.0)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--ego-half-width", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required. Install stable-baselines3, gymnasium, and torch."
        ) from exc

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = Monitor(
        CarlaPPOEnvTouchPhase(
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
        out_path = os.path.join(args.checkpoint_dir, "ppo_left_turn_touch_phase.zip")
        model.save(out_path)
        print(f"Saved PPO model to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
