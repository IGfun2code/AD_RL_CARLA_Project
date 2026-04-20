#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from unprotected_left_turn_env_touch_phase import CarlaPPOEnvTouchPhase


def main():
    parser = argparse.ArgumentParser(description="Simple test runner for touch-phase left-turn env.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--ego-half-width", type=float, default=1.0)
    args = parser.parse_args()

    env = CarlaPPOEnvTouchPhase(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        seed=args.seed,
        debug_draw=args.draw,
        debug_draw_interval_steps=1,
        debug_draw_lifetime_s=1.0,
        no_rendering=args.no_rendering,
        width_scale=args.width_scale,
        ego_half_width_m=args.ego_half_width,
    )

    obs, info = env.reset()
    print("reset info:", info)
    print("obs shape:", obs.shape)

    try:
        for step in range(args.steps):
            # simple conservative policy:
            # approach/decide: moderate speed, near-zero offsets
            # clear: higher speed, same offsets
            phase = env.phase_name
            if phase == "CLEAR":
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)
            else:
                action = np.array([0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if step % 20 == 0:
                print(
                    f"step={step:04d} phase={info['phase']:<16} "
                    f"speed={info['speed_kmh']:.1f}km/h "
                    f"dist_conflict={info['dist_to_conflict']:.2f} "
                    f"reward={reward:.3f}"
                )

            if terminated or truncated:
                print("episode end:", info)
                obs, info = env.reset()
                print("reset info:", info)

    finally:
        env.close()


if __name__ == "__main__":
    main()
