# RL Training Guide — Dynamic Trajectory PPO

## What changed
The PPO stack is no longer selecting from CARLA's pre-existing route waypoints.

New hierarchy:
- CARLA GlobalRoutePlanner computes a **global A\*** backbone from the ego pose to the destination once per episode.
- PPO parameterizes a **local trajectory segment** every simulator step.
- The local segment starts at the ego pose, bends relative to the global backbone using learned lateral offsets, then reconnects to the global route and continues to the destination.
- A PID-based low-level tracker follows the generated local trajectory.

So PPO is doing **reactive local path generation + speed selection** on top of a stable long-horizon route, not waypoint lookup and not full global replanning every tick.

## Design goal
The intended behavior is:
- keep the end goal and route backbone stable so the task remains destination-driven
- let PPO behave like a **local planner** that updates every step in response to traffic
- encourage PPO to make decisive merge and turn decisions in high-interaction areas instead of waiting for all traffic to clear
- allow global rerouting only as recovery if the ego drifts materially off-route

This means the local trajectory still updates every step, but the full A* route does not.

## PPO action space
The action vector now has 5 continuous values:

1. lateral offset for anchor point near 8 m ahead
2. lateral offset for anchor point near 18 m ahead
3. lateral offset for anchor point near 30 m ahead
4. heading bias at the reconnect point
5. target speed

These actions are converted into a smooth local trajectory using a spline, then the remaining global route is appended so the full generated path still terminates at the destination waypoint.

## Observation space
The observation now contains:
- ego speed
- goal distance
- route heading error
- signed lane offset relative to route
- local trajectory tracking error
- current steering
- last target speed
- average previous lateral offset
- previous heading bias
- nearby vehicle features for up to `obs_vehicle_count` neighbors

Each nearby vehicle contributes:
- relative x
- relative y
- relative vx
- relative vy
- relative distance
- vehicle length
- vehicle width
- bounding-box safety radius

That added size information is intended to help the agent avoid large vehicles like buses more safely.

## Reward structure
The reward now encourages:
- progress toward the destination
- staying comfortable and smooth
- keeping safe clearance using size-aware proximity penalties
- maintaining safe headway to forward conflicts
- reacting to low-TTC merge situations with safer speed choices
- staying near the driving lane / route corridor
- avoiding idling
- heavily penalizing early off-center crashes
- reaching the destination without collision or timeout

## Files to replace
Replace the repo's RL files with the versions in this folder:
- `carla_ppo_env.py`
- `eval_ppo.py`
- `README_RL.md`

`train_ppo.py` can remain unchanged.

## Training
Smoke test:

```powershell
python .\train_ppo.py --scenario highway_merge --timesteps 5000
```

Longer run:

```powershell
python .\train_ppo.py --scenario highway_merge --timesteps 50000
```

## Evaluation
```powershell
python .\eval_ppo.py --scenario highway_merge --model results/ppo/ppo_highway_merge.zip --episodes 20 --out results/ppo/highway_merge_eval.csv
```

The evaluation CSV now includes lane offset, target speed, and the number of points in the generated trajectory.

## Current Best Model
Current best saved dynamic PPO highway-merge checkpoint:
- `rl/ppo_dynamic_update/results/ppo/ppo_highway_merge.zip`

Current best recorded 100-episode eval for that checkpoint:
- `rl/ppo_dynamic_update/results/ppo/highway_merge_eval_100.csv`
- `77 / 100` successes on the latest reviewed run
