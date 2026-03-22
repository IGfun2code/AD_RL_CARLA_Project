# RL Training Guide

## Environment
Use the repo-local Python 3.12 environment:

```powershell
cd 'C:\Users\Leo Shaw\Documents\VSCode\VS_CMU_S26\MLAI Project\AD_RL_CARLA_Project'
& '.\.venv312\Scripts\Activate.ps1'
$env:CARLA_ROOT='C:\Users\Leo Shaw\Documents\VSCode\VS_CMU_S26\MLAI Project'
```

This environment is expected to contain:
- `carla==0.9.16` from `PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl`
- `numpy`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `shapely`
- `tqdm`
- `rich`

## Start CARLA
From `C:\Users\Leo Shaw\Documents\VSCode\VS_CMU_S26\MLAI Project`:

```powershell
& '.\CarlaUE4.exe' -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=2001
```

Leave that terminal open while training or evaluation is running.

## Baseline Runs
Left turn:

```powershell
python .\scene_runner_baseline.py --scenario left_turn --hide-layers --freeze-lights --traffic-stream --agent behavior --behavior aggressive --episodes 10 --out results/left_turn/baseline_behavior_aggressive.csv
```

Highway merge:

```powershell
python .\scene_runner_baseline.py --scenario highway_merge --hide-layers --freeze-lights --traffic-stream --agent behavior --behavior normal --episodes 10 --out results/highway_merge/baseline_behavior_normal.csv
```

## PPO Training
The current PPO stack is hierarchical:
- CARLA GlobalRoutePlanner builds an A*-based route
- the route is replanned from the ego pose every timestep
- PPO chooses waypoint lookahead and target speed
- CARLA PID control executes the low-level vehicle motion

Short smoke test:

```powershell
python .\train_ppo.py --scenario highway_merge --timesteps 5000
```

Longer run:

```powershell
python .\train_ppo.py --scenario highway_merge --timesteps 50000
```

Artifacts:
- model checkpoints go to `results/ppo/`
- tensorboard logs go to `results/ppo_tb/`

## PPO Evaluation
Highway merge:

```powershell
python .\eval_ppo.py --scenario highway_merge --model results/ppo/ppo_highway_merge.zip --episodes 20 --out results/ppo/highway_merge_eval.csv
```

Left turn:

```powershell
python .\eval_ppo.py --scenario left_turn --model results/ppo/ppo_left_turn.zip --episodes 20 --out results/ppo/left_turn_eval.csv
```

## Notes
- PPO weights update in batches, not every simulator timestep.
- The target route is replanned every timestep.
- The final waypoint in the route is forced to be the destination waypoint.
- If `agents` imports fail, confirm `CARLA_ROOT` points to the CARLA install root.
- If CARLA connection fails, check port `2000` with `netstat -ano | findstr :2000`.
