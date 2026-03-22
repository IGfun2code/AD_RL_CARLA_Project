# Trajectory Visualization Update

This update adds CARLA debug drawing for the PPO-generated local trajectory.

## What gets drawn
- Green polyline: PPO local trajectory currently being tracked
- Orange point/arrow: current preview target waypoint sent to CARLA PID
- White point: first point in the generated trajectory
- Magenta points: approximate anchor locations along the generated local path
- Blue polyline (optional): global A* route backbone

## New CLI flags
Both `train_ppo.py` and `eval_ppo.py` now accept:
- `--draw-trajectory`
- `--draw-interval N`
- `--draw-lifetime-s T`
- `--draw-route`

## Typical Windows PowerShell commands

### 1) Start CARLA
```powershell
# From your CARLA install folder
.\CarlaUE4.exe -quality-level=Low -windowed -ResX=1280 -ResY=720
```

### 2) Activate your virtual environment
```powershell
cd C:\Users\ishan\OneDrive\Documents\AD_RL_CARLA_Project
.\.venv\Scripts\Activate.ps1
```

### 3) Train with trajectory visualization
```powershell
python .\train_ppo.py --scenario highway_merge --timesteps 50000 --draw-trajectory --draw-interval 10 --draw-lifetime-s 1.0 --draw-route
```

### 4) Quick smoke test with visualization
```powershell
python .\train_ppo.py --scenario highway_merge --timesteps 5000 --draw-trajectory --draw-interval 10 --draw-lifetime-s 1.0 --draw-route
```

### 5) Evaluate a trained model with visualization
```powershell
python .\eval_ppo.py --scenario highway_merge --model results/ppo/ppo_highway_merge.zip --episodes 5 --draw-trajectory --draw-interval 10 --draw-lifetime-s 1.0 --draw-route --out results/ppo/highway_merge_eval.csv
```

### 6) Left-turn versions
```powershell
python .\train_ppo.py --scenario left_turn --timesteps 50000 --draw-trajectory --draw-interval 10 --draw-lifetime-s 1.0 --draw-route
python .\eval_ppo.py --scenario left_turn --model results/ppo/ppo_left_turn.zip --episodes 5 --draw-trajectory --draw-interval 10 --draw-lifetime-s 1.0 --draw-route --out results/ppo/left_turn_eval.csv
```

## Notes
- Drawings appear inside the CARLA world using the debug renderer.
- If drawings flicker too quickly, increase `--draw-lifetime-s` to `1.5` or `2.0`.
- If drawing every 10 steps is too sparse, change `--draw-interval 5`.
- Visualization during training can slow simulation slightly.
