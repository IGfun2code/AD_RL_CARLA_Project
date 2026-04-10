# Touch-Phase Left Turn RL (CARLA)

## Motivation
This project studies unprotected left-turn decision making using reinforcement learning.
We model the interaction between ego vehicle and oncoming traffic using a **conflict zone** representation.

Key idea:
- Represent the intersection using a conflict zone (oncoming lane region)
- Split ego trajectory into phases:
  - APPROACH / DECIDE
  - CLEAR
- Train PPO to learn when to wait vs commit

---

## File Structure

### Environment
- unprotected_left_turn_env_touch_phase.py  
Implements:
- conflict zone extraction
- touch-based phase split
- observation + reward design
- PPO-compatible gym environment

### Test Script
- test_touch_phase_env.py  
Used to verify environment behavior and visualization.

### Training Script
- train_ppo_touch_phase.py  
Used to train PPO policy.

### Evaluation Script
- eval_ppo_touch_phase_p90_all_summary.py  
Used to evaluate trained models and compute metrics.

---

## Running Instructions

### 1. Start CARLA
Run CARLA simulator first.

---

### 2. Test Environment

python test_touch_phase_env.py --steps 500 --draw

---

### 3. Train Model

python train_ppo_touch_phase.py --timesteps 100000 --draw --draw-interval 10 --draw-lifetime-s 1.0

---

### 4. Evaluate Model

python eval_ppo_touch_phase_p90_all_summary.py --model results/ppo_left_turn_touch/ppo_left_turn_touch_phase.zip --episodes 100 --draw

---

## Visualization

- Red: conflict zone
- Green: approach route
- Cyan: clear route
- Blue: PPO trajectory

---

## Notes

- Warmup is required to stabilize traffic flow before training.
- Visualization may slow down training.
- PPO may learn conservative waiting behavior if reward is unbalanced.
