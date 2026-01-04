# gym-cartlataccel

RL learns to drive a (very simple) car in <1s!

CartLatAccel is a simple 1D controls environment for testing RL [driving controls](https://blog.comma.ai/rlcontrols/). Think CartPole but without the pole!

**Inputs:** `[pos, vel, target_pos]`  
**Action:** steering force/accel `[-1, 1]`  
**Episode:** 500 timesteps @ 50 FPS (10 seconds simulated)  
**Reward:** `-|pos - target_pos| / max_x` per step (normalized to ~[-1, 0])

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Run a controller

```bash
python tinycart.py --controller pid                    # Run PID
python tinycart.py --controller ppo --render           # Run PPO with visualization
python tinycart.py --controller pid --episodes 10      # Run 10 episodes, show avg
python tinycart.py --controller ppo --noise REALISTIC  # Add noise
```

### Train PPO

```bash
python train_ppo.py                        # Train and save to out/ppo.pt
python train_ppo.py --render               # Show eval after training
python train_ppo.py --max_evals 50000      # More training steps
```

### Programmatic

```python
import gymnasium as gym
import gym_cartlataccel

env = gym.make("CartLatAccel-v1")
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

## Structure

```
tinycart.py          # Runner - executes any controller on the env
train_ppo.py         # Training script for PPO
controllers/
  pid.py             # PID controller
  ppo.py             # PPO controller (loads trained model)
gym_cartlataccel/    # Gymnasium environment
```

## Results

PPO trains in <1s on GPU:

```
eps 30000.00, reward -4.67, t 0.67
Model saved to out/ppo.pt
Eval reward: -6.41
```

After training, use `--render` to see the pygame visualization. The black rectangle is the cart tracking the red target dot.

![cartlataccel](https://github.com/user-attachments/assets/7c9e5570-bb28-4276-9bda-c1ff84ce7448)

---

Version 1.1 (2024-11-12): fixed normalization for reward, action space so now (-1,1) per timestep. Added action clipping to -1,1

Input: [pos, vel, target_pos]
Output: action (steering force) [-1, 1]
Goal: minimize |pos - target_pos| (position tracking error)