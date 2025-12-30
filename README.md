# gym-cartlataccel

RL learns to drive a (very simple) car in <1s!

CartLatAccel is a simple 1D controls environment with added realistic noise and trajectory following. This task is a very simple, vectorized cart dynamics environment for testing RL [driving controls](https://blog.comma.ai/rlcontrols/). Think CartPole but without the pole!

The only inputs are x, v, and x_target, and the action is the steering force/accel.

## 🎯 Quick Results

After extensive experimentation, **evolution-based methods outperform PPO training!**

| Method | Cost | vs PID | Winner |
|--------|------|--------|--------|
| **PID** | 2,081 | 1.00x | 🏆 Best Overall |
| **Evolved Neural (16x2)** | 2,457 | 1.18x | 🥈 Best Learned |
| **Evolved Analytical** | 2,541 | 1.22x | 🥉 |
| PPO (256x4, jerk-aware) | 2,731 | 1.31x | Best Deep RL |

**Key Insight:** Evolution (CMA-ES) beats gradient-based PPO for this task!

📖 **See [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) for complete details of all experiments.**

---

## Installation

```bash
# Clone repository
git clone https://github.com/ellenjxu/gym-cartlataccel
cd gym-cartlataccel

# Create environment
conda create -n cartlataccel python=3.11
conda activate cartlataccel

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Use Best Controller (Evolved Neural Network)

```python
import numpy as np
import gymnasium as gym
from models.jerk_env import CartLatAccelJerkEnv

# Load evolved controller (best learned method)
weights = np.load('models/evolved_neural_controller.npy')
# See training/evolve_neural.py for NeuralController class

# Create environment
env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
               jerk_penalty_weight=0.005,
               action_penalty_weight=0.0005)
```

### 2. Use PID (Best Overall)

```python
from controllers import PIDController

controller = PIDController(p=0.195, i=0.100, d=-0.053)
action = controller.update(target_lataccel, current_lataccel, state, None)
```

### 3. Train PPO (Optional)

```bash
# Train with jerk-aware rewards (required for stability)
python ppo.py --device cpu --max_evals 200000

# Use 256x4 network (optimal size)
# Note: Larger networks overfit!
```

---

## Repository Structure

```
cartlataccel/
├── EXPERIMENT_SUMMARY.md        # 📊 Complete overview of all experiments
├── ppo.py                       # Main PPO training
├── controllers.py               # Controller classes
│
├── models/                      # Model definitions & weights
│   ├── model.py                 # Actor-critic architecture
│   ├── jerk_env.py              # Jerk-aware environment
│   └── evolved_neural_controller.npy  # Best learned controller
│
├── training/                    # Training scripts
│   ├── evolve_neural.py         # ⭐ Evolve neural network (BEST!)
│   ├── evolve_controller.py     # Evolve analytical controller
│   ├── train_jerk_aware.py      # PPO with jerk penalties
│   └── train_large.py           # Train large networks
│
├── evaluation/                  # Evaluation & comparison
│   ├── eval_cost.py             # Cost calculation
│   ├── eval_pid.py              # Evaluate PID
│   └── eval_ppo.py              # Evaluate PPO
│
├── scripts/                     # Analysis tools
├── docs/                        # Detailed documentation
└── logs/                        # Training logs
```

---

## Evaluation

### Run Evaluations

```bash
# PID baseline
python evaluation/eval_pid.py --n_rollouts 10

# Evolved neural network
python evaluation/eval_ppo.py --model models/evolved_neural_controller.npy

# Compare methods
python evaluation/compare_results.py
```

### Cost Metrics

- `lataccel_cost`: Position tracking error
- `jerk_cost`: Control smoothness (rate of change of acceleration)
- `total_cost`: `(lataccel_cost × 50) + jerk_cost`

Lower is better. Goal: minimize both tracking error and smoothness.

---

## Key Experiments

We conducted 9 major experiment phases:

1. **Baseline PPO** - Failed badly (72x worse than PID)
2. **Jerk-Aware Rewards** - Stabilized training (7x vs PID)
3. **Network Scaling** - Found optimal size: 256x4 (1.31x vs PID)
4. **Learning Rate Analysis** - Discovered overfitting in large networks
5. **Long Training** - Best at 200k steps, overfits after
6. **Evolution (Analytical)** - 7 parameters → 1.22x vs PID
7. **Evolution (Neural)** - 353 parameters → **1.18x vs PID** ⭐
8. **Bigger Evolved Networks** - Marginal gains, not worth the time

**See [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) for complete details.**

---

## Main Findings

### 1. Evolution Beats PPO
- Evolved neural: 1.18x vs PID
- Best PPO: 1.31x vs PID
- Optimization method matters more than model size!

### 2. Bigger Networks Overfit in PPO
- 256x4: 2,731 (best)
- 512x4: 15,848 (5x worse!)
- More parameters ≠ better performance

### 3. PID is Hard to Beat
Classical control is still best for simple linear systems.

### 4. Reward Shaping is Critical
Position-error only → disaster. Must include jerk penalties.

---

## Documentation

- **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)** - Complete overview (READ THIS FIRST!)
- **[docs/FINAL_COMPARISON.md](docs/FINAL_COMPARISON.md)** - Method comparison
- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluation system
- **[docs/SCALING_RESULTS.md](docs/SCALING_RESULTS.md)** - Network scaling experiments
- **[docs/SCALING_DIAGNOSIS.md](docs/SCALING_DIAGNOSIS.md)** - Why large networks fail

---

## Citation

```
CartLatAccel PPO Training Experiments
December 2024
Evolution-based methods for learned control
```

---

## Version History

- **v1.2 (2024-12-29)**: Complete experiment suite, evolution methods
- **v1.1 (2024-11-12)**: Fixed normalization, action space (-1,1)
- **v1.0**: Initial release

---

**Bottom Line:** Evolution (1.18x) > PPO (1.31x), but PID (1.00x) still wins! 🏆
