# Training Benchmark Results

**Date:** 2024-12-28
**Run ID:** benchmark_20251228_165504
**Total Training Steps:** 100,000
**Evaluation Interval:** 10,000 steps
**Device:** CPU

---

## Executive Summary

The 100k step training run reveals a **critical problem**: PPO optimizes for the wrong objective.

### Key Finding
- ✅ **Reward improved** by 95.3% (from -475 to -22)
- ❌ **Total cost got WORSE** by 5,877% (from 16k to 959k)
- ❌ **Jerk exploded** by 126,000x (from 7.5 to 949k)

**The agent learns aggressive, jerky control that minimizes position error but creates terrible jerk.**

---

## Best Checkpoint

**10,000 steps** achieved the best total cost:
- Total cost: **6,763.83**
- Lataccel cost: 65.75
- Jerk cost: 3,476.57
- Reward: -7.185

However, even this is **3.3x worse than PID** (2,080.39).

---

## Comparison: PPO vs PID

| Controller | Total Cost | Lataccel Cost | Jerk Cost | Performance |
|-----------|-----------|---------------|-----------|-------------|
| **PID** | 2,080 | 19.07 | 1,127 | ✓ Best |
| **PPO (10k)** | 6,764 | 65.75 | 3,477 | 3.3x worse |
| **PPO (100k)** | 958,682 | 189.99 | 949,233 | 461x worse |

---

## Training Progression

```
Steps    Total Cost    Lataccel   Jerk Cost    Reward
    0        16,040      320.65          8     -475.2
10,000         6,764       65.75      3,477       -7.2  ← BEST
20,000       187,598      107.50    182,223      -29.1
30,000       747,236      163.53    739,060      -18.4
40,000       949,047      190.36    939,529      -31.1
50,000       950,147      186.51    940,821      -26.9
60,000       949,096      189.50    939,621      -27.9
70,000       956,942      194.40    947,222      -31.7
80,000       955,829      192.93    946,182      -26.9
90,000       953,212      189.15    943,755      -21.4
100,000      958,682      188.99    949,233      -22.4  ← WORST
```

---

## Problem Analysis

### Why PPO Fails

1. **Wrong Reward Function**: PPO trains on `reward = -position_error` only
2. **No Jerk Penalty**: Nothing discourages jerky, aggressive actions
3. **Misaligned Objectives**: Training optimizes position error, evaluation measures jerk
4. **Overfitting to Wrong Metric**: More training makes it worse

### Visualization of the Problem

```
Training:     Minimize position error ❌
              ↓
              Learns aggressive control
              ↓
Evaluation:   Jerk penalty dominates ❌
              ↓
              Total cost explodes
```

---

## Recommendations

### 1. Fix the Reward Function (Critical)

Current:
```python
reward = -error/self.max_x
```

Proposed:
```python
# Track previous action for jerk calculation
if hasattr(self, 'prev_action'):
    jerk = (action - self.prev_action) / self.tau
    jerk_penalty = 0.001 * jerk**2  # tunable weight
else:
    jerk_penalty = 0

pos_error = abs(new_x - new_x_target)
reward = -pos_error/self.max_x - jerk_penalty
self.prev_action = action
```

### 2. Hyperparameter Tuning

- Reduce entropy coefficient (0.01 → 0.001) for less exploration
- Increase training steps (100k → 500k) after fixing reward
- Try lower learning rate (0.1 → 0.01) for stability

### 3. Alternative Approaches

- **PID remains best** for this task (3.3x better)
- Consider SAC or TD3 for smoother continuous control
- Add action space constraints or filters

---

## Files Generated

```
logs/benchmark_20251228_165504/
├── benchmark_log.json      # Full results (all checkpoints)
├── summary.json            # Training summary with config
├── report.txt              # Text table of results
├── results.csv             # CSV for plotting
├── model_step_0000000.pt   # Initial model (untrained)
├── model_step_0010000.pt   # ← BEST checkpoint
├── model_step_0020000.pt
├── ...
└── model_step_0100000.pt   # Final model
```

---

## Usage

### Load Best Checkpoint
```bash
python eval_ppo.py --model_path logs/benchmark_20251228_165504/model_step_0010000.pt \
                   --n_rollouts 10 --device cpu
```

### Rerun Benchmark
```bash
python benchmark_training.py --max_evals 100000 \
                             --benchmark_interval 10000 \
                             --device cpu
```

### Analyze Results
```bash
python analyze_benchmark.py logs/benchmark_20251228_165504
```

---

## Conclusion

**PPO can learn, but it learns the wrong thing.** The 10k checkpoint is usable but still inferior to PID. To make PPO competitive:

1. **Must** add jerk penalty to reward function
2. Retrain with corrected objective
3. Expect 10x+ improvement in total cost

Until then, **use PID for production** (2,080 total cost vs 6,764 for PPO).

