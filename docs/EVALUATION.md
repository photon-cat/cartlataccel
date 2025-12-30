# Evaluation System Summary

## Files Created

1. **`eval_cost.py`** - Cost calculation functions
   - `calculate_costs()`: Computes lataccel_cost, jerk_cost, total_cost
   - `evaluate_rollout()`: Wrapper for evaluating rollouts

2. **`controllers.py`** - Controller implementations
   - `BaseController`: Base class for controllers
   - `PIDController`: PID controller with configurable gains (P, I, D)

3. **`eval_pid.py`** - PID evaluation script
   - Runs PID controller for N rollouts
   - Calculates cost metrics
   - Supports different noise modes and PID gains

4. **`eval_ppo.py`** - PPO evaluation script
   - Trains or loads PPO model
   - Evaluates with same cost metrics as PID
   - Fair comparison between controllers

5. **`compare_results.py`** - Results summary
   - Displays comparison between PID and PPO
   - Explains why PPO underperforms
   - Suggestions for improvement

6. **`run_pid_eval.sh`** - Batch evaluation script
   - Tests PID under various conditions
   - Different noise levels
   - Different PID gains

## Cost Metrics (as requested)

```python
lataccel_cost = sum((actual_lat_accel - target_lat_accel)^2) / steps * 100
jerk_cost = sum((diff(actual_lat_accel) / dt)^2) / (steps - 1) * 100
total_cost = (lataccel_cost * 50) + jerk_cost
```

## Usage Examples

### Evaluate PID Controller
```bash
# Basic evaluation
python eval_pid.py --n_rollouts 5

# With noise
python eval_pid.py --n_rollouts 5 --noise_mode REALISTIC

# Custom PID gains
python eval_pid.py --p 0.3 --i 0.05 --d -0.1 --n_rollouts 5

# With visualization
python eval_pid.py --render --n_rollouts 1
```

### Evaluate PPO Controller
```bash
# Train and evaluate
python eval_ppo.py --train_first --max_evals 10000 --n_rollouts 5 --device cpu

# Use saved model
python eval_ppo.py --model_path out/best.pt --n_rollouts 5 --device cpu
```

### Run comprehensive comparison
```bash
bash run_pid_eval.sh
```

## Results

### No Noise Condition

**PID Controller (P=0.195, I=0.1, D=-0.053)**
- lataccel_cost: 19.07
- jerk_cost: 1127.07
- total_cost: **2080.39** ✓ WINNER

**PPO Controller (10k training steps)**
- lataccel_cost: 60.05
- jerk_cost: 14817.81
- total_cost: 17820.30

**PID is 8.6x better!**

## Why PID Wins

1. **Smooth control** - 13.1x lower jerk
2. **Better tracking** - 3.1x better lataccel error
3. **Designed for the task** - PID explicitly targets tracking error
4. **No training needed** - Works immediately with good gains

## Why PPO Struggles

1. **Wrong objective** - Trained only on position error, not lataccel/jerk
2. **No smoothness penalty** - Nothing discourages jerky actions
3. **Misaligned metrics** - Evaluation uses different objective than training
4. **Short training** - May need more samples

## How to Improve PPO

To make PPO competitive, modify the reward function in `env_v1.py`:

```python
# Current (line 105-106)
error = abs(new_x - new_x_target)
reward = -error/self.max_x

# Better - include jerk penalty
pos_error = abs(new_x - new_x_target)
if hasattr(self, 'prev_action'):
    jerk = (action - self.prev_action) / self.tau
    jerk_penalty = 0.01 * jerk**2  # tunable weight
else:
    jerk_penalty = 0
reward = -pos_error/self.max_x - jerk_penalty
self.prev_action = action
```

Then retrain with more steps (30k+).

