"""
Benchmark Polychromic PPO vs PID Controller on CartLatAccel

Uses the standard evaluation metrics:
- lataccel_cost: MSE of actual vs target acceleration * 100  
- jerk_cost: MSE of jerk (rate of change of acceleration) * 100
- total_cost: (lataccel_cost * 50) + jerk_cost

Also evaluates:
- pass@k: success rate with k attempts
- perturbation robustness: degradation under state perturbations
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import gymnasium as gym
import gym_cartlataccel

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import ActorCritic
from controllers import PIDController
from evaluation.eval_cost import calculate_costs
from ppo_poly import PolychroPPO


def rollout_pid(env, controller, max_steps=500):
  """Rollout PID controller, return actions and targets for cost calculation."""
  state, _ = env.reset()
  controller.reset()
  
  actions = []
  target_lataccels = []
  rewards = []
  positions = []
  target_positions = []
  current_lataccel = 0.0
  
  for step in range(max_steps):
    pos, vel, target_pos = state.flatten()[:3]
    
    # Track position error (the actual task metric)
    positions.append(float(pos))
    target_positions.append(float(target_pos))
    
    # Target lataccel from position error
    pos_error = target_pos - pos
    target_lataccel = np.clip(pos_error * 10.0, -1.0, 1.0)
    
    # PID control
    action = controller.update(target_lataccel, current_lataccel, state, None)
    action = np.clip(action, -1.0, 1.0)
    
    next_state, reward, terminated, truncated, info = env.step(np.array([action]))
    
    actual_lataccel = info.get('noisy_action', action)
    if hasattr(actual_lataccel, '__iter__'):
      actual_lataccel = actual_lataccel[0]
    
    actions.append(float(actual_lataccel))
    target_lataccels.append(float(target_lataccel))
    rewards.append(float(np.sum(reward)))
    
    current_lataccel = actual_lataccel
    state = next_state
    
    if terminated or truncated:
      break
  
  return np.array(actions), np.array(target_lataccels), rewards, np.array(positions), np.array(target_positions)


def rollout_poly(env, model, max_steps=500, deterministic=True, device='cpu'):
  """Rollout Polychromic PPO model, return actions and targets for cost calculation."""
  state, _ = env.reset()
  
  actions = []
  target_lataccels = []
  rewards = []
  positions = []
  target_positions = []
  
  for step in range(max_steps):
    pos, vel, target_pos = state.flatten()[:3]
    
    # Track position error (the actual task metric)
    positions.append(float(pos))
    target_positions.append(float(target_pos))
    
    state_tensor = torch.FloatTensor(state).to(device)
    
    with torch.no_grad():
      action = model.actor.get_action(state_tensor, deterministic=deterministic)
    
    if hasattr(action, '__iter__'):
      action_val = float(action[0]) if len(action) > 0 else float(action)
    else:
      action_val = float(action)
    
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Extract target lataccel (position error based)
    target_lataccel = np.clip((target_pos - pos) * 10.0, -1.0, 1.0)
    
    actual_lataccel = info.get('noisy_action', action_val)
    if hasattr(actual_lataccel, '__iter__'):
      actual_lataccel = float(actual_lataccel[0])
    else:
      actual_lataccel = float(actual_lataccel)
    
    actions.append(actual_lataccel)
    target_lataccels.append(float(target_lataccel))
    rewards.append(float(np.sum(reward)))
    
    state = next_state
    
    if terminated or truncated:
      break
  
  return np.array(actions), np.array(target_lataccels), rewards, np.array(positions), np.array(target_positions)


def evaluate_pass_at_k(env, rollout_fn, k_values=[1, 5, 10, 20], n_configs=20, success_threshold=-50):
  """
  Evaluate pass@k: probability of at least one success in k attempts.
  Success = total reward > threshold
  """
  results = {k: [] for k in k_values}
  
  for config_idx in range(n_configs):
    env.reset(seed=config_idx * 100)
    
    # Run max(k) attempts
    episode_rewards = []
    for attempt in range(max(k_values)):
      _, _, rewards = rollout_fn(env, max_steps=200)
      episode_rewards.append(sum(rewards))
    
    # Check pass@k for each k
    for k in k_values:
      top_k = episode_rewards[:k]
      passed = any(r > success_threshold for r in top_k)
      results[k].append(1.0 if passed else 0.0)
  
  return {k: np.mean(results[k]) for k in k_values}


def evaluate_perturbation_robustness(env, rollout_fn, n_perturbs=10):
  """Evaluate robustness to initial state perturbations."""
  original_rewards = []
  perturbed_rewards = []
  
  for _ in range(n_perturbs):
    # Original
    _, _, rewards = rollout_fn(env, max_steps=200)
    original_rewards.append(sum(rewards))
    
    # Perturbed
    env.reset()
    env.state[:, 0] += np.random.uniform(-1, 1, size=env.state.shape[0])
    env.state[:, 1] += np.random.uniform(-0.5, 0.5, size=env.state.shape[0])
    _, _, rewards = rollout_fn(env, max_steps=200)
    perturbed_rewards.append(sum(rewards))
  
  orig_mean = np.mean(original_rewards)
  pert_mean = np.mean(perturbed_rewards)
  degradation = (orig_mean - pert_mean) / (abs(orig_mean) + 1e-8)
  
  return {
    'original_mean': orig_mean,
    'perturbed_mean': pert_mean,
    'degradation': degradation
  }


def train_poly_ppo(max_evals=50000, device='cpu'):
  """Train Polychromic PPO and return model."""
  print("\n" + "=" * 60)
  print("TRAINING POLYCHROMIC PPO")
  print("=" * 60)
  
  env = gym.make("CartLatAccel-v1", env_bs=1)
  model = ActorCritic(
    env.observation_space.shape[-1],
    {"pi": [64, 64], "vf": [64, 64]},  # Slightly larger for better learning
    env.action_space.shape[-1]
  )
  
  ppo = PolychroPPO(
    env, model,
    lr=3e-4,
    n_steps=100,
    epochs=4,
    n_set=4,
    n_vines=8,
    n_rollout_states=2,
    poly_window=5,
    ent_coeff=0.005,  # Lower entropy for more exploitation
    device=device
  )
  
  start = time.time()
  best_model, hist = ppo.train(max_evals)
  train_time = time.time() - start
  
  env.close()
  print(f"Training completed in {train_time:.2f}s")
  
  return ppo.model, hist


def main():
  parser = argparse.ArgumentParser(description="Benchmark Poly PPO vs PID")
  parser.add_argument("--max_evals", type=int, default=50000, help="Training steps for Poly PPO")
  parser.add_argument("--n_rollouts", type=int, default=10, help="Evaluation rollouts per method")
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--load_model", type=str, default=None, help="Load pre-trained Poly PPO model")
  parser.add_argument("--save_model", action="store_true", help="Save trained model")
  args = parser.parse_args()
  
  print("=" * 70)
  print("  POLYCHROMIC PPO vs PID BENCHMARK")
  print("  CartLatAccel Controller Comparison")
  print("=" * 70)
  
  # === Setup ===
  env = gym.make("CartLatAccel-v1", env_bs=1)
  pid_controller = PIDController(p=0.195, i=0.100, d=-0.053)
  
  # === Train or Load Poly PPO ===
  if args.load_model and os.path.exists(args.load_model):
    print(f"\nLoading pre-trained model from {args.load_model}")
    poly_model = ActorCritic(
      env.observation_space.shape[-1],
      {"pi": [64, 64], "vf": [64, 64]},
      env.action_space.shape[-1]
    )
    poly_model.actor = torch.load(args.load_model)
  else:
    poly_model, _ = train_poly_ppo(args.max_evals, args.device)
    
    if args.save_model:
      os.makedirs('out', exist_ok=True)
      torch.save(poly_model.actor, 'out/poly_ppo_benchmark.pt')
      print(f"Model saved to out/poly_ppo_benchmark.pt")
  
  # === Evaluation Functions ===
  def pid_rollout_wrapper(env, max_steps=500):
    actions, targets, rewards, positions, target_positions = rollout_pid(env, pid_controller, max_steps)
    return actions, targets, rewards
  
  def poly_rollout_wrapper(env, max_steps=500):
    actions, targets, rewards, positions, target_positions = rollout_poly(env, poly_model, max_steps, deterministic=False, device=args.device)
    return actions, targets, rewards
  
  def poly_rollout_deterministic(env, max_steps=500):
    actions, targets, rewards, positions, target_positions = rollout_poly(env, poly_model, max_steps, deterministic=True, device=args.device)
    return actions, targets, rewards
  
  # === Evaluate Cost Metrics ===
  print("\n" + "=" * 60)
  print("COST METRICS EVALUATION")
  print("=" * 60)
  
  pid_costs = []
  poly_costs = []
  pid_rewards = []
  poly_rewards = []
  pid_pos_errors = []
  poly_pos_errors = []
  pid_action_jerks = []
  poly_action_jerks = []
  
  print(f"\nRunning {args.n_rollouts} rollouts per method...")
  
  for i in range(args.n_rollouts):
    # PID
    actions, targets, rewards, positions, target_positions = rollout_pid(env, pid_controller, max_steps=500)
    pid_controller.reset()
    costs = calculate_costs(actions, targets, dt=0.02)
    pid_costs.append(costs)
    pid_rewards.append(sum(rewards))
    
    # Position tracking error (MSE)
    pos_error_mse = np.mean((positions - target_positions) ** 2)
    pid_pos_errors.append(pos_error_mse)
    
    # Action jerk (smoothness)
    action_jerk = np.mean(np.diff(actions) ** 2)
    pid_action_jerks.append(action_jerk)
    
    # Poly PPO (deterministic for fair comparison)
    actions, targets, rewards, positions, target_positions = rollout_poly(env, poly_model, max_steps=500, deterministic=True, device=args.device)
    costs = calculate_costs(actions, targets, dt=0.02)
    poly_costs.append(costs)
    poly_rewards.append(sum(rewards))
    
    # Position tracking error (MSE)
    pos_error_mse = np.mean((positions - target_positions) ** 2)
    poly_pos_errors.append(pos_error_mse)
    
    # Action jerk (smoothness)
    action_jerk = np.mean(np.diff(actions) ** 2)
    poly_action_jerks.append(action_jerk)
  
  # Calculate averages
  pid_avg = {
    'lataccel_cost': np.mean([c['lataccel_cost'] for c in pid_costs]),
    'jerk_cost': np.mean([c['jerk_cost'] for c in pid_costs]),
    'total_cost': np.mean([c['total_cost'] for c in pid_costs]),
    'reward': np.mean(pid_rewards),
    'pos_error_mse': np.mean(pid_pos_errors),
    'action_jerk': np.mean(pid_action_jerks)
  }
  
  poly_avg = {
    'lataccel_cost': np.mean([c['lataccel_cost'] for c in poly_costs]),
    'jerk_cost': np.mean([c['jerk_cost'] for c in poly_costs]),
    'total_cost': np.mean([c['total_cost'] for c in poly_costs]),
    'reward': np.mean(poly_rewards),
    'pos_error_mse': np.mean(poly_pos_errors),
    'action_jerk': np.mean(poly_action_jerks)
  }
  
  # === Evaluate Pass@k ===
  print("\n" + "=" * 60)
  print("PASS@K EVALUATION (Diversity Test)")
  print("=" * 60)
  
  k_values = [1, 5, 10, 20]
  print("\nEvaluating PID pass@k...")
  pid_pass_k = evaluate_pass_at_k(env, pid_rollout_wrapper, k_values=k_values)
  
  print("Evaluating Poly PPO pass@k (stochastic)...")
  poly_pass_k = evaluate_pass_at_k(env, poly_rollout_wrapper, k_values=k_values)
  
  # === Evaluate Perturbation Robustness ===
  print("\n" + "=" * 60)
  print("PERTURBATION ROBUSTNESS EVALUATION")
  print("=" * 60)
  
  print("\nEvaluating PID robustness...")
  pid_robust = evaluate_perturbation_robustness(env, pid_rollout_wrapper)
  
  print("Evaluating Poly PPO robustness...")
  poly_robust = evaluate_perturbation_robustness(env, poly_rollout_deterministic)
  
  env.close()
  
  # === Print Results ===
  print("\n" + "=" * 70)
  print("  BENCHMARK RESULTS")
  print("=" * 70)
  
  print(f"\n{'Metric':<30} {'PID':<15} {'Poly PPO':<15} {'Ratio':<10} {'Winner':<10}")
  print("-" * 80)
  
  # Position tracking error (THE KEY METRIC - lower is better)
  p_val = pid_avg['pos_error_mse']
  poly_val = poly_avg['pos_error_mse']
  ratio = poly_val / (p_val + 1e-8)
  winner = 'PID' if p_val < poly_val else 'Poly PPO'
  print(f"{'*** POSITION ERROR (MSE) ***':<30} {p_val:<15.4f} {poly_val:<15.4f} {ratio:<10.2f}x {winner:<10}")
  
  # Action smoothness (lower is better)
  p_val = pid_avg['action_jerk']
  poly_val = poly_avg['action_jerk']
  ratio = poly_val / (p_val + 1e-8)
  winner = 'PID' if p_val < poly_val else 'Poly PPO'
  print(f"{'Action Jerk (smoothness)':<30} {p_val:<15.4f} {poly_val:<15.4f} {ratio:<10.2f}x {winner:<10}")
  
  # Environment reward (higher is better = closer to 0)
  p_val = pid_avg['reward']
  poly_val = poly_avg['reward']
  winner = 'PID' if p_val > poly_val else 'Poly PPO'
  print(f"{'Env Reward (higher=better)':<30} {p_val:<15.2f} {poly_val:<15.2f} {'':<10} {winner:<10}")
  
  print(f"\n{'Legacy Cost Metrics:':<50}")
  print("-" * 80)
  
  # Cost metrics (lower is better)
  for metric in ['lataccel_cost', 'jerk_cost', 'total_cost']:
    p_val = pid_avg[metric]
    poly_val = poly_avg[metric]
    ratio = poly_val / (p_val + 1e-8)
    winner = 'PID' if p_val < poly_val else 'Poly PPO'
    print(f"  {metric:<28} {p_val:<15.2f} {poly_val:<15.2f} {ratio:<10.2f}x {winner:<10}")
  
  # Pass@k (higher is better)
  print(f"\n{'Pass@k Results (higher = better):':<50}")
  print("-" * 80)
  for k in k_values:
    p_val = pid_pass_k[k]
    poly_val = poly_pass_k[k]
    winner = 'PID' if p_val > poly_val else ('Poly PPO' if poly_val > p_val else 'Tie')
    print(f"{'  pass@' + str(k):<30} {p_val:<15.1%} {poly_val:<15.1%} {'':<10} {winner:<10}")
  
  # Robustness (lower degradation is better)
  print(f"\n{'Perturbation Robustness:':<50}")
  print("-" * 80)
  print(f"{'  Original Reward':<30} {pid_robust['original_mean']:<15.2f} {poly_robust['original_mean']:<15.2f}")
  print(f"{'  Perturbed Reward':<30} {pid_robust['perturbed_mean']:<15.2f} {poly_robust['perturbed_mean']:<15.2f}")
  p_deg = pid_robust['degradation']
  poly_deg = poly_robust['degradation']
  winner = 'PID' if abs(p_deg) < abs(poly_deg) else 'Poly PPO'
  print(f"{'  Degradation':<30} {p_deg:<15.1%} {poly_deg:<15.1%} {'':<10} {winner:<10}")
  
  # Summary
  print("\n" + "=" * 70)
  print("  SUMMARY")
  print("=" * 70)
  
  pid_pos_err = pid_avg['pos_error_mse']
  poly_pos_err = poly_avg['pos_error_mse']
  pid_reward = pid_avg['reward']
  poly_reward = poly_avg['reward']
  
  print(f"""
  Position Tracking Performance (THE ACTUAL TASK):
    PID position MSE:      {pid_pos_err:.4f}
    Poly PPO position MSE: {poly_pos_err:.4f}
    Ratio (Poly/PID):      {poly_pos_err/(pid_pos_err+1e-8):.2f}x
  
  Environment Reward (higher = better):
    PID reward:            {pid_reward:.2f}
    Poly PPO reward:       {poly_reward:.2f}
    Winner:                {'PID' if pid_reward > poly_reward else 'Poly PPO'}
  
  Diversity (Pass@k - with multiple attempts):
    PID pass@5:            {pid_pass_k[5]:.1%}
    Poly PPO pass@5:       {poly_pass_k[5]:.1%}
    PID pass@20:           {pid_pass_k[20]:.1%}
    Poly PPO pass@20:      {poly_pass_k[20]:.1%}
    
  Robustness (degradation under perturbation):
    PID degradation:       {pid_robust['degradation']:.1%}
    Poly PPO degradation:  {poly_robust['degradation']:.1%}
  
  VERDICT:
  - Position Tracking: {'PID wins' if pid_pos_err < poly_pos_err else 'Poly PPO wins'} ({poly_pos_err/(pid_pos_err+1e-8):.2f}x ratio)
  - Env Reward: {'PID wins' if pid_reward > poly_reward else 'Poly PPO wins'}
  - Pass@5: {'PID wins' if pid_pass_k[5] > poly_pass_k[5] else ('Poly PPO wins' if poly_pass_k[5] > pid_pass_k[5] else 'Tie')}
  - Robustness: {'PID wins' if abs(pid_robust['degradation']) < abs(poly_robust['degradation']) else 'Poly PPO wins'}
  
  Note: Polychromic PPO is designed to maintain diversity for pass@k and
  robustness, not necessarily to beat hand-tuned PID on single-shot performance.
  """)


if __name__ == "__main__":
  main()

