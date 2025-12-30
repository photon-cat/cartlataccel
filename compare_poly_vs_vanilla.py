"""
Compare Polychromic PPO vs Vanilla PPO on CartLatAccel

Key metrics:
1. Average reward (standard success)
2. Pass@k (diversity-aware success)
3. Robustness to initial state perturbations
"""

import os
import time
import torch
import numpy as np
import gymnasium as gym
import gym_cartlataccel
from models.model import ActorCritic
from ppo import PPO
from ppo_poly import PolychroPPO


def evaluate_pass_at_k(env, model, k_values=[1, 5, 10, 20], n_configs=20, max_steps=100, device='cpu'):
  """
  Evaluate pass@k: probability that at least one of k attempts succeeds.
  
  For CartLatAccel, "success" = reward > threshold (e.g., > -20 per episode)
  """
  results = {k: [] for k in k_values}
  success_threshold = -20  # Episode is "successful" if reward > this
  
  for config_idx in range(n_configs):
    # Reset with different seed for each config
    env.reset(seed=config_idx * 100)
    
    # Run k attempts
    rewards = []
    for attempt in range(max(k_values)):
      env.reset()
      total_reward = 0
      state = env.state
      
      for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
          action = model.get_action(state_tensor, deterministic=False)  # Stochastic for diversity
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward.item() if hasattr(reward, 'item') else np.sum(reward)
        state = next_state
        if terminated or truncated:
          break
      
      rewards.append(total_reward)
    
    # Check pass@k for each k
    for k in k_values:
      top_k_rewards = rewards[:k]
      passed = any(r > success_threshold for r in top_k_rewards)
      results[k].append(1.0 if passed else 0.0)
  
  # Compute pass rates
  pass_rates = {k: np.mean(results[k]) for k in k_values}
  return pass_rates


def evaluate_perturbation_robustness(env, model, n_perturbs=10, max_steps=100, device='cpu'):
  """
  Evaluate robustness to initial state perturbations.
  
  Perturbs the initial position/velocity and measures success rate.
  """
  original_rewards = []
  perturbed_rewards = []
  
  for _ in range(n_perturbs):
    # Original starting state
    env.reset()
    state = env.state.copy()
    
    total_reward = 0
    for step in range(max_steps):
      state_tensor = torch.FloatTensor(state).to(device)
      with torch.no_grad():
        action = model.get_action(state_tensor, deterministic=True)
      next_state, reward, terminated, truncated, _ = env.step(action)
      total_reward += reward.item() if hasattr(reward, 'item') else np.sum(reward)
      state = next_state
      if terminated or truncated:
        break
    original_rewards.append(total_reward)
    
    # Perturbed starting state
    env.reset()
    # Add perturbation to position and velocity
    env.state[:, 0] += np.random.uniform(-1, 1, size=env.state.shape[0])  # Position perturbation
    env.state[:, 1] += np.random.uniform(-0.5, 0.5, size=env.state.shape[0])  # Velocity perturbation
    state = env.state.copy()
    
    total_reward = 0
    for step in range(max_steps):
      state_tensor = torch.FloatTensor(state).to(device)
      with torch.no_grad():
        action = model.get_action(state_tensor, deterministic=True)
      next_state, reward, terminated, truncated, _ = env.step(action)
      total_reward += reward.item() if hasattr(reward, 'item') else np.sum(reward)
      state = next_state
      if terminated or truncated:
        break
    perturbed_rewards.append(total_reward)
  
  return {
    'original_mean': np.mean(original_rewards),
    'perturbed_mean': np.mean(perturbed_rewards),
    'degradation': (np.mean(original_rewards) - np.mean(perturbed_rewards)) / abs(np.mean(original_rewards) + 1e-8)
  }


def train_and_compare(max_evals=30000, device='cpu'):
  """Train both methods and compare."""
  
  print("=" * 60)
  print("POLYCHROMIC PPO vs VANILLA PPO COMPARISON")
  print("=" * 60)
  
  # === Train Vanilla PPO ===
  print("\n[1/2] Training Vanilla PPO...")
  env_vanilla = gym.make("CartLatAccel-v1", env_bs=1000)
  model_vanilla = ActorCritic(
    env_vanilla.observation_space.shape[-1],
    {"pi": [32], "vf": [32]},
    env_vanilla.action_space.shape[-1]
  )
  
  ppo_vanilla = PPO(env_vanilla, model_vanilla, env_bs=1000, device=device)
  start = time.time()
  vanilla_model, vanilla_hist = ppo_vanilla.train(max_evals)
  vanilla_time = time.time() - start
  env_vanilla.close()
  
  # === Train Polychromic PPO ===
  print("\n[2/2] Training Polychromic PPO...")
  env_poly = gym.make("CartLatAccel-v1", env_bs=1)
  model_poly = ActorCritic(
    env_poly.observation_space.shape[-1],
    {"pi": [32], "vf": [32]},
    env_poly.action_space.shape[-1]
  )
  
  ppo_poly = PolychroPPO(
    env_poly, model_poly,
    n_steps=100,
    n_set=4,
    n_vines=8,
    n_rollout_states=2,
    device=device
  )
  start = time.time()
  poly_model, poly_hist = ppo_poly.train(max_evals)
  poly_time = time.time() - start
  env_poly.close()
  
  # === Evaluate Both ===
  print("\n" + "=" * 60)
  print("EVALUATION")
  print("=" * 60)
  
  eval_env = gym.make("CartLatAccel-v1", env_bs=1)
  
  # Standard rewards
  print("\n[Evaluating Standard Performance...]")
  
  vanilla_rewards = []
  for _ in range(10):
    eval_env.reset()
    states, actions, rewards, dones, _ = PPO.rollout(eval_env, vanilla_model, max_steps=100, deterministic=True, device=device)
    vanilla_rewards.append(sum(rewards))
  
  poly_rewards = []
  for _ in range(10):
    traj = ppo_poly.rollout_single(eval_env, ppo_poly.model, max_steps=100, deterministic=True)
    poly_rewards.append(sum(traj['rewards']))
  
  # Pass@k
  print("\n[Evaluating Pass@k...]")
  k_values = [1, 5, 10, 20]
  vanilla_pass_k = evaluate_pass_at_k(eval_env, vanilla_model, k_values=k_values, device=device)
  poly_pass_k = evaluate_pass_at_k(eval_env, poly_model, k_values=k_values, device=device)
  
  # Perturbation robustness  
  print("\n[Evaluating Perturbation Robustness...]")
  vanilla_robust = evaluate_perturbation_robustness(eval_env, vanilla_model, device=device)
  poly_robust = evaluate_perturbation_robustness(eval_env, poly_model, device=device)
  
  eval_env.close()
  
  # === Print Results ===
  print("\n" + "=" * 60)
  print("RESULTS")
  print("=" * 60)
  
  print(f"\n{'Metric':<30} {'Vanilla PPO':<15} {'Poly PPO':<15} {'Winner':<10}")
  print("-" * 70)
  
  # Training time
  print(f"{'Training Time (s)':<30} {vanilla_time:<15.2f} {poly_time:<15.2f} {'Vanilla' if vanilla_time < poly_time else 'Poly':<10}")
  
  # Average reward
  v_mean = np.mean(vanilla_rewards)
  p_mean = np.mean(poly_rewards)
  print(f"{'Avg Reward (10 rollouts)':<30} {v_mean:<15.2f} {p_mean:<15.2f} {'Vanilla' if v_mean > p_mean else 'Poly':<10}")
  
  # Pass@k
  print(f"\nPass@k Results:")
  for k in k_values:
    v_pass = vanilla_pass_k[k]
    p_pass = poly_pass_k[k]
    winner = 'Vanilla' if v_pass > p_pass else ('Poly' if p_pass > v_pass else 'Tie')
    print(f"{'  Pass@' + str(k):<30} {v_pass:<15.2%} {p_pass:<15.2%} {winner:<10}")
  
  # Robustness
  print(f"\nPerturbation Robustness:")
  print(f"{'  Original Reward':<30} {vanilla_robust['original_mean']:<15.2f} {poly_robust['original_mean']:<15.2f}")
  print(f"{'  Perturbed Reward':<30} {vanilla_robust['perturbed_mean']:<15.2f} {poly_robust['perturbed_mean']:<15.2f}")
  v_deg = vanilla_robust['degradation']
  p_deg = poly_robust['degradation']
  print(f"{'  Degradation %':<30} {v_deg:<15.2%} {p_deg:<15.2%} {'Vanilla' if v_deg < p_deg else 'Poly':<10}")
  
  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)
  print("""
Polychromic PPO is designed to maintain DIVERSITY in generated trajectories.
This is most beneficial when:
1. You want pass@k performance (multiple attempts)
2. You need robustness to different initial conditions
3. You want to avoid mode collapse onto a single strategy

For simple control tasks like CartLatAccel, vanilla PPO may win on
average reward, but Polychromic PPO should show advantages on diversity
metrics.
""")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_evals", type=int, default=30000)
  parser.add_argument("--device", type=str, default="cpu")
  args = parser.parse_args()
  
  train_and_compare(max_evals=args.max_evals, device=args.device)

