"""
Polychromic PPO for CartLatAccel

Based on: "Polychromic Objectives for Reinforcement Learning" (Stanford, 2025)

Key differences from vanilla PPO:
1. Set RL: Optimizes over sets of trajectories, not individual ones
2. Polychromic objective: f_poly = mean_reward × diversity
3. Vine sampling: Multiple trajectories spawned from rollout states
4. Modified advantage: Shared advantage for all trajectories in a set
"""

import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_cartlataccel
from collections import defaultdict
from models.model import ActorCritic


class PolychroPPO:
  """Polychromic PPO with vine sampling and set-level objectives."""
  
  def __init__(
    self,
    env,
    model,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_range=0.2,
    epochs=2,
    n_steps=100,
    ent_coeff=0.01,
    kl_coeff=0.01,
    vf_coeff=0.5,
    # Polychromic-specific params
    n_set=4,           # Set size for polychromic objective
    n_vines=8,         # Number of vines per rollout state  
    n_rollout_states=2,# Rollout states per trajectory
    poly_window=5,     # Window for polychromic advantage
    device='cuda',
    debug=False
  ):
    self.env = env
    self.model = model.to(device)
    self.gamma = gamma
    self.lam = lam
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.kl_coeff = kl_coeff
    self.vf_coeff = vf_coeff
    
    # Polychromic params
    self.n_set = n_set
    self.n_vines = n_vines
    self.n_rollout_states = n_rollout_states
    self.poly_window = poly_window
    
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.device = device
    self.debug = debug
    self.hist = []
    self.start = time.time()

  def compute_diversity(self, trajectories):
    """
    Compute diversity of a set of trajectories.
    
    For CartLatAccel: Use trajectory "signature" based on:
    1. Mean action (overall steering tendency)
    2. Action variance (aggressiveness)
    3. State region visited (position bins)
    
    Returns: float in [0, 1], where 0 = all identical, 1 = all unique
    """
    if len(trajectories) <= 1:
      return 0.0
    
    # Extract trajectory signatures
    signatures = []
    for traj in trajectories:
      # Flatten actions
      actions = []
      for a in traj['actions']:
        if hasattr(a, '__iter__'):
          actions.extend(np.array(a).flatten())
        else:
          actions.append(float(a))
      actions = np.array(actions)
      
      # Flatten states to get positions
      positions = []
      for s in traj['states']:
        s_flat = np.array(s).flatten()
        if len(s_flat) > 0:
          positions.append(s_flat[0])  # x position
      positions = np.array(positions) if positions else np.array([0])
      
      # Create signature: (mean_action_bin, action_var_bin, mean_pos_bin)
      mean_action = np.mean(actions) if len(actions) > 0 else 0
      action_var = np.var(actions) if len(actions) > 0 else 0
      mean_pos = np.mean(positions)
      
      # Coarse binning for meaningful diversity
      mean_action_bin = int(np.clip((mean_action + 1) * 2, 0, 3))  # 4 bins
      action_var_bin = int(np.clip(action_var * 10, 0, 3))  # 4 bins
      mean_pos_bin = int(np.clip((mean_pos + 3) / 1.5, 0, 3))  # 4 bins
      
      signatures.append((mean_action_bin, action_var_bin, mean_pos_bin))
    
    # Count unique signatures
    unique_sigs = len(set(signatures))
    diversity = unique_sigs / len(trajectories)
    
    return diversity

  def compute_polychromic_objective(self, trajectory_set):
    """
    Compute polychromic objective: f_poly = mean_reward × diversity
    
    Args:
      trajectory_set: list of trajectory dicts with 'rewards', 'actions', 'states'
    
    Returns:
      float: polychromic score
    """
    # Mean reward across trajectories (normalized to [0, 1])
    rewards = [sum(t['rewards']) for t in trajectory_set]
    mean_reward = np.mean(rewards)
    # Normalize: rewards are negative, so we shift and scale
    # Typical reward range is roughly [-10, 0] for this env
    norm_reward = (mean_reward + 10) / 10  # maps [-10, 0] -> [0, 1]
    norm_reward = np.clip(norm_reward, 0, 1)
    
    # Diversity
    diversity = self.compute_diversity(trajectory_set)
    
    # Polychromic objective
    f_poly = norm_reward * diversity
    
    return f_poly, norm_reward, diversity

  def rollout_single(self, env, model, max_steps, state=None, deterministic=False):
    """
    Perform a single rollout, optionally from a specific state.
    
    Returns dict with states, actions, rewards, dones, logprobs
    """
    trajectory = {
      'states': [],
      'actions': [],
      'rewards': [],
      'dones': [],
      'logprobs': [],
      'values': []
    }
    
    if state is not None:
      # Reset to specific state (for vine sampling)
      env.state = state.copy()
      env.curr_step = 0
      current_state = state
    else:
      current_state, _ = env.reset()
    
    for step in range(max_steps):
      state_tensor = torch.FloatTensor(current_state).to(self.device)
      
      with torch.no_grad():
        action = model.actor.get_action(state_tensor, deterministic=deterministic)
        value = model.critic(state_tensor.unsqueeze(0)).cpu().numpy().squeeze()
        model.actor.std = model.actor.log_std.exp()
        action_tensor = torch.FloatTensor(action).to(self.device)
        logprob = model.actor.get_logprob(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0)).cpu().numpy().squeeze()
      
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      
      trajectory['states'].append(current_state.copy())
      trajectory['actions'].append(action)
      trajectory['rewards'].append(reward)
      trajectory['dones'].append(done)
      trajectory['logprobs'].append(logprob)
      trajectory['values'].append(value)
      
      current_state = next_state
      if done:
        break
    
    trajectory['final_state'] = current_state
    return trajectory

  def vine_sampling(self, seed_trajectories):
    """
    Perform vine sampling: from selected rollout states, spawn additional trajectories.
    
    Args:
      seed_trajectories: list of initial trajectories
    
    Returns:
      rollout_data: dict mapping rollout_state_idx -> list of trajectory sets
    """
    rollout_data = defaultdict(list)
    
    for seed_traj in seed_trajectories:
      states = seed_traj['states']
      if len(states) < self.n_rollout_states + 1:
        continue
      
      # Select rollout states (evenly spaced)
      indices = np.linspace(0, len(states) - 1, self.n_rollout_states + 2, dtype=int)[1:-1]
      
      for idx in indices:
        rollout_state = np.array(states[idx])
        
        # Generate N vine trajectories from this state
        vine_trajectories = []
        for _ in range(self.n_vines):
          # Create a fresh env for vine rollout
          vine_env = gym.make("CartLatAccel-v1", env_bs=1)
          vine_env.reset()
          vine_env.state = rollout_state.reshape(1, -1)
          vine_env.curr_step = idx
          
          vine_traj = self.rollout_single(
            vine_env, 
            self.model, 
            max_steps=self.n_steps - idx,
            state=rollout_state.reshape(1, -1)
          )
          vine_trajectories.append(vine_traj)
          vine_env.close()
        
        # Store rollout state and its vines
        rollout_data[tuple(rollout_state.flatten())].append({
          'state_idx': idx,
          'state': rollout_state,
          'vines': vine_trajectories
        })
    
    return rollout_data

  def compute_gae(self, rewards, values, dones, next_value):
    """Compute Generalized Advantage Estimation."""
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
      if t == len(rewards) - 1:
        next_val = next_value
      else:
        next_val = values[t + 1]
      
      delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
      gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
      advantages[t] = gae
      returns[t] = gae + values[t]
    
    return returns, advantages

  def compute_polychromic_advantage(self, rollout_data):
    """
    Compute polychromic advantages for rollout states.
    
    For each rollout state:
    1. Form M sets of n trajectories
    2. Compute f_poly for each set
    3. Advantage = f_poly(set) - mean(f_poly across sets)
    """
    poly_advantages = {}
    
    for state_key, rollout_info_list in rollout_data.items():
      for rollout_info in rollout_info_list:
        vines = rollout_info['vines']
        state_idx = rollout_info['state_idx']
        
        if len(vines) < self.n_set:
          continue
        
        # Form M sets of n trajectories
        n_sets = len(vines) // self.n_set
        sets = []
        for i in range(n_sets):
          start = i * self.n_set
          end = start + self.n_set
          sets.append(vines[start:end])
        
        # Compute f_poly for each set
        set_scores = []
        for traj_set in sets:
          f_poly, _, _ = self.compute_polychromic_objective(traj_set)
          set_scores.append(f_poly)
        
        # Baseline: mean of all set scores
        baseline = np.mean(set_scores) if set_scores else 0
        
        # Assign advantages to each trajectory in each set
        for set_idx, traj_set in enumerate(sets):
          advantage = set_scores[set_idx] - baseline
          
          for traj in traj_set:
            # All actions in window get the same polychromic advantage
            for t in range(min(self.poly_window, len(traj['states']))):
              key = (state_key, state_idx + t)
              poly_advantages[key] = advantage
    
    return poly_advantages

  def train(self, max_evals=100000):
    """Main training loop with polychromic PPO."""
    total_steps = 0
    iteration = 0
    
    while total_steps < max_evals:
      iteration += 1
      start_iter = time.perf_counter()
      
      # === Phase 1: Collect seed trajectories ===
      seed_trajectories = []
      for _ in range(self.n_vines):
        traj = self.rollout_single(self.env, self.model, self.n_steps)
        seed_trajectories.append(traj)
      
      # === Phase 2: Vine sampling ===
      rollout_data = self.vine_sampling(seed_trajectories)
      
      # === Phase 3: Compute polychromic advantages ===
      poly_advantages = self.compute_polychromic_advantage(rollout_data)
      
      # === Phase 4: Prepare training data ===
      all_states = []
      all_actions = []
      all_returns = []
      all_advantages = []
      all_logprobs = []
      
      for traj in seed_trajectories:
        states = np.array(traj['states'])
        actions = np.array(traj['actions'])
        rewards = np.array(traj['rewards'])
        dones = np.array(traj['dones'])
        values = np.array(traj['values'])
        logprobs = np.array(traj['logprobs'])
        
        # Get next value for GAE
        with torch.no_grad():
          final_state = torch.FloatTensor(traj['final_state']).to(self.device)
          next_value = self.model.critic(final_state.unsqueeze(0)).cpu().numpy().squeeze()
        
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        
        # Check if any states have polychromic advantages
        for t in range(len(states)):
          state_key = tuple(states[t].flatten())
          poly_key = (state_key, t)
          
          if poly_key in poly_advantages:
            # Use polychromic advantage
            advantages[t] = poly_advantages[poly_key]
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_returns.extend(returns)
        all_advantages.extend(advantages)
        all_logprobs.extend(logprobs)
      
      # Convert to tensors
      states_t = torch.FloatTensor(np.array(all_states)).to(self.device)
      actions_t = torch.FloatTensor(np.array(all_actions)).to(self.device)
      returns_t = torch.FloatTensor(np.array(all_returns)).to(self.device)
      advantages_t = torch.FloatTensor(np.array(all_advantages)).to(self.device)
      old_logprobs_t = torch.FloatTensor(np.array(all_logprobs)).to(self.device)
      
      # Normalize advantages
      advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
      
      # === Phase 5: PPO update ===
      for epoch in range(self.epochs):
        # Shuffle data
        indices = torch.randperm(len(states_t))
        batch_size = min(64, len(states_t))
        
        for start in range(0, len(states_t), batch_size):
          end = min(start + batch_size, len(states_t))
          batch_idx = indices[start:end]
          
          batch_states = states_t[batch_idx]
          batch_actions = actions_t[batch_idx]
          batch_returns = returns_t[batch_idx]
          batch_advantages = advantages_t[batch_idx]
          batch_old_logprobs = old_logprobs_t[batch_idx]
          
          # Forward pass
          self.model.actor.std = self.model.actor.log_std.exp()
          new_logprobs = self.model.actor.get_logprob(batch_states, batch_actions)
          values = self.model.critic(batch_states).squeeze()
          
          # Policy loss (clipped)
          ratio = torch.exp(new_logprobs - batch_old_logprobs)
          surr1 = ratio * batch_advantages
          surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
          policy_loss = -torch.min(surr1, surr2).mean()
          
          # Value loss
          value_loss = nn.MSELoss()(values, batch_returns)
          
          # Entropy bonus
          entropy = (torch.log(self.model.actor.std) + 0.5 * (1 + np.log(2 * np.pi))).sum()
          entropy_loss = -self.ent_coeff * entropy
          
          # KL penalty (for stability)
          kl_div = (batch_old_logprobs - new_logprobs).mean()
          kl_loss = self.kl_coeff * kl_div
          
          # Total loss
          loss = policy_loss + self.vf_coeff * value_loss + entropy_loss + kl_loss
          
          self.optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
          self.optimizer.step()
      
      # === Logging ===
      total_steps += len(all_states)
      avg_reward = np.mean([sum(t['rewards']) for t in seed_trajectories])
      
      # Compute diversity stats
      diversity_scores = []
      for rollout_info_list in rollout_data.values():
        for rollout_info in rollout_info_list:
          vines = rollout_info['vines']
          if len(vines) >= self.n_set:
            _, _, div = self.compute_polychromic_objective(vines[:self.n_set])
            diversity_scores.append(div)
      avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
      
      iter_time = time.perf_counter() - start_iter
      
      print(f"iter {iteration:4d} | steps {total_steps:6d} | reward {avg_reward:.3f} | "
            f"diversity {avg_diversity:.3f} | time {iter_time:.2f}s")
      
      self.hist.append((total_steps, avg_reward, avg_diversity))
      
      if self.debug:
        print(f"  policy_loss: {policy_loss.item():.4f}, value_loss: {value_loss.item():.4f}")
        print(f"  n_rollout_states: {len(rollout_data)}, n_poly_advantages: {len(poly_advantages)}")
    
    return self.model.actor, self.hist


def main():
  parser = argparse.ArgumentParser(description="Polychromic PPO for CartLatAccel")
  parser.add_argument("--max_evals", type=int, default=50000)
  parser.add_argument("--n_steps", type=int, default=100, help="Steps per rollout")
  parser.add_argument("--n_set", type=int, default=4, help="Set size for polychromic objective")
  parser.add_argument("--n_vines", type=int, default=8, help="Vines per rollout state")
  parser.add_argument("--n_rollout_states", type=int, default=2, help="Rollout states per trajectory")
  parser.add_argument("--poly_window", type=int, default=5, help="Window for polychromic advantage")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--save_model", action="store_true")
  parser.add_argument("--render", action="store_true")
  args = parser.parse_args()
  
  print(f"=== Polychromic PPO ===")
  print(f"Config: n_set={args.n_set}, n_vines={args.n_vines}, "
        f"n_rollout_states={args.n_rollout_states}, poly_window={args.poly_window}")
  
  # Create environment
  env = gym.make("CartLatAccel-v1", env_bs=1)
  
  # Create model
  model = ActorCritic(
    env.observation_space.shape[-1],
    {"pi": [32], "vf": [32]},
    env.action_space.shape[-1]
  )
  
  # Train
  ppo = PolychroPPO(
    env, model,
    lr=args.lr,
    n_steps=args.n_steps,
    epochs=args.epochs,
    n_set=args.n_set,
    n_vines=args.n_vines,
    n_rollout_states=args.n_rollout_states,
    poly_window=args.poly_window,
    device=args.device,
    debug=args.debug
  )
  
  start = time.time()
  best_model, hist = ppo.train(args.max_evals)
  train_time = time.time() - start
  
  print(f"\n=== Training Complete ===")
  print(f"Total time: {train_time:.2f}s")
  
  # Evaluate
  print(f"\nEvaluating best model...")
  eval_env = gym.make("CartLatAccel-v1", env_bs=1, render_mode="human" if args.render else None)
  
  total_reward = 0
  for _ in range(5):
    traj = ppo.rollout_single(eval_env, ppo.model, max_steps=200, deterministic=True)
    total_reward += sum(traj['rewards'])
  
  print(f"Average eval reward: {total_reward / 5:.3f}")
  
  if args.save_model:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/poly_ppo_best.pt')
    print(f"Model saved to out/poly_ppo_best.pt")
    
    # Save training history
    np.save('out/poly_ppo_hist.npy', np.array(hist))
    print(f"History saved to out/poly_ppo_hist.npy")
  
  env.close()
  eval_env.close()


if __name__ == "__main__":
  main()

