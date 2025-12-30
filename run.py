#!/usr/bin/env python
"""
Unified runner for CartLatAccel controllers.
Supports: PID, PPO (vanilla), Poly (Polychromic PPO)

Usage:
  python run.py pid --eval                    # Evaluate PID
  python run.py ppo --train --max_evals 50000 # Train PPO
  python run.py poly --train --max_evals 50000 # Train Polychromic PPO
  python run.py ppo --eval --model out/ppo/model.pt  # Evaluate trained PPO
"""

import os
import json
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import numpy as np
import torch
import gymnasium as gym
import gym_cartlataccel
import matplotlib.pyplot as plt

from controllers import PIDController, PPOController, PolyController
from models.model import ActorCritic


@dataclass
class RunConfig:
  """Configuration for a run."""
  controller: str
  mode: str  # train or eval
  max_evals: int = 50000
  n_steps: int = 100
  lr: float = 3e-4
  epochs: int = 2
  device: str = "cpu"
  seed: int = 42
  # Poly-specific
  n_set: int = 4
  n_vines: int = 8
  n_rollout_states: int = 2
  poly_window: int = 5
  # PID-specific (tuned for position control)
  pid_p: float = 2.0
  pid_i: float = 0.01
  pid_d: float = 0.5
  # Eval
  n_eval_episodes: int = 10
  eval_steps: int = 200
  model_path: Optional[str] = None
  # Environment
  n_segments: int = 10  # number of waypoints for target trajectory
  # Output
  output_dir: str = "out"
  render: bool = False


@dataclass 
class RunLog:
  """Training/eval log entry."""
  step: int
  reward: float
  timestamp: float
  # Optional training metrics
  loss: Optional[float] = None
  diversity: Optional[float] = None
  entropy: Optional[float] = None


class Runner:
  """Unified runner for all controller types."""
  
  def __init__(self, config: RunConfig):
    self.config = config
    self.logs: List[RunLog] = []
    self.start_time = time.time()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.run_dir = os.path.join(config.output_dir, config.controller, timestamp)
    os.makedirs(self.run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
      json.dump(asdict(config), f, indent=2)
    
    print(f"Run directory: {self.run_dir}")
  
  def log(self, step: int, reward: float, **kwargs):
    """Add a log entry."""
    entry = RunLog(
      step=step,
      reward=reward,
      timestamp=time.time() - self.start_time,
      **kwargs
    )
    self.logs.append(entry)
    
    # Save logs incrementally
    self._save_logs()
  
  def _save_logs(self):
    """Save logs to JSON."""
    log_path = os.path.join(self.run_dir, "logs.json")
    with open(log_path, 'w') as f:
      json.dump([asdict(l) for l in self.logs], f, indent=2)
  
  def save_model(self, model, name="model.pt"):
    """Save a model."""
    path = os.path.join(self.run_dir, name)
    torch.save(model, path)
    print(f"Model saved: {path}")
    return path
  
  def save_history(self, history: List[Tuple], name="history.npy"):
    """Save training history."""
    path = os.path.join(self.run_dir, name)
    np.save(path, np.array(history))
    print(f"History saved: {path}")
    return path
  
  def plot_trajectory(self, positions, targets, actions=None, rewards=None, name="trajectory.png"):
    """Plot position tracking trajectory."""
    plt.style.use('dark_background')
    
    n_panels = 2 if actions is None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels))
    
    time_steps = np.arange(len(positions))
    
    # Colors
    pos_color = '#4ECDC4'    # Teal
    target_color = '#FFE66D'  # Yellow
    error_color = '#FF6B6B'   # Coral
    action_color = '#A8E6CF'  # Mint
    
    # Panel 1: Position tracking
    ax1 = axes[0]
    ax1.plot(time_steps, targets, '--', color=target_color, linewidth=2, label='Target', alpha=0.8)
    ax1.plot(time_steps, positions, color=pos_color, linewidth=1.5, label='Position')
    ax1.fill_between(time_steps, positions, targets, alpha=0.2, color=error_color)
    ax1.set_ylabel('Position', fontsize=11)
    ax1.set_title(f'{self.config.controller.upper()} Position Tracking', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Error
    ax2 = axes[1]
    error = np.abs(np.array(positions) - np.array(targets))
    ax2.fill_between(time_steps, 0, error, color=error_color, alpha=0.5)
    ax2.plot(time_steps, error, color=error_color, linewidth=1)
    ax2.set_ylabel('|Error|', fontsize=11)
    ax2.set_title(f'Tracking Error (mean: {np.mean(error):.3f})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Actions (if provided)
    if actions is not None and n_panels == 3:
      ax3 = axes[2]
      ax3.plot(time_steps[:len(actions)], actions, color=action_color, linewidth=1)
      ax3.axhline(y=0, color='white', linestyle='--', alpha=0.3)
      ax3.set_ylabel('Action', fontsize=11)
      ax3.set_xlabel('Time Step', fontsize=11)
      ax3.set_title('Control Actions', fontsize=12)
      ax3.grid(True, alpha=0.3)
      ax3.set_ylim(-1.5, 1.5)
    else:
      axes[-1].set_xlabel('Time Step', fontsize=11)
    
    plt.tight_layout()
    
    path = os.path.join(self.run_dir, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Plot saved: {path}")
    return path


# ============================================================
# PID Controller
# ============================================================

def run_pid(runner: Runner):
  """Run PID controller (eval only)."""
  config = runner.config
  
  print("\n" + "=" * 50)
  print("  PID Controller")
  print("=" * 50)
  print(f"  P={config.pid_p}, I={config.pid_i}, D={config.pid_d}")
  
  env = gym.make("CartLatAccel-v1", env_bs=1, n_segments=config.n_segments,
                 render_mode="human" if config.render else None)
  
  # Use PID controller from controllers module
  controller = PIDController()
  controller.p = config.pid_p
  controller.i = config.pid_i
  controller.d = config.pid_d
  
  all_rewards = []
  all_positions = []
  all_targets = []
  all_actions = []
  
  for ep in range(config.n_eval_episodes):
    state, _ = env.reset(seed=config.seed + ep)
    controller.reset()
    
    episode_reward = 0
    positions = []
    targets = []
    actions = []
    
    for step in range(config.eval_steps):
      pos, vel, target_pos = state.flatten()[:3]
      positions.append(float(pos))
      targets.append(float(target_pos))
      
      # PID directly controls position: error = target - current
      # target_lataccel = target_pos, current_lataccel = pos
      action = controller.update(target_pos, pos, state, None)
      action = np.clip(action, -1.0, 1.0)
      actions.append(float(action))
      
      next_state, reward, terminated, truncated, info = env.step(np.array([action]))
      episode_reward += float(np.sum(reward))
      
      state = next_state
      if terminated or truncated:
        break
    
    all_rewards.append(episode_reward)
    all_positions.append(positions)
    all_targets.append(targets)
    all_actions.append(actions)
    
    runner.log(ep + 1, episode_reward)
    print(f"  Episode {ep+1}/{config.n_eval_episodes}: reward={episode_reward:.2f}")
  
  env.close()
  
  # Summary
  avg_reward = np.mean(all_rewards)
  std_reward = np.std(all_rewards)
  
  # Calculate position MSE
  all_pos = np.array(all_positions)
  all_tgt = np.array(all_targets)
  position_mse = np.mean((all_pos - all_tgt) ** 2)
  
  summary = {
    "controller": "pid",
    "avg_reward": float(avg_reward),
    "std_reward": float(std_reward),
    "position_mse": float(position_mse),
    "n_episodes": config.n_eval_episodes,
    "params": {"p": config.pid_p, "i": config.pid_i, "d": config.pid_d}
  }
  
  with open(os.path.join(runner.run_dir, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
  
  print(f"\n  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
  print(f"  Position MSE: {position_mse:.4f}")
  
  # Plot best episode trajectory
  best_ep = np.argmax(all_rewards)
  runner.plot_trajectory(
    all_positions[best_ep], 
    all_targets[best_ep], 
    all_actions[best_ep],
    name="trajectory.png"
  )
  
  return summary


# ============================================================
# PPO (Vanilla)
# ============================================================

def run_ppo(runner: Runner):
  """Run vanilla PPO."""
  from ppo import PPO
  
  config = runner.config
  
  print("\n" + "=" * 50)
  print("  PPO (Vanilla)")
  print("=" * 50)
  
  if config.mode == "train":
    print(f"  Training for {config.max_evals} steps")
    
    env = gym.make("CartLatAccel-v1", env_bs=1, n_segments=config.n_segments)
    model = ActorCritic(
      env.observation_space.shape[-1],
      {"pi": [64, 64], "vf": [64, 64]},
      env.action_space.shape[-1]
    )
    
    ppo = PPO(
      env, model,
      lr=config.lr,
      n_steps=config.n_steps,
      epochs=config.epochs,
      env_bs=1,
      device=config.device,
      debug=False
    )
    
    # Custom training loop with logging
    eps = 0
    while eps < config.max_evals:
      states, actions, rewards, dones, next_state = ppo.rollout(
        env, ppo.model.actor, ppo.n_steps, device=config.device
      )
      
      with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(states)).to(config.device)
        next_state_tensor = torch.FloatTensor(next_state).to(config.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(config.device)
        values = ppo.model.critic(state_tensor).cpu().numpy().squeeze()
        next_values = ppo.model.critic(next_state_tensor).cpu().numpy().squeeze()
        ppo.model.actor.std = ppo.model.actor.log_std.exp().to(config.device)
        logprobs = ppo.model.actor.get_logprob(state_tensor, action_tensor).cpu().numpy().squeeze()
      
      returns, advantages = ppo.compute_gae(np.array(rewards), values, np.array(dones), next_values)
      
      from tensordict import TensorDict
      episode_dict = TensorDict({
        "states": state_tensor,
        "actions": action_tensor,
        "returns": torch.FloatTensor(returns).to(config.device),
        "advantages": torch.FloatTensor(advantages).to(config.device),
        "logprobs": logprobs,
      }, batch_size=ppo.n_steps)
      ppo.replay_buffer.extend(episode_dict)
      
      total_loss = 0
      for _ in range(ppo.epochs):
        for batch in ppo.replay_buffer:
          adv = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
          costs = ppo.evaluate_cost(batch['states'], batch['actions'], batch['returns'], adv, batch['logprobs'])
          loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"]
          ppo.optimizer.zero_grad()
          loss.backward()
          ppo.optimizer.step()
          total_loss = loss.item()
          break
      ppo.replay_buffer.empty()
      
      eps += 1
      avg_reward = np.sum(rewards)
      
      # Log with entropy
      entropy = -costs["entropy"].item() / ppo.ent_coeff if hasattr(costs["entropy"], 'item') else 0
      runner.log(eps, avg_reward, loss=total_loss, entropy=entropy)
      
      if eps % 100 == 0:
        print(f"  Step {eps}/{config.max_evals}: reward={avg_reward:.2f}, loss={total_loss:.4f}")
    
    env.close()
    
    # Save model
    runner.save_model(ppo.model.actor, "model.pt")
    runner.save_history([(l.step, l.reward) for l in runner.logs], "history.npy")
    
    # Evaluate
    config.model_path = os.path.join(runner.run_dir, "model.pt")
    config.mode = "eval"
    return _eval_ppo(runner, ppo.model.actor)
  
  else:  # eval mode
    if not config.model_path:
      print("  ERROR: --model required for eval mode")
      return None
    
    model = torch.load(config.model_path, map_location=config.device, weights_only=False)
    return _eval_ppo(runner, model)


def _eval_ppo(runner: Runner, model):
  """Evaluate a PPO model."""
  config = runner.config
  
  print(f"\n  Evaluating over {config.n_eval_episodes} episodes...")
  
  env = gym.make("CartLatAccel-v1", env_bs=1, n_segments=config.n_segments,
                 render_mode="human" if config.render else None)
  
  all_rewards = []
  all_positions = []
  all_targets = []
  
  for ep in range(config.n_eval_episodes):
    state, _ = env.reset(seed=config.seed + ep)
    episode_reward = 0
    positions = []
    targets = []
    
    for step in range(config.eval_steps):
      pos, vel, target_pos = state.flatten()[:3]
      positions.append(float(pos))
      targets.append(float(target_pos))
      
      state_tensor = torch.FloatTensor(state).to(config.device)
      with torch.no_grad():
        action = model.get_action(state_tensor, deterministic=True)
      
      next_state, reward, terminated, truncated, info = env.step(action)
      episode_reward += float(np.sum(reward))
      
      state = next_state
      if terminated or truncated:
        break
    
    all_rewards.append(episode_reward)
    all_positions.append(positions)
    all_targets.append(targets)
    print(f"  Episode {ep+1}: reward={episode_reward:.2f}")
  
  env.close()
  
  avg_reward = np.mean(all_rewards)
  std_reward = np.std(all_rewards)
  
  all_pos = np.array(all_positions)
  all_tgt = np.array(all_targets)
  position_mse = np.mean((all_pos - all_tgt) ** 2)
  
  summary = {
    "controller": "ppo",
    "avg_reward": float(avg_reward),
    "std_reward": float(std_reward),
    "position_mse": float(position_mse),
    "n_episodes": config.n_eval_episodes,
  }
  
  with open(os.path.join(runner.run_dir, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
  
  print(f"\n  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
  print(f"  Position MSE: {position_mse:.4f}")
  
  return summary


# ============================================================
# Polychromic PPO
# ============================================================

def run_poly(runner: Runner):
  """Run Polychromic PPO."""
  from ppo_poly import PolychroPPO
  
  config = runner.config
  
  print("\n" + "=" * 50)
  print("  Polychromic PPO")
  print("=" * 50)
  print(f"  n_set={config.n_set}, n_vines={config.n_vines}, poly_window={config.poly_window}")
  
  if config.mode == "train":
    print(f"  Training for {config.max_evals} steps")
    
    env = gym.make("CartLatAccel-v1", env_bs=1, n_segments=config.n_segments)
    model = ActorCritic(
      env.observation_space.shape[-1],
      {"pi": [64, 64], "vf": [64, 64]},
      env.action_space.shape[-1]
    )
    
    ppo = PolychroPPO(
      env, model,
      lr=config.lr,
      n_steps=config.n_steps,
      epochs=config.epochs,
      n_set=config.n_set,
      n_vines=config.n_vines,
      n_rollout_states=config.n_rollout_states,
      poly_window=config.poly_window,
      device=config.device,
      debug=False
    )
    
    # Train with logging callback
    class LogCallback:
      def __init__(self, runner):
        self.runner = runner
        self.step = 0
      
      def __call__(self, reward, diversity=None, loss=None):
        self.step += 1
        self.runner.log(self.step, reward, diversity=diversity, loss=loss)
    
    callback = LogCallback(runner)
    
    # Modified train loop
    eps = 0
    while eps < config.max_evals:
      # Collect trajectories using vine sampling
      trajectory_sets = []
      for _ in range(config.n_set):
        traj = ppo.rollout_single(env, ppo.model, max_steps=config.n_steps, deterministic=False)
        trajectory_sets.append(traj)
      
      # Compute polychromic objective
      f_poly, norm_reward, diversity = ppo.compute_polychromic_objective(trajectory_sets)
      
      # Process trajectories for PPO update
      all_states, all_actions, all_rewards_list = [], [], []
      all_values, all_logprobs = [], []
      
      for traj in trajectory_sets:
        states = np.array(traj['states'])
        if states.ndim == 3:
          states = states.squeeze(axis=1)
        actions = np.array(traj['actions']).reshape(-1, 1)
        rewards = np.array(traj['rewards'])
        logprobs = np.array(traj['logprobs'])
        
        with torch.no_grad():
          state_t = torch.FloatTensor(states).to(config.device)
          values = ppo.model.critic(state_t).cpu().numpy().squeeze()
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards_list.extend(rewards)
        all_values.extend(values if hasattr(values, '__iter__') else [values])
        all_logprobs.extend(logprobs)
      
      # Compute advantages with polychromic weighting
      all_states = np.array(all_states)
      all_actions = np.array(all_actions)
      all_rewards_arr = np.array(all_rewards_list)
      all_values = np.array(all_values)
      all_logprobs = np.array(all_logprobs)
      
      # Simple GAE for now
      returns = np.zeros_like(all_rewards_arr)
      advantages = np.zeros_like(all_rewards_arr)
      gae = 0
      for t in reversed(range(len(all_rewards_arr))):
        next_val = all_values[t + 1] if t + 1 < len(all_values) else 0
        delta = all_rewards_arr[t] + 0.99 * next_val - all_values[t]
        gae = delta + 0.99 * 0.95 * gae
        advantages[t] = gae
        returns[t] = gae + all_values[t]
      
      # Apply diversity bonus to advantages
      advantages = advantages * (1 + 0.1 * diversity)
      
      # PPO update
      state_t = torch.FloatTensor(all_states).to(config.device)
      action_t = torch.FloatTensor(all_actions).to(config.device)
      returns_t = torch.FloatTensor(returns).to(config.device)
      adv_t = torch.FloatTensor(advantages).to(config.device)
      old_logprobs_t = torch.FloatTensor(all_logprobs).to(config.device)
      
      adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
      
      total_loss = 0
      for _ in range(config.epochs):
        ppo.model.actor.std = ppo.model.actor.log_std.exp()
        new_logprobs = ppo.model.actor.get_logprob(state_t, action_t).squeeze()
        
        ratio = torch.exp(new_logprobs - old_logprobs_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = torch.nn.MSELoss()(ppo.model.critic(state_t).squeeze(), returns_t)
        
        entropy = (torch.log(ppo.model.actor.std) + 0.5 * (1 + np.log(2 * np.pi))).sum()
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        ppo.optimizer.zero_grad()
        loss.backward()
        ppo.optimizer.step()
        total_loss = loss.item()
      
      eps += config.n_set
      avg_reward = np.mean([sum(t['rewards']) for t in trajectory_sets])
      
      runner.log(eps, avg_reward, diversity=diversity, loss=total_loss)
      
      if eps % 100 == 0:
        print(f"  Step {eps}/{config.max_evals}: reward={avg_reward:.2f}, diversity={diversity:.3f}")
    
    env.close()
    
    # Save
    runner.save_model(ppo.model.actor, "model.pt")
    runner.save_history([(l.step, l.reward, l.diversity) for l in runner.logs], "history.npy")
    
    # Evaluate
    config.model_path = os.path.join(runner.run_dir, "model.pt")
    config.mode = "eval"
    return _eval_poly(runner, ppo.model.actor)
  
  else:  # eval
    if not config.model_path:
      print("  ERROR: --model required for eval mode")
      return None
    
    model = torch.load(config.model_path, map_location=config.device, weights_only=False)
    return _eval_poly(runner, model)


def _eval_poly(runner: Runner, model):
  """Evaluate a Polychromic PPO model."""
  config = runner.config
  
  print(f"\n  Evaluating over {config.n_eval_episodes} episodes...")
  
  env = gym.make("CartLatAccel-v1", env_bs=1, n_segments=config.n_segments,
                 render_mode="human" if config.render else None)
  
  all_rewards = []
  all_positions = []
  all_targets = []
  
  for ep in range(config.n_eval_episodes):
    state, _ = env.reset(seed=config.seed + ep)
    episode_reward = 0
    positions = []
    targets = []
    
    for step in range(config.eval_steps):
      pos, vel, target_pos = state.flatten()[:3]
      positions.append(float(pos))
      targets.append(float(target_pos))
      
      state_tensor = torch.FloatTensor(state).to(config.device)
      with torch.no_grad():
        action = model.get_action(state_tensor, deterministic=True)
      
      next_state, reward, terminated, truncated, info = env.step(action)
      episode_reward += float(np.sum(reward))
      
      state = next_state
      if terminated or truncated:
        break
    
    all_rewards.append(episode_reward)
    all_positions.append(positions)
    all_targets.append(targets)
    print(f"  Episode {ep+1}: reward={episode_reward:.2f}")
  
  env.close()
  
  avg_reward = np.mean(all_rewards)
  std_reward = np.std(all_rewards)
  
  all_pos = np.array(all_positions)
  all_tgt = np.array(all_targets)
  position_mse = np.mean((all_pos - all_tgt) ** 2)
  
  summary = {
    "controller": "poly",
    "avg_reward": float(avg_reward),
    "std_reward": float(std_reward),
    "position_mse": float(position_mse),
    "n_episodes": config.n_eval_episodes,
  }
  
  with open(os.path.join(runner.run_dir, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
  
  print(f"\n  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
  print(f"  Position MSE: {position_mse:.4f}")
  
  return summary


# ============================================================
# Main
# ============================================================

def main():
  parser = argparse.ArgumentParser(
    description="Unified runner for CartLatAccel controllers",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python run.py pid --eval
  python run.py ppo --train --max_evals 10000
  python run.py poly --train --max_evals 10000
  python run.py ppo --eval --model out/ppo/model.pt
    """
  )
  
  parser.add_argument("controller", choices=["pid", "ppo", "poly"],
                      help="Controller type")
  
  # Mode
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument("--train", action="store_true", help="Train the controller")
  mode.add_argument("--eval", action="store_true", help="Evaluate the controller")
  
  # Training
  parser.add_argument("--max_evals", type=int, default=10000, help="Max training steps")
  parser.add_argument("--n_steps", type=int, default=100, help="Steps per rollout")
  parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
  parser.add_argument("--epochs", type=int, default=2, help="PPO epochs")
  parser.add_argument("--device", type=str, default="cpu", help="Device")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  
  # Poly-specific
  parser.add_argument("--n_set", type=int, default=4, help="Poly: set size")
  parser.add_argument("--n_vines", type=int, default=8, help="Poly: vines per state")
  parser.add_argument("--n_rollout_states", type=int, default=2, help="Poly: rollout states")
  parser.add_argument("--poly_window", type=int, default=5, help="Poly: advantage window")
  
  # PID-specific (tuned for position control)
  parser.add_argument("--pid_p", type=float, default=2.0, help="PID P gain")
  parser.add_argument("--pid_i", type=float, default=0.01, help="PID I gain")
  parser.add_argument("--pid_d", type=float, default=0.5, help="PID D gain")
  
  # Eval
  parser.add_argument("--n_eval", type=int, default=10, help="Eval episodes")
  parser.add_argument("--eval_steps", type=int, default=200, help="Steps per eval episode")
  parser.add_argument("--model", type=str, default=None, help="Model path for eval")
  
  # Environment
  parser.add_argument("--n_segments", type=int, default=10, help="Number of waypoints in target trajectory")
  
  # Output
  parser.add_argument("--output_dir", type=str, default="out", help="Output directory")
  parser.add_argument("--render", action="store_true", help="Render environment")
  
  args = parser.parse_args()
  
  # Build config
  config = RunConfig(
    controller=args.controller,
    mode="train" if args.train else "eval",
    max_evals=args.max_evals,
    n_steps=args.n_steps,
    lr=args.lr,
    epochs=args.epochs,
    device=args.device,
    seed=args.seed,
    n_set=args.n_set,
    n_vines=args.n_vines,
    n_rollout_states=args.n_rollout_states,
    poly_window=args.poly_window,
    pid_p=args.pid_p,
    pid_i=args.pid_i,
    pid_d=args.pid_d,
    n_eval_episodes=args.n_eval,
    eval_steps=args.eval_steps,
    model_path=args.model,
    n_segments=args.n_segments,
    output_dir=args.output_dir,
    render=args.render,
  )
  
  # Create runner
  runner = Runner(config)
  
  print("\n" + "=" * 50)
  print(f"  CartLatAccel Runner")
  print("=" * 50)
  print(f"  Controller: {config.controller.upper()}")
  print(f"  Mode: {config.mode}")
  
  # Run
  if config.controller == "pid":
    if config.mode == "train":
      print("  Note: PID has no training, running eval instead")
    summary = run_pid(runner)
  elif config.controller == "ppo":
    summary = run_ppo(runner)
  elif config.controller == "poly":
    summary = run_poly(runner)
  
  print("\n" + "=" * 50)
  print(f"  Run complete!")
  print(f"  Output: {runner.run_dir}")
  print("=" * 50)
  
  return summary


if __name__ == "__main__":
  main()

