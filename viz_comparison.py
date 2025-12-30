"""
Visualization tool for Polychromic PPO vs PID comparison.
Outputs matplotlib graphs saved as PNG.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_cartlataccel

from models.model import ActorCritic
from controllers import PIDController


def rollout_pid(env, controller, max_steps=500):
  """Rollout PID controller."""
  state, _ = env.reset()
  controller.reset()
  
  positions = []
  target_positions = []
  actions = []
  rewards = []
  current_lataccel = 0.0
  
  for step in range(max_steps):
    pos, vel, target_pos = state.flatten()[:3]
    positions.append(float(pos))
    target_positions.append(float(target_pos))
    
    pos_error = target_pos - pos
    target_lataccel = np.clip(pos_error * 10.0, -1.0, 1.0)
    
    action = controller.update(target_lataccel, current_lataccel, state, None)
    action = np.clip(action, -1.0, 1.0)
    
    next_state, reward, terminated, truncated, info = env.step(np.array([action]))
    
    actual_lataccel = info.get('noisy_action', action)
    if hasattr(actual_lataccel, '__iter__'):
      actual_lataccel = float(actual_lataccel[0])
    
    actions.append(float(actual_lataccel))
    rewards.append(float(np.sum(reward)))
    
    current_lataccel = actual_lataccel
    state = next_state
    
    if terminated or truncated:
      break
  
  return {
    'positions': np.array(positions),
    'target_positions': np.array(target_positions),
    'actions': np.array(actions),
    'rewards': np.array(rewards)
  }


def rollout_poly(env, model, max_steps=500, deterministic=True, device='cpu'):
  """Rollout Polychromic PPO model."""
  state, _ = env.reset()
  
  positions = []
  target_positions = []
  actions = []
  rewards = []
  
  for step in range(max_steps):
    pos, vel, target_pos = state.flatten()[:3]
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
    
    actual_lataccel = info.get('noisy_action', action_val)
    if hasattr(actual_lataccel, '__iter__'):
      actual_lataccel = float(actual_lataccel[0])
    
    actions.append(float(actual_lataccel))
    rewards.append(float(np.sum(reward)))
    
    state = next_state
    
    if terminated or truncated:
      break
  
  return {
    'positions': np.array(positions),
    'target_positions': np.array(target_positions),
    'actions': np.array(actions),
    'rewards': np.array(rewards)
  }


def create_comparison_plot(pid_data, poly_data, save_path='out/poly_vs_pid_comparison.png'):
  """Create a multi-panel comparison plot."""
  
  # Set up the figure with dark theme
  plt.style.use('dark_background')
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  fig.suptitle('Polychromic PPO vs PID Controller', fontsize=16, fontweight='bold', color='white')
  
  time_steps = np.arange(len(pid_data['positions']))
  time_poly = np.arange(len(poly_data['positions']))
  
  # Color scheme
  pid_color = '#FF6B6B'      # Coral red
  poly_color = '#4ECDC4'     # Teal
  target_color = '#FFE66D'   # Yellow
  
  # === Panel 1: Position Tracking ===
  ax1 = axes[0, 0]
  ax1.plot(time_steps, pid_data['target_positions'], '--', color=target_color, 
           linewidth=2, label='Target', alpha=0.8)
  ax1.plot(time_steps, pid_data['positions'], color=pid_color, 
           linewidth=1.5, label=f'PID (MSE: {np.mean((pid_data["positions"] - pid_data["target_positions"])**2):.2f})')
  ax1.plot(time_poly, poly_data['positions'], color=poly_color, 
           linewidth=1.5, label=f'Poly PPO (MSE: {np.mean((poly_data["positions"] - poly_data["target_positions"])**2):.2f})')
  ax1.set_xlabel('Time Step', fontsize=11)
  ax1.set_ylabel('Position', fontsize=11)
  ax1.set_title('Position Tracking', fontsize=13, fontweight='bold')
  ax1.legend(loc='upper right', fontsize=9)
  ax1.grid(True, alpha=0.3)
  
  # === Panel 2: Tracking Error ===
  ax2 = axes[0, 1]
  pid_error = np.abs(pid_data['positions'] - pid_data['target_positions'])
  poly_error = np.abs(poly_data['positions'] - poly_data['target_positions'])
  
  ax2.fill_between(time_steps, 0, pid_error, color=pid_color, alpha=0.4, label=f'PID (mean: {np.mean(pid_error):.3f})')
  ax2.fill_between(time_poly, 0, poly_error, color=poly_color, alpha=0.4, label=f'Poly PPO (mean: {np.mean(poly_error):.3f})')
  ax2.plot(time_steps, pid_error, color=pid_color, linewidth=1)
  ax2.plot(time_poly, poly_error, color=poly_color, linewidth=1)
  ax2.set_xlabel('Time Step', fontsize=11)
  ax2.set_ylabel('|Position Error|', fontsize=11)
  ax2.set_title('Tracking Error', fontsize=13, fontweight='bold')
  ax2.legend(loc='upper right', fontsize=9)
  ax2.grid(True, alpha=0.3)
  
  # === Panel 3: Control Actions ===
  ax3 = axes[1, 0]
  ax3.plot(time_steps, pid_data['actions'], color=pid_color, linewidth=1, label='PID', alpha=0.8)
  ax3.plot(time_poly, poly_data['actions'], color=poly_color, linewidth=1, label='Poly PPO', alpha=0.8)
  ax3.axhline(y=0, color='white', linestyle='--', alpha=0.3)
  ax3.set_xlabel('Time Step', fontsize=11)
  ax3.set_ylabel('Action (Acceleration)', fontsize=11)
  ax3.set_title('Control Actions', fontsize=13, fontweight='bold')
  ax3.legend(loc='upper right', fontsize=9)
  ax3.grid(True, alpha=0.3)
  ax3.set_ylim(-1.5, 1.5)
  
  # === Panel 4: Metrics Summary ===
  ax4 = axes[1, 1]
  ax4.axis('off')
  
  # Calculate metrics
  pid_pos_mse = np.mean((pid_data['positions'] - pid_data['target_positions'])**2)
  poly_pos_mse = np.mean((poly_data['positions'] - poly_data['target_positions'])**2)
  
  pid_reward = np.sum(pid_data['rewards'])
  poly_reward = np.sum(poly_data['rewards'])
  
  pid_jerk = np.mean(np.diff(pid_data['actions'])**2)
  poly_jerk = np.mean(np.diff(poly_data['actions'])**2)
  
  # Create metrics table
  metrics_text = f"""
╔══════════════════════════════════════════════════════╗
║           PERFORMANCE COMPARISON                      ║
╠══════════════════════════════════════════════════════╣
║  Metric                    PID         Poly PPO       ║
╠══════════════════════════════════════════════════════╣
║  Position MSE              {pid_pos_mse:>8.4f}    {poly_pos_mse:>8.4f}       ║
║  Total Reward              {pid_reward:>8.2f}    {poly_reward:>8.2f}       ║
║  Action Jerk               {pid_jerk:>8.4f}    {poly_jerk:>8.4f}       ║
╠══════════════════════════════════════════════════════╣
║  WINNER:                                              ║
║    Position Tracking: {'Poly PPO ✓' if poly_pos_mse < pid_pos_mse else 'PID ✓':>12}                    ║
║    Total Reward:      {'Poly PPO ✓' if poly_reward > pid_reward else 'PID ✓':>12}                    ║
║    Smoothness:        {'PID ✓' if pid_jerk < poly_jerk else 'Poly PPO ✓':>12}                    ║
╚══════════════════════════════════════════════════════╝

  Polychromic PPO: Maintains diverse strategies
  to prevent entropy collapse during RL training.
"""
  
  ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#2C3E50', alpha=0.8, edgecolor='#4ECDC4'))
  
  plt.tight_layout()
  
  # Save
  os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
  plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
  print(f"Saved comparison plot to {save_path}")
  
  return fig


def create_pass_at_k_plot(env, model, pid_controller, save_path='out/pass_at_k_comparison.png', 
                          n_attempts=20, n_configs=10, device='cpu'):
  """Create pass@k visualization."""
  
  plt.style.use('dark_background')
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  fig.suptitle('Pass@k Performance (Diversity Evaluation)', fontsize=14, fontweight='bold')
  
  k_values = list(range(1, n_attempts + 1))
  success_threshold = -50
  
  # Collect pass@k data
  pid_pass_rates = []
  poly_pass_rates = []
  
  for k in k_values:
    pid_successes = 0
    poly_successes = 0
    
    for config in range(n_configs):
      env.reset(seed=config * 42)
      
      # PID attempts
      pid_rewards = []
      for _ in range(k):
        pid_controller.reset()
        data = rollout_pid(env, pid_controller, max_steps=200)
        pid_rewards.append(np.sum(data['rewards']))
      if max(pid_rewards) > success_threshold:
        pid_successes += 1
      
      # Poly attempts (stochastic)
      poly_rewards = []
      for _ in range(k):
        data = rollout_poly(env, model, max_steps=200, deterministic=False, device=device)
        poly_rewards.append(np.sum(data['rewards']))
      if max(poly_rewards) > success_threshold:
        poly_successes += 1
    
    pid_pass_rates.append(pid_successes / n_configs)
    poly_pass_rates.append(poly_successes / n_configs)
    
    if k % 5 == 0:
      print(f"  Computed pass@{k}: PID={pid_successes}/{n_configs}, Poly={poly_successes}/{n_configs}")
  
  # Colors
  pid_color = '#FF6B6B'
  poly_color = '#4ECDC4'
  
  # === Panel 1: Pass@k Line Plot ===
  ax1 = axes[0]
  ax1.plot(k_values, pid_pass_rates, 'o-', color=pid_color, linewidth=2, markersize=4, label='PID')
  ax1.plot(k_values, poly_pass_rates, 's-', color=poly_color, linewidth=2, markersize=4, label='Poly PPO')
  ax1.fill_between(k_values, pid_pass_rates, alpha=0.2, color=pid_color)
  ax1.fill_between(k_values, poly_pass_rates, alpha=0.2, color=poly_color)
  ax1.set_xlabel('k (Number of Attempts)', fontsize=11)
  ax1.set_ylabel('Pass Rate', fontsize=11)
  ax1.set_title('Pass@k: Probability of Success with k Attempts', fontsize=12, fontweight='bold')
  ax1.legend(loc='lower right', fontsize=10)
  ax1.grid(True, alpha=0.3)
  ax1.set_ylim(0, 1.05)
  ax1.set_xlim(1, n_attempts)
  
  # Add annotations
  for k in [1, 5, 10, 20]:
    if k <= n_attempts:
      idx = k - 1
      ax1.annotate(f'{poly_pass_rates[idx]:.0%}', 
                   xy=(k, poly_pass_rates[idx]), 
                   xytext=(k+1, poly_pass_rates[idx]+0.08),
                   fontsize=9, color=poly_color)
  
  # === Panel 2: Bar Chart for Key k Values ===
  ax2 = axes[1]
  key_ks = [1, 5, 10, 20]
  key_ks = [k for k in key_ks if k <= n_attempts]
  x = np.arange(len(key_ks))
  width = 0.35
  
  pid_bars = [pid_pass_rates[k-1] for k in key_ks]
  poly_bars = [poly_pass_rates[k-1] for k in key_ks]
  
  bars1 = ax2.bar(x - width/2, pid_bars, width, label='PID', color=pid_color, alpha=0.8)
  bars2 = ax2.bar(x + width/2, poly_bars, width, label='Poly PPO', color=poly_color, alpha=0.8)
  
  ax2.set_xlabel('k (Number of Attempts)', fontsize=11)
  ax2.set_ylabel('Pass Rate', fontsize=11)
  ax2.set_title('Pass@k Comparison', fontsize=12, fontweight='bold')
  ax2.set_xticks(x)
  ax2.set_xticklabels([f'pass@{k}' for k in key_ks])
  ax2.legend(loc='upper left', fontsize=10)
  ax2.set_ylim(0, 1.1)
  ax2.grid(True, alpha=0.3, axis='y')
  
  # Add value labels on bars
  for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.0%}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=pid_color)
  
  for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.0%}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=poly_color)
  
  plt.tight_layout()
  plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
  print(f"Saved pass@k plot to {save_path}")
  
  return fig


def main():
  import argparse
  parser = argparse.ArgumentParser(description="Visualize Poly PPO vs PID comparison")
  parser.add_argument("--model_path", type=str, default="out/poly_ppo_benchmark.pt", 
                      help="Path to trained Poly PPO model")
  parser.add_argument("--output_dir", type=str, default="out", help="Output directory for plots")
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--n_rollouts", type=int, default=1, help="Rollouts for trajectory plot")
  parser.add_argument("--skip_passk", action="store_true", help="Skip pass@k evaluation (slow)")
  args = parser.parse_args()
  
  print("=" * 60)
  print("  POLYCHROMIC PPO vs PID VISUALIZATION")
  print("=" * 60)
  
  # Create environment
  env = gym.make("CartLatAccel-v1", env_bs=1)
  
  # Load Poly PPO model
  if os.path.exists(args.model_path):
    print(f"\nLoading model from {args.model_path}")
    model = ActorCritic(
      env.observation_space.shape[-1],
      {"pi": [64, 64], "vf": [64, 64]},
      env.action_space.shape[-1]
    )
    model.actor = torch.load(args.model_path, map_location=args.device)
  else:
    print(f"\nModel not found at {args.model_path}")
    print("Please run benchmark_poly_vs_pid.py first to train a model.")
    return
  
  # Create PID controller
  pid_controller = PIDController(p=0.195, i=0.100, d=-0.053)
  
  # Set seed for reproducibility
  np.random.seed(args.seed)
  env.reset(seed=args.seed)
  
  # === Generate trajectory data ===
  print("\nGenerating rollout data...")
  pid_controller.reset()
  pid_data = rollout_pid(env, pid_controller, max_steps=500)
  
  env.reset(seed=args.seed)  # Same seed for fair comparison
  poly_data = rollout_poly(env, model, max_steps=500, deterministic=True, device=args.device)
  
  # === Create comparison plot ===
  print("\nCreating comparison plot...")
  os.makedirs(args.output_dir, exist_ok=True)
  create_comparison_plot(
    pid_data, poly_data, 
    save_path=os.path.join(args.output_dir, 'poly_vs_pid_comparison.png')
  )
  
  # === Create pass@k plot ===
  if not args.skip_passk:
    print("\nCreating pass@k plot (this may take a minute)...")
    create_pass_at_k_plot(
      env, model, pid_controller,
      save_path=os.path.join(args.output_dir, 'pass_at_k_comparison.png'),
      n_attempts=20,
      n_configs=10,
      device=args.device
    )
  else:
    print("\nSkipping pass@k plot (use --skip_passk=false to include)")
  
  env.close()
  
  print("\n" + "=" * 60)
  print(f"  Plots saved to {args.output_dir}/")
  print("=" * 60)


if __name__ == "__main__":
  main()

