#!/usr/bin/env python3
"""
tinycart.py - Run any controller on the CartLatAccel environment

Usage:
  python tinycart.py --controller pid
  python tinycart.py --controller ppo --render
  python tinycart.py --controller pid --episodes 10
  python tinycart.py --controller ppo --plot           # saves to plots/plot.png
  python tinycart.py --controller pid --log logs/run.csv  # save trajectory log
"""
import argparse
import importlib
import os
import gymnasium as gym
import numpy as np
import json
import gym_cartlataccel

def run_episode(controller, env, max_steps=500, on_target=False):
  """Run one episode, return total reward and trajectory data
  
  Args:
    on_target: If True, start cart exactly on trajectory (pos=target, vel=target_vel)
               This tests controller with perfect initial conditions
  """
  obs, _ = env.reset()
  controller.reset()
  
  # Get trajectory from env (it's generated on reset)
  trajectory = env.unwrapped.x_targets[0]  # first batch
  
  # Pass trajectory to 2DOF controller for feedforward
  if hasattr(controller, 'set_trajectory'):
    controller.set_trajectory(trajectory)
  
  # Optionally start on target trajectory
  if on_target:
    # Start at trajectory[0] with vel=0 (trajectory is generated from rest)
    target_pos = trajectory[0]
    env.unwrapped.state[0] = np.array([target_pos, 0.0, target_pos], dtype=np.float32)
    obs = env.unwrapped.state[0]
  
  # Collect trajectory data (including noise info from env)
  data = {
    'pos': [], 'target': [], 'vel': [], 
    'action': [], 'noisy_action': [], 'reward': []
  }
  
  for step in range(max_steps):
    pos, vel, target = obs[0], obs[1], obs[2]
    action = controller.act(obs)
    
    data['pos'].append(pos)
    data['target'].append(target)
    data['vel'].append(vel)
    data['action'].append(float(action))
    
    obs, reward, done, truncated, info = env.step(np.array([[action]]))
    data['noisy_action'].append(float(info['noisy_action']))
    data['reward'].append(float(reward))
    
    if done or truncated:
      break
  
  # Convert to numpy
  for k in data:
    data[k] = np.array(data[k])
  
  data['error'] = np.abs(data['pos'] - data['target'])
  data['action_noise'] = data['noisy_action'] - data['action']  # noise injected
  data['cum_reward'] = np.cumsum(data['reward'])
  data['total_reward'] = float(data['reward'].sum())
  
  return data

def plot_episode(data, filename='plot.png'):
  """Plot trajectory data from an episode"""
  import matplotlib.pyplot as plt
  
  fig, axes = plt.subplots(3, 2, figsize=(12, 8))
  t = np.arange(len(data['pos']))
  
  # Position tracking
  ax = axes[0, 0]
  ax.plot(t, data['pos'], label='pos', linewidth=1.5)
  ax.plot(t, data['target'], label='target', linewidth=1.5, linestyle='--')
  ax.set_ylabel('Position')
  ax.set_title('Position Tracking')
  ax.legend()
  ax.grid(True, alpha=0.3)
  
  # Error
  ax = axes[0, 1]
  ax.plot(t, data['error'], color='red', linewidth=1.5)
  ax.set_ylabel('|pos - target|')
  ax.set_title(f'Tracking Error (mean: {data["error"].mean():.3f})')
  ax.grid(True, alpha=0.3)
  
  # Velocity
  ax = axes[1, 0]
  ax.plot(t, data['vel'], color='green', linewidth=1.5)
  ax.set_ylabel('Velocity')
  ax.set_title('Velocity')
  ax.grid(True, alpha=0.3)
  
  # Action (show both commanded and actual noisy action)
  ax = axes[1, 1]
  ax.plot(t, data['action'], color='purple', linewidth=1.5, label='commanded')
  if 'noisy_action' in data:
    ax.plot(t, data['noisy_action'], color='magenta', linewidth=1, alpha=0.7, label='actual (w/ noise)')
  ax.set_ylabel('Action')
  ax.set_title('Action')
  ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)
  
  # Reward
  ax = axes[2, 0]
  ax.plot(t, data['reward'], color='orange', linewidth=1.5)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('Reward')
  ax.set_title('Reward per Step')
  ax.grid(True, alpha=0.3)
  
  # Cumulative reward
  ax = axes[2, 1]
  ax.plot(t, data['cum_reward'], color='blue', linewidth=1.5)
  ax.set_xlabel('Timestep')
  ax.set_ylabel('Cumulative Reward')
  ax.set_title(f'Cumulative Reward (total: {data["total_reward"]:.2f})')
  ax.grid(True, alpha=0.3)
  
  plt.tight_layout()
  os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
  plt.savefig(filename, dpi=150)
  print(f'Plot saved to {filename}')
  plt.close()

def save_log(data, filename, noise_level, controller_name):
  """Save trajectory log to CSV or JSON"""
  import csv
  
  os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
  
  if filename.endswith('.json'):
    # JSON format - includes metadata
    log = {
      'metadata': {
        'controller': controller_name,
        'noise': noise_level,
        'total_reward': data['total_reward'],
        'mean_error': float(data['error'].mean()),
        'timesteps': len(data['pos']),
      },
      'trajectory': {
        'pos': data['pos'].tolist(),
        'target': data['target'].tolist(),
        'vel': data['vel'].tolist(),
        'action': data['action'].tolist(),
        'noisy_action': data['noisy_action'].tolist(),
        'action_noise': data['action_noise'].tolist(),
        'reward': data['reward'].tolist(),
        'error': data['error'].tolist(),
      }
    }
    with open(filename, 'w') as f:
      json.dump(log, f, indent=2)
  else:
    # CSV format - one row per timestep
    with open(filename, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['step', 'pos', 'target', 'vel', 'action', 'noisy_action', 'action_noise', 'reward', 'error'])
      for i in range(len(data['pos'])):
        writer.writerow([
          i, 
          data['pos'][i], data['target'][i], data['vel'][i],
          data['action'][i], data['noisy_action'][i], data['action_noise'][i],
          data['reward'][i], data['error'][i]
        ])
  print(f'Log saved to {filename}')

def main():
  parser = argparse.ArgumentParser(description='Run a controller on CartLatAccel')
  parser.add_argument('--controller', type=str, default='pid', choices=['pid', 'ppo', 'twodof', 'ff'],
                      help='Controller to run (default: pid)')
  parser.add_argument('--episodes', type=int, default=1,
                      help='Number of episodes to run (default: 1)')
  parser.add_argument('--render', action='store_true',
                      help='Show pygame visualization')
  parser.add_argument('--noise', type=float, default=0.5,
                      help='Noise level 0-1 (default: 0.5, 0=none, 1=high)')
  parser.add_argument('--model_path', type=str, default='models/ppo.pt',
                      help='Path to trained model for ppo controller')
  parser.add_argument('--plot', type=str, nargs='?', const='plots/plot.png', default=None,
                      help='Save plot to file (default: plots/plot.png)')
  parser.add_argument('--log', type=str, default=None,
                      help='Save trajectory log to logs/ folder (.csv or .json)')
  parser.add_argument('--on_target', action='store_true',
                      help='Start cart on target trajectory (pos=target, vel=target_vel)')
  args = parser.parse_args()

  # Load controller
  controller_module = importlib.import_module(f'controllers.{args.controller}')
  if args.controller == 'ppo':
    controller = controller_module.Controller(model_path=args.model_path)
  else:
    controller = controller_module.Controller()

  # Create env
  render_mode = 'human' if args.render else None
  env = gym.make('CartLatAccel-v1', noise=args.noise, render_mode=render_mode)

  # Run episodes
  rewards = []
  last_data = None
  for ep in range(args.episodes):
    data = run_episode(controller, env, max_steps=500, on_target=args.on_target)
    rewards.append(data['total_reward'])
    last_data = data
    if args.episodes > 1:
      print(f'Episode {ep+1}: reward {data["total_reward"]:.3f}')

  # Summary
  rewards = np.array(rewards)
  print(f'\n=== Results ({args.controller}) ===')
  print(f'Episodes: {args.episodes}')
  print(f'Avg reward: {rewards.mean():.3f}')
  if args.episodes > 1:
    print(f'Std reward: {rewards.std():.3f}')
    print(f'Min/Max: {rewards.min():.3f} / {rewards.max():.3f}')

  # Plot last episode
  if args.plot and last_data:
    plot_episode(last_data, args.plot)

  # Save log
  if args.log and last_data:
    save_log(last_data, args.log, args.noise, args.controller)

  env.close()

if __name__ == '__main__':
  main()
