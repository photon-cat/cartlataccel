import os
import time
import json
import numpy as np
import torch
import gymnasium as gym
import gym_cartlataccel
from model import ActorCritic
from ppo import PPO
from eval_cost import calculate_costs
import argparse
from datetime import datetime

def evaluate_model(model, env, n_rollouts=5, device='cpu'):
  """Evaluate model with cost metrics"""
  all_costs = []
  all_rewards = []
  
  for _ in range(n_rollouts):
    states, actions, target_lataccels, actual_lataccels = [], [], [], []
    state, _ = env.reset()
    current_lataccel = 0.0
    episode_reward = 0
    
    for step in range(500):
      state_tensor = torch.FloatTensor(state).to(device)
      action = model.get_action(state_tensor, deterministic=True)
      
      pos, vel, target_pos = state
      pos_error = target_pos - pos
      target_lataccel = pos_error * 10.0
      target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
      
      next_state, reward, terminated, truncated, info = env.step(action)
      
      actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
      
      actions.append(action)
      target_lataccels.append(target_lataccel)
      actual_lataccels.append(actual_lataccel)
      episode_reward += reward
      
      current_lataccel = actual_lataccel
      state = next_state
      
      if terminated or truncated:
        break
    
    costs = calculate_costs(np.array(actual_lataccels), np.array(target_lataccels), dt=env.unwrapped.tau)
    all_costs.append(costs)
    all_rewards.append(episode_reward)
  
  # Aggregate results
  avg_results = {
    'lataccel_cost': np.mean([c['lataccel_cost'] for c in all_costs]),
    'jerk_cost': np.mean([c['jerk_cost'] for c in all_costs]),
    'total_cost': np.mean([c['total_cost'] for c in all_costs]),
    'reward': np.mean(all_rewards),
    'lataccel_cost_std': np.std([c['lataccel_cost'] for c in all_costs]),
    'jerk_cost_std': np.std([c['jerk_cost'] for c in all_costs]),
    'total_cost_std': np.std([c['total_cost'] for c in all_costs]),
    'reward_std': np.std(all_rewards)
  }
  
  return avg_results

def benchmark_training(args):
  """Run training with periodic benchmarking"""
  
  # Setup
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = f"logs/benchmark_{timestamp}"
  os.makedirs(log_dir, exist_ok=True)
  
  print("=" * 70)
  print(f"TRAINING BENCHMARK - {timestamp}")
  print("=" * 70)
  print(f"Total steps: {args.max_evals}")
  print(f"Benchmark interval: {args.benchmark_interval}")
  print(f"Eval rollouts per checkpoint: {args.eval_rollouts}")
  print(f"Device: {args.device}")
  print(f"Noise mode: {args.noise_mode}")
  print(f"Log directory: {log_dir}")
  print("=" * 70)
  print()
  
  # Training environment
  train_env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=args.env_bs)
  
  # Evaluation environment
  eval_env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1)
  
  # Initialize model
  model = ActorCritic(
    train_env.observation_space.shape[-1], 
    {"pi": [32], "vf": [32]}, 
    train_env.action_space.shape[-1]
  )
  
  # PPO trainer
  ppo = PPO(train_env, model, env_bs=args.env_bs, device=args.device, debug=args.debug)
  
  # Benchmark log
  benchmark_log = []
  
  # Initial evaluation (before training)
  print("Evaluating initial (untrained) model...")
  initial_results = evaluate_model(ppo.model.actor, eval_env, n_rollouts=args.eval_rollouts, device=args.device)
  initial_results['timesteps'] = 0
  initial_results['wall_time'] = 0
  benchmark_log.append(initial_results)
  
  print(f"Initial - total_cost: {initial_results['total_cost']:.2f}, reward: {initial_results['reward']:.3f}\n")
  
  # Save initial checkpoint
  checkpoint_path = os.path.join(log_dir, f"model_step_0000000.pt")
  torch.save(ppo.model.actor, checkpoint_path)
  
  # Training loop with periodic benchmarking
  current_steps = 0
  start_time = time.time()
  
  num_benchmarks = args.max_evals // args.benchmark_interval
  
  for benchmark_idx in range(num_benchmarks):
    target_steps = (benchmark_idx + 1) * args.benchmark_interval
    steps_to_train = target_steps - current_steps
    
    print(f"Training from {current_steps} to {target_steps} steps...")
    train_start = time.time()
    
    # Train for benchmark_interval steps
    ppo.train(steps_to_train)
    
    train_time = time.time() - train_start
    current_steps = target_steps
    wall_time = time.time() - start_time
    
    # Evaluate
    print(f"Evaluating at {current_steps} steps...")
    results = evaluate_model(ppo.model.actor, eval_env, n_rollouts=args.eval_rollouts, device=args.device)
    results['timesteps'] = current_steps
    results['wall_time'] = wall_time
    results['train_time_interval'] = train_time
    benchmark_log.append(results)
    
    # Log results
    print(f"Step {current_steps:>6d} | "
          f"total_cost: {results['total_cost']:>8.2f} | "
          f"lataccel: {results['lataccel_cost']:>6.2f} | "
          f"jerk: {results['jerk_cost']:>8.2f} | "
          f"reward: {results['reward']:>7.3f} | "
          f"time: {wall_time:.1f}s")
    
    # Save checkpoint
    checkpoint_path = os.path.join(log_dir, f"model_step_{current_steps:07d}.pt")
    torch.save(ppo.model.actor, checkpoint_path)
    
    # Save log incrementally
    log_path = os.path.join(log_dir, "benchmark_log.json")
    with open(log_path, 'w') as f:
      json.dump(benchmark_log, f, indent=2)
    
    print()
  
  total_time = time.time() - start_time
  
  # Final summary
  print("=" * 70)
  print("TRAINING COMPLETE")
  print("=" * 70)
  print(f"Total time: {total_time:.2f}s")
  print(f"Total steps: {current_steps}")
  print(f"Steps/second: {current_steps/total_time:.0f}")
  print()
  
  # Compare first vs last
  initial = benchmark_log[0]
  final = benchmark_log[-1]
  
  print("INITIAL vs FINAL:")
  print(f"  total_cost:    {initial['total_cost']:>8.2f} → {final['total_cost']:>8.2f} ({((final['total_cost']/initial['total_cost']-1)*100):+.1f}%)")
  print(f"  lataccel_cost: {initial['lataccel_cost']:>8.2f} → {final['lataccel_cost']:>8.2f} ({((final['lataccel_cost']/initial['lataccel_cost']-1)*100):+.1f}%)")
  print(f"  jerk_cost:     {initial['jerk_cost']:>8.2f} → {final['jerk_cost']:>8.2f} ({((final['jerk_cost']/initial['jerk_cost']-1)*100):+.1f}%)")
  print(f"  reward:        {initial['reward']:>8.3f} → {final['reward']:>8.3f} ({((final['reward']/initial['reward']-1)*100):+.1f}%)")
  print("=" * 70)
  
  # Save final summary
  summary = {
    'config': vars(args),
    'timestamp': timestamp,
    'total_time': total_time,
    'total_steps': current_steps,
    'steps_per_second': current_steps/total_time,
    'initial_results': initial,
    'final_results': final,
    'improvement': {
      'total_cost_pct': ((final['total_cost']/initial['total_cost']-1)*100),
      'reward_pct': ((final['reward']/initial['reward']-1)*100)
    }
  }
  
  summary_path = os.path.join(log_dir, "summary.json")
  with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
  
  print(f"\nResults saved to: {log_dir}")
  print(f"  - benchmark_log.json (all checkpoints)")
  print(f"  - summary.json (training summary)")
  print(f"  - model_step_*.pt (model checkpoints)")
  
  return log_dir, benchmark_log

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_evals", type=int, default=100000, help="Total training steps")
  parser.add_argument("--benchmark_interval", type=int, default=10000, help="Steps between benchmarks")
  parser.add_argument("--eval_rollouts", type=int, default=5, help="Rollouts per benchmark")
  parser.add_argument("--env_bs", type=int, default=1000, help="Training batch size")
  parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
  parser.add_argument("--noise_mode", default=None, help="Noise mode (None, REALISTIC, HIGH)")
  parser.add_argument("--debug", action='store_true', help="Enable PPO debug output")
  args = parser.parse_args()
  
  log_dir, benchmark_log = benchmark_training(args)
  
  # Create simple text report
  report_path = os.path.join(log_dir, "report.txt")
  with open(report_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("TRAINING BENCHMARK REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Timesteps | Total Cost | Lataccel Cost | Jerk Cost | Reward\n")
    f.write("-" * 70 + "\n")
    for entry in benchmark_log:
      f.write(f"{entry['timesteps']:>9d} | {entry['total_cost']:>10.2f} | {entry['lataccel_cost']:>13.2f} | {entry['jerk_cost']:>9.2f} | {entry['reward']:>6.3f}\n")
  
  print(f"  - report.txt (text summary)")

if __name__ == "__main__":
  main()

