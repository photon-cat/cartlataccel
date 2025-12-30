"""
Train PPO with jerk-aware rewards to prevent self-destruction
Compares original env vs jerk-aware env
"""

import time
import json
import os
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
import gym_cartlataccel
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from model import ActorCritic
from ppo import PPO
from eval_cost import calculate_costs
import argparse


# Register jerk environment
try:
    register(
        id='CartLatAccel-Jerk-v1',
        entry_point='jerk_env:CartLatAccelJerkEnv',
        max_episode_steps=500,
    )
except:
    pass  # Already registered


def rollout_and_evaluate(env, model, max_steps=500, device='cpu'):
    """Rollout and compute costs"""
    states, actions, target_lataccels, actual_lataccels = [], [], [], []
    
    state, _ = env.reset()
    current_lataccel = 0.0
    total_reward = 0
    
    for step in range(max_steps):
        pos, vel, target_pos = state[0], state[1], state[2]
        pos_error = target_pos - pos
        target_lataccel = pos_error * 10.0
        target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
        
        state_tensor = torch.FloatTensor(state).to(device)
        action = model.get_action(state_tensor, deterministic=True)
        
        next_state, reward, terminated, truncated, info = env.step(np.array([action]))
        
        actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
        
        actions.append(action)
        target_lataccels.append(target_lataccel)
        actual_lataccels.append(actual_lataccel)
        total_reward += reward
        
        current_lataccel = actual_lataccel
        state = next_state
        
        if terminated or truncated:
            break
    
    costs = calculate_costs(np.array(actual_lataccels), np.array(target_lataccels), dt=0.02)
    costs['reward'] = total_reward
    
    return costs


def train_and_evaluate(env_type, max_evals, n_eval_rollouts=10, device='cpu', jerk_weight=0.01, action_weight=0.001):
    """Train PPO and evaluate at checkpoints"""
    print(f"\nTraining on {env_type} environment...")
    print(f"  Max evals: {max_evals:,}")
    if env_type == "jerk":
        print(f"  Jerk penalty weight: {jerk_weight}")
        print(f"  Action penalty weight: {action_weight}")
    
    # Create training environment
    if env_type == "original":
        train_env = gym.make("CartLatAccel-v1", env_bs=1000)
        eval_env = gym.make("CartLatAccel-v1", env_bs=1)
    else:
        train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000, 
                            jerk_penalty_weight=jerk_weight,
                            action_penalty_weight=action_weight)
        eval_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                           jerk_penalty_weight=jerk_weight,
                           action_penalty_weight=action_weight)
    
    # Create model
    model = ActorCritic(
        train_env.observation_space.shape[-1],
        {"pi": [32], "vf": [32]},
        train_env.action_space.shape[-1]
    )
    
    # Create PPO trainer
    ppo = PPO(train_env, model, env_bs=1000, device=device)
    
    # Training checkpoints
    checkpoints = [10000, 30000, 50000, 100000]
    checkpoints = [c for c in checkpoints if c <= max_evals]
    
    results = {}
    current_steps = 0
    
    for checkpoint in checkpoints:
        steps_to_train = checkpoint - current_steps
        
        print(f"  Training from {current_steps:,} to {checkpoint:,} steps...")
        start = time.time()
        ppo.train(steps_to_train)
        train_time = time.time() - start
        
        current_steps = checkpoint
        
        print(f"  Evaluating at {checkpoint:,} steps...")
        checkpoint_results = []
        for _ in range(n_eval_rollouts):
            costs = rollout_and_evaluate(eval_env, ppo.model.actor, device=device)
            checkpoint_results.append(costs)
        
        avg_costs = {
            'lataccel_cost': np.mean([c['lataccel_cost'] for c in checkpoint_results]),
            'jerk_cost': np.mean([c['jerk_cost'] for c in checkpoint_results]),
            'total_cost': np.mean([c['total_cost'] for c in checkpoint_results]),
            'reward': np.mean([c['reward'] for c in checkpoint_results]),
            'train_time': train_time,
            'timesteps': checkpoint,
        }
        
        results[checkpoint] = avg_costs
        
        print(f"    total_cost: {avg_costs['total_cost']:>10,.2f}")
        print(f"    jerk_cost:  {avg_costs['jerk_cost']:>10,.2f}")
        print(f"    reward:     {avg_costs['reward']:>10.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_evals", type=int, default=100000, help="Max training steps")
    parser.add_argument("--jerk_weight", type=float, default=0.01, help="Jerk penalty weight")
    parser.add_argument("--action_weight", type=float, default=0.001, help="Action penalty weight")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--n_eval_rollouts", type=int, default=10, help="Evaluation rollouts")
    args = parser.parse_args()
    
    print("=" * 80)
    print("PPO TRAINING: ORIGINAL vs JERK-AWARE REWARDS")
    print("=" * 80)
    print(f"Max training steps: {args.max_evals:,}")
    print(f"Evaluation rollouts per checkpoint: {args.n_eval_rollouts}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Train on original environment
    print("\n" + "=" * 80)
    print("1. ORIGINAL ENVIRONMENT (position error only)")
    print("=" * 80)
    original_results = train_and_evaluate(
        "original", 
        args.max_evals, 
        n_eval_rollouts=args.n_eval_rollouts,
        device=args.device
    )
    
    # Train on jerk-aware environment
    print("\n" + "=" * 80)
    print("2. JERK-AWARE ENVIRONMENT (position error + jerk penalty)")
    print("=" * 80)
    jerk_results = train_and_evaluate(
        "jerk", 
        args.max_evals,
        n_eval_rollouts=args.n_eval_rollouts,
        device=args.device,
        jerk_weight=args.jerk_weight,
        action_weight=args.action_weight
    )
    
    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: ORIGINAL vs JERK-AWARE")
    print("=" * 80)
    print("\nSteps   | Environment | Total Cost | Lataccel | Jerk Cost  | Reward")
    print("-" * 80)
    
    for steps in sorted(original_results.keys()):
        orig = original_results[steps]
        jerk = jerk_results[steps]
        
        print(f"{steps:>7,} | Original    | {orig['total_cost']:>10,.0f} | "
              f"{orig['lataccel_cost']:>8.2f} | {orig['jerk_cost']:>10,.0f} | {orig['reward']:>7.3f}")
        print(f"{steps:>7,} | Jerk-Aware  | {jerk['total_cost']:>10,.0f} | "
              f"{jerk['lataccel_cost']:>8.2f} | {jerk['jerk_cost']:>10,.0f} | {jerk['reward']:>7.3f}")
        
        improvement = ((orig['total_cost'] - jerk['total_cost']) / orig['total_cost']) * 100
        print(f"         Improvement: {improvement:+.1f}%")
        print()
    
    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Check if jerk-aware prevents self-destruction
    orig_10k = original_results[10000]['total_cost']
    orig_100k = original_results[100000]['total_cost']
    jerk_10k = jerk_results[10000]['total_cost']
    jerk_100k = jerk_results[100000]['total_cost']
    
    orig_degradation = ((orig_100k / orig_10k) - 1) * 100
    jerk_degradation = ((jerk_100k / jerk_10k) - 1) * 100
    
    print(f"\nOriginal Environment:")
    print(f"  10k → 100k steps: {orig_degradation:+.1f}% change")
    if orig_degradation > 100:
        print(f"  ⚠️  SELF-DESTRUCTS with more training!")
    
    print(f"\nJerk-Aware Environment:")
    print(f"  10k → 100k steps: {jerk_degradation:+.1f}% change")
    if abs(jerk_degradation) < 50:
        print(f"  ✓ STABLE training!")
    elif jerk_degradation < 0:
        print(f"  ✓ IMPROVES with more training!")
    
    # PID comparison
    pid_cost = 2086  # From previous benchmarks
    best_orig = min([r['total_cost'] for r in original_results.values()])
    best_jerk = min([r['total_cost'] for r in jerk_results.values()])
    
    print(f"\nComparison with PID (cost = {pid_cost:,}):")
    print(f"  Original PPO (best):   {best_orig/pid_cost:.1f}x worse")
    print(f"  Jerk-Aware PPO (best): {best_jerk/pid_cost:.1f}x worse")
    
    if best_jerk < best_orig:
        improvement = ((best_orig - best_jerk) / best_orig) * 100
        print(f"  ✓ Jerk-Aware is {improvement:.1f}% better than Original!")
    
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/jerk_comparison_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    results = {
        'config': vars(args),
        'timestamp': timestamp,
        'original': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                        for kk, vv in v.items()} 
                    for k, v in original_results.items()},
        'jerk_aware': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                          for kk, vv in v.items()} 
                      for k, v in jerk_results.items()},
        'analysis': {
            'orig_degradation_pct': float(orig_degradation),
            'jerk_degradation_pct': float(jerk_degradation),
            'best_orig': float(best_orig),
            'best_jerk': float(best_jerk),
        }
    }
    
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    main()

