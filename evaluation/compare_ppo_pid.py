"""
Simple comparison: Custom PPO vs PID using same evaluation metrics
"""

import time
import json
import os
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
import gym_cartlataccel
from model import ActorCritic
from ppo import PPO
from controllers import PIDController
from eval_cost import calculate_costs

def rollout_and_evaluate(env, model_or_controller, is_ppo=True, max_steps=500, device='cpu'):
    """Rollout and compute costs"""
    states, actions, target_lataccels, actual_lataccels = [], [], [], []
    
    state, _ = env.reset()
    if not is_ppo:
        model_or_controller.reset()
    current_lataccel = 0.0
    total_reward = 0
    
    for step in range(max_steps):
        # Calculate target lataccel
        pos, vel, target_pos = state[0], state[1], state[2]
        pos_error = target_pos - pos
        target_lataccel = pos_error * 10.0
        target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
        
        if is_ppo:
            # PPO model
            state_tensor = torch.FloatTensor(state).to(device)
            action = model_or_controller.get_action(state_tensor, deterministic=True)
        else:
            # PID controller
            action = model_or_controller.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, terminated, truncated, info = env.step(np.array([action]))
        
        actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
        
        states.append(state)
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

def main():
    print("=" * 80)
    print("COMPREHENSIVE COMPARISON: Custom PPO vs PID")
    print("=" * 80)
    print()
    
    device = 'cpu'
    noise_mode = None
    n_rollouts = 10
    
    # Create environment
    env = gym.make("CartLatAccel-v1", noise_mode=noise_mode, env_bs=1)
    
    results = {
        'pid': [],
        'ppo_10k': [],
        'ppo_30k': [],
        'ppo_100k': []
    }
    
    # 1. PID Baseline
    print("1. Evaluating PID Controller...")
    print("   P=0.195, I=0.1, D=-0.053")
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    for i in range(n_rollouts):
        costs = rollout_and_evaluate(env, pid, is_ppo=False)
        results['pid'].append(costs)
        if i == 0:
            print(f"   First rollout: total_cost={costs['total_cost']:.2f}")
    
    pid_avg = {
        'lataccel_cost': np.mean([c['lataccel_cost'] for c in results['pid']]),
        'jerk_cost': np.mean([c['jerk_cost'] for c in results['pid']]),
        'total_cost': np.mean([c['total_cost'] for c in results['pid']]),
        'reward': np.mean([c['reward'] for c in results['pid']]),
    }
    print(f"   Average: total_cost={pid_avg['total_cost']:.2f}, reward={pid_avg['reward']:.3f}")
    print()
    
    # 2. Train PPO for 10k, 30k, 100k and evaluate each
    training_configs = [
        ('ppo_10k', 10000),
        ('ppo_30k', 30000),
        ('ppo_100k', 100000),
    ]
    
    for name, max_evals in training_configs:
        print(f"2. Training PPO with {max_evals:,} steps...")
        train_env = gym.make("CartLatAccel-v1", noise_mode=noise_mode, env_bs=1000)
        model = ActorCritic(
            train_env.observation_space.shape[-1],
            {"pi": [32], "vf": [32]},
            train_env.action_space.shape[-1]
        )
        ppo = PPO(train_env, model, env_bs=1000, device=device)
        
        start = time.time()
        best_model, hist = ppo.train(max_evals)
        train_time = time.time() - start
        
        print(f"   Training complete in {train_time:.2f}s")
        print(f"   Evaluating {name}...")
        
        for i in range(n_rollouts):
            costs = rollout_and_evaluate(env, best_model, is_ppo=True, device=device)
            results[name].append(costs)
            if i == 0:
                print(f"   First rollout: total_cost={costs['total_cost']:.2f}")
        
        ppo_avg = {
            'lataccel_cost': np.mean([c['lataccel_cost'] for c in results[name]]),
            'jerk_cost': np.mean([c['jerk_cost'] for c in results[name]]),
            'total_cost': np.mean([c['total_cost'] for c in results[name]]),
            'reward': np.mean([c['reward'] for c in results[name]]),
        }
        print(f"   Average: total_cost={ppo_avg['total_cost']:.2f}, reward={ppo_avg['reward']:.3f}")
        print(f"   vs PID: {ppo_avg['total_cost']/pid_avg['total_cost']:.2f}x")
        print()
    
    # Final comparison table
    print("=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print()
    print("Method      | Total Cost | Lataccel | Jerk Cost | Reward   | vs PID")
    print("-" * 80)
    
    for name in ['pid', 'ppo_10k', 'ppo_30k', 'ppo_100k']:
        avg = {
            'lataccel_cost': np.mean([c['lataccel_cost'] for c in results[name]]),
            'jerk_cost': np.mean([c['jerk_cost'] for c in results[name]]),
            'total_cost': np.mean([c['total_cost'] for c in results[name]]),
            'reward': np.mean([c['reward'] for c in results[name]]),
        }
        
        vs_pid = avg['total_cost'] / pid_avg['total_cost']
        label = name.replace('_', ' ').upper().ljust(11)
        
        print(f"{label} | {avg['total_cost']:>10,.0f} | {avg['lataccel_cost']:>8.2f} | "
              f"{avg['jerk_cost']:>9,.0f} | {avg['reward']:>8.3f} | {vs_pid:>5.2f}x")
    
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/comparison_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        # Convert numpy types to native Python types for JSON
        json_results = {}
        for key, val_list in results.items():
            json_results[key] = []
            for item in val_list:
                json_item = {k: float(v) if isinstance(v, np.ndarray) else v 
                            for k, v in item.items()}
                json_results[key].append(json_item)
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {log_dir}")

if __name__ == "__main__":
    main()

