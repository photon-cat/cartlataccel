"""
Train large network PPO - last serious attempt to beat PID
"""

import time
import json
import os
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from model_large import ActorCriticLarge
from model import ActorCritic
from ppo import PPO
from eval_cost import calculate_costs
from controllers import PIDController
import argparse

# Register environment
try:
    register(
        id='CartLatAccel-Jerk-v1',
        entry_point='jerk_env:CartLatAccelJerkEnv',
        max_episode_steps=500,
    )
except:
    pass


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


def train_large_network(args):
    """Train with large network"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/large_network_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 80)
    print("LARGE NETWORK PPO - FINAL ATTEMPT TO BEAT PID")
    print("=" * 80)
    print(f"Network: {args.hidden_size} hidden units x {args.n_layers} layers")
    print(f"Total steps: {args.max_evals:,}")
    print(f"Jerk weight: {args.jerk_weight}")
    print(f"Action weight: {args.action_weight}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Log directory: {log_dir}")
    print("=" * 80)
    print()
    
    # Create environments
    train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000,
                        jerk_penalty_weight=args.jerk_weight,
                        action_penalty_weight=args.action_weight)
    eval_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                       jerk_penalty_weight=args.jerk_weight,
                       action_penalty_weight=args.action_weight)
    
    # Create LARGE model (using original ActorCritic with bigger dims)
    from model import ActorCritic
    
    hidden_dims_pi = [args.hidden_size] * args.n_layers
    hidden_dims_vf = [args.hidden_size] * args.n_layers
    
    print(f"Creating model with architecture:")
    print(f"  Actor: {train_env.observation_space.shape[-1]} -> {hidden_dims_pi} -> {train_env.action_space.shape[-1]}")
    print(f"  Critic: {train_env.observation_space.shape[-1]} -> {hidden_dims_vf} -> 1")
    
    model = ActorCritic(
        train_env.observation_space.shape[-1],
        {"pi": hidden_dims_pi, "vf": hidden_dims_vf},
        train_env.action_space.shape[-1]
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print()
    
    # Create PPO trainer with lower learning rate for bigger network
    ppo = PPO(train_env, model, env_bs=1000, device=args.device, lr=args.lr)
    
    # PID baseline
    print("Evaluating PID baseline...")
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    pid_results = []
    for _ in range(5):
        state, _ = eval_env.reset()
        pid.reset()
        actions_pid, target_accels, actual_accels = [], [], []
        current_lataccel = 0.0
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            pos_error = target_pos - pos
            target_lataccel = pos_error * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            action = pid.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, terminated, truncated, info = eval_env.step(np.array([action]))
            
            actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
            
            actions_pid.append(action)
            target_accels.append(target_lataccel)
            actual_accels.append(actual_lataccel)
            
            current_lataccel = actual_lataccel
            state = next_state
            
            if terminated or truncated:
                break
        
        costs = calculate_costs(np.array(actual_accels), np.array(target_accels), dt=0.02)
        pid_results.append(costs)
    
    pid_avg = {
        'total_cost': np.mean([r['total_cost'] for r in pid_results]),
        'lataccel_cost': np.mean([r['lataccel_cost'] for r in pid_results]),
        'jerk_cost': np.mean([r['jerk_cost'] for r in pid_results]),
    }
    print(f"PID baseline: total_cost={pid_avg['total_cost']:.2f}")
    print()
    
    # Training loop
    checkpoints = list(range(args.checkpoint_interval, args.max_evals + 1, args.checkpoint_interval))
    benchmark_log = []
    current_steps = 0
    start_time = time.time()
    
    best_cost = float('inf')
    best_checkpoint = 0
    
    for checkpoint in checkpoints:
        steps_to_train = checkpoint - current_steps
        
        print(f"Training from {current_steps:,} to {checkpoint:,} steps...")
        train_start = time.time()
        ppo.train(steps_to_train)
        train_time = time.time() - train_start
        
        current_steps = checkpoint
        
        print(f"  Training completed in {train_time:.2f}s")
        print(f"  Evaluating...")
        
        # Evaluate
        checkpoint_results = []
        for _ in range(args.eval_rollouts):
            costs = rollout_and_evaluate(eval_env, ppo.model.actor, device=args.device)
            checkpoint_results.append(costs)
        
        avg_costs = {
            'timesteps': checkpoint,
            'wall_time': time.time() - start_time,
            'train_time': train_time,
            'lataccel_cost': np.mean([c['lataccel_cost'] for c in checkpoint_results]),
            'jerk_cost': np.mean([c['jerk_cost'] for c in checkpoint_results]),
            'total_cost': np.mean([c['total_cost'] for c in checkpoint_results]),
            'reward': np.mean([c['reward'] for c in checkpoint_results]),
            'vs_pid': np.mean([c['total_cost'] for c in checkpoint_results]) / pid_avg['total_cost'],
        }
        
        benchmark_log.append(avg_costs)
        
        # Track best
        if avg_costs['total_cost'] < best_cost:
            best_cost = avg_costs['total_cost']
            best_checkpoint = checkpoint
            print(f"  ⭐ NEW BEST! Cost={best_cost:.2f} ({avg_costs['vs_pid']:.2f}x vs PID)")
        
        print(f"  Total cost:    {avg_costs['total_cost']:>10,.2f} (vs PID: {avg_costs['vs_pid']:.2f}x)")
        print(f"  Lataccel cost: {avg_costs['lataccel_cost']:>10.2f}")
        print(f"  Jerk cost:     {avg_costs['jerk_cost']:>10,.2f}")
        
        # Check if we beat PID!
        if avg_costs['vs_pid'] < 1.0:
            print(f"  🎉🎉🎉 WE BEAT PID! 🎉🎉🎉")
        elif avg_costs['vs_pid'] < 2.0:
            print(f"  🎯 Within 2x of PID!")
        elif avg_costs['vs_pid'] < 5.0:
            print(f"  ✓ Within 5x of PID")
        
        print()
        
        # Save checkpoint
        checkpoint_path = os.path.join(log_dir, f"model_step_{checkpoint:07d}.pt")
        torch.save(ppo.model.actor.state_dict(), checkpoint_path)
        
        # Save log
        log_path = os.path.join(log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump({
                'config': vars(args),
                'pid_baseline': pid_avg,
                'checkpoints': benchmark_log,
                'best_checkpoint': best_checkpoint,
                'best_cost': float(best_cost),
            }, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print()
    
    best_result = [r for r in benchmark_log if r['timesteps'] == best_checkpoint][0]
    final_result = benchmark_log[-1]
    
    print(f"PID Baseline:       {pid_avg['total_cost']:>10,.2f} (1.00x)")
    print(f"PPO Best:           {best_result['total_cost']:>10,.2f} ({best_result['vs_pid']:.2f}x) at {best_checkpoint:,} steps")
    print(f"PPO Final:          {final_result['total_cost']:>10,.2f} ({final_result['vs_pid']:.2f}x)")
    print()
    
    if best_result['vs_pid'] < 1.0:
        print("🎉🎉🎉 SUCCESS! PPO BEAT PID! 🎉🎉🎉")
    elif best_result['vs_pid'] < 2.0:
        print("🎯 CLOSE! Within 2x of PID - Very impressive!")
    elif best_result['vs_pid'] < 5.0:
        print("✓ Good! Within 5x of PID")
    elif best_result['vs_pid'] < 10.0:
        print("Within 10x - Respectable performance")
    else:
        print("Still significantly worse than PID")
    
    print("=" * 80)
    
    return log_dir, benchmark_log, pid_avg


def main():
    parser = argparse.ArgumentParser(description="Large Network PPO Training")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--max_evals", type=int, default=500000, help="Total training steps")
    parser.add_argument("--checkpoint_interval", type=int, default=50000, help="Steps between checkpoints")
    parser.add_argument("--eval_rollouts", type=int, default=10, help="Rollouts per evaluation")
    parser.add_argument("--jerk_weight", type=float, default=0.005, help="Jerk penalty weight")
    parser.add_argument("--action_weight", type=float, default=0.0005, help="Action penalty weight")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()
    
    log_dir, benchmark_log, pid_avg = train_large_network(args)
    
    # Save summary
    best = min(benchmark_log, key=lambda x: x['total_cost'])
    
    summary_path = os.path.join(log_dir, "SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LARGE NETWORK PPO - FINAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Network: {args.hidden_size} hidden units x {args.n_layers} layers\n")
        f.write(f"Total parameters: {sum(p.numel() for p in ActorCriticLarge(3, {'pi': [args.hidden_size]*args.n_layers, 'vf': [args.hidden_size]*args.n_layers}, 1).parameters()):,}\n\n")
        f.write(f"PID:      {pid_avg['total_cost']:,.2f}\n")
        f.write(f"PPO Best: {best['total_cost']:,.2f} ({best['vs_pid']:.2f}x) at {best['timesteps']:,} steps\n\n")
        
        if best['vs_pid'] < 1.0:
            f.write("🎉 PPO BEAT PID!\n")
        elif best['vs_pid'] < 2.0:
            f.write("🎯 Within 2x of PID!\n")
        else:
            f.write(f"PPO is {best['vs_pid']:.2f}x worse than PID\n")
    
    print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    main()

