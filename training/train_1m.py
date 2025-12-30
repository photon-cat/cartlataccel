"""
1 Million Step Training Run with Jerk-Aware PPO
Test how good PPO can get with extended stable training
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


def train_1m_steps(args):
    """Train for 1M steps with regular checkpoints"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/1m_training_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 80)
    print("1 MILLION STEP TRAINING - JERK-AWARE PPO")
    print("=" * 80)
    print(f"Total steps: {args.max_evals:,}")
    print(f"Checkpoint interval: {args.checkpoint_interval:,}")
    print(f"Evaluation rollouts: {args.eval_rollouts}")
    print(f"Jerk weight: {args.jerk_weight}")
    print(f"Action weight: {args.action_weight}")
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
    
    # Create model
    model = ActorCritic(
        train_env.observation_space.shape[-1],
        {"pi": [32], "vf": [32]},
        train_env.action_space.shape[-1]
    )
    
    # Create PPO trainer
    ppo = PPO(train_env, model, env_bs=1000, device=args.device)
    
    # PID baseline for comparison
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    print("Evaluating PID baseline...")
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
    
    # Training loop with checkpoints
    checkpoints = list(range(args.checkpoint_interval, args.max_evals + 1, args.checkpoint_interval))
    benchmark_log = []
    current_steps = 0
    start_time = time.time()
    
    for checkpoint in checkpoints:
        steps_to_train = checkpoint - current_steps
        
        print(f"Training from {current_steps:,} to {checkpoint:,} steps...")
        train_start = time.time()
        ppo.train(steps_to_train)
        train_time = time.time() - train_start
        
        current_steps = checkpoint
        
        print(f"  Training completed in {train_time:.2f}s")
        print(f"  Evaluating at {checkpoint:,} steps...")
        
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
        
        print(f"  Results:")
        print(f"    Total cost:    {avg_costs['total_cost']:>10,.2f}")
        print(f"    Lataccel cost: {avg_costs['lataccel_cost']:>10.2f}")
        print(f"    Jerk cost:     {avg_costs['jerk_cost']:>10,.2f}")
        print(f"    Reward:        {avg_costs['reward']:>10.3f}")
        print(f"    vs PID:        {avg_costs['vs_pid']:>10.2f}x")
        print()
        
        # Save checkpoint
        checkpoint_path = os.path.join(log_dir, f"model_step_{checkpoint:07d}.pt")
        torch.save(ppo.model.actor.state_dict(), checkpoint_path)
        
        # Save log incrementally
        log_path = os.path.join(log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump({
                'config': vars(args),
                'pid_baseline': pid_avg,
                'checkpoints': benchmark_log,
            }, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Final analysis
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - ANALYSIS")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Steps/second: {args.max_evals/total_time:.0f}")
    print()
    
    # Find best checkpoint
    best_checkpoint = min(benchmark_log, key=lambda x: x['total_cost'])
    final_checkpoint = benchmark_log[-1]
    
    print("Best Checkpoint:")
    print(f"  Steps:         {best_checkpoint['timesteps']:,}")
    print(f"  Total cost:    {best_checkpoint['total_cost']:,.2f}")
    print(f"  vs PID:        {best_checkpoint['vs_pid']:.2f}x")
    print()
    
    print("Final Checkpoint (1M steps):")
    print(f"  Total cost:    {final_checkpoint['total_cost']:,.2f}")
    print(f"  vs PID:        {final_checkpoint['vs_pid']:.2f}x")
    print()
    
    # Check improvement trend
    first = benchmark_log[0]
    last = benchmark_log[-1]
    improvement = ((first['total_cost'] - last['total_cost']) / first['total_cost']) * 100
    
    print("Training Progression:")
    print(f"  First checkpoint ({first['timesteps']:,}):  {first['total_cost']:,.2f}")
    print(f"  Last checkpoint ({last['timesteps']:,}):   {last['total_cost']:,.2f}")
    print(f"  Change:                                    {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"  ✓ Training IMPROVED performance!")
    elif improvement > -10:
        print(f"  ✓ Training STABLE (no degradation)!")
    else:
        print(f"  ⚠️  Performance degraded slightly")
    print()
    
    # Comparison with PID
    print(f"Comparison with PID:")
    print(f"  PID:           {pid_avg['total_cost']:>10,.2f}")
    print(f"  PPO (best):    {best_checkpoint['total_cost']:>10,.2f}  ({best_checkpoint['vs_pid']:.2f}x)")
    print(f"  PPO (final):   {final_checkpoint['total_cost']:>10,.2f}  ({final_checkpoint['vs_pid']:.2f}x)")
    
    if best_checkpoint['vs_pid'] < 10:
        print(f"  ✓ Within 10x of PID!")
    if best_checkpoint['vs_pid'] < 5:
        print(f"  ✓✓ Within 5x of PID - Excellent!")
    if best_checkpoint['vs_pid'] < 2:
        print(f"  ✓✓✓ Within 2x of PID - AMAZING!")
    
    print("=" * 80)
    
    # Create summary report
    with open(os.path.join(log_dir, "SUMMARY.txt"), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("1 MILLION STEP TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
        f.write(f"Best checkpoint: {best_checkpoint['timesteps']:,} steps\n")
        f.write(f"  Total cost: {best_checkpoint['total_cost']:,.2f}\n")
        f.write(f"  vs PID: {best_checkpoint['vs_pid']:.2f}x\n\n")
        f.write(f"Final checkpoint: 1,000,000 steps\n")
        f.write(f"  Total cost: {final_checkpoint['total_cost']:,.2f}\n")
        f.write(f"  vs PID: {final_checkpoint['vs_pid']:.2f}x\n\n")
        f.write(f"Training improvement: {improvement:+.1f}%\n")
    
    print(f"\nResults saved to: {log_dir}")
    
    return log_dir, benchmark_log, pid_avg


def main():
    parser = argparse.ArgumentParser(description="1M Step Training for Jerk-Aware PPO")
    parser.add_argument("--max_evals", type=int, default=1000000, help="Total training steps")
    parser.add_argument("--checkpoint_interval", type=int, default=100000, help="Steps between checkpoints")
    parser.add_argument("--eval_rollouts", type=int, default=10, help="Rollouts per evaluation")
    parser.add_argument("--jerk_weight", type=float, default=0.01, help="Jerk penalty weight")
    parser.add_argument("--action_weight", type=float, default=0.001, help="Action penalty weight")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()
    
    log_dir, benchmark_log, pid_avg = train_1m_steps(args)
    
    # Create progress table
    print("\n" + "=" * 80)
    print("DETAILED PROGRESS TABLE")
    print("=" * 80)
    print()
    print("Steps    | Total Cost | Lataccel | Jerk Cost | Reward    | vs PID | Improv")
    print("-" * 80)
    
    for i, entry in enumerate(benchmark_log):
        if i == 0:
            improv = "baseline"
        else:
            prev_cost = benchmark_log[i-1]['total_cost']
            curr_cost = entry['total_cost']
            improv_pct = ((prev_cost - curr_cost) / prev_cost) * 100
            improv = f"{improv_pct:+.1f}%"
        
        print(f"{entry['timesteps']:>8,} | {entry['total_cost']:>10,.0f} | "
              f"{entry['lataccel_cost']:>8.2f} | {entry['jerk_cost']:>9,.0f} | "
              f"{entry['reward']:>9.3f} | {entry['vs_pid']:>6.2f}x | {improv:>6}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

