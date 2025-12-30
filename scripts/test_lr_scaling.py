"""
Test 512x4 with proper learning rate scaling
"""

import time
import numpy as np
import torch
import gymnasium as gym
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from model import ActorCritic
from ppo import PPO
from eval_cost import calculate_costs
from controllers import PIDController

# Register environment
try:
    register(
        id='CartLatAccel-Jerk-v1',
        entry_point='jerk_env:CartLatAccelJerkEnv',
        max_episode_steps=500,
    )
except:
    pass


def eval_model(model, device='cpu', n_rollouts=10):
    """Evaluate a model"""
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1, 
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    costs = []
    for _ in range(n_rollouts):
        state, _ = env.reset()
        actions, targets, actuals = [], [], []
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            target_lataccel = (target_pos - pos) * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            state_tensor = torch.FloatTensor(state).to(device)
            action = model.get_action(state_tensor, deterministic=True)
            
            next_state, _, terminated, truncated, info = env.step(np.array([action]))
            
            actual = info['noisy_action'] if 'noisy_action' in info else action
            actions.append(action)
            targets.append(target_lataccel)
            actuals.append(actual)
            
            state = next_state
            if terminated or truncated:
                break
        
        cost = calculate_costs(np.array(actuals), np.array(targets), dt=0.02)
        costs.append(cost['total_cost'])
    
    return np.mean(costs), np.std(costs)


def train_with_lr(hidden_size, n_layers, lr, steps=200000, eval_freq=50000):
    """Train with specific learning rate"""
    
    print(f"\nTraining {hidden_size}x{n_layers} with lr={lr:.0e}")
    print("-" * 60)
    
    # Create environment and model
    train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000,
                        jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    model = ActorCritic(3, 
                       {"pi": [hidden_size]*n_layers, "vf": [hidden_size]*n_layers},
                       1)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Train
    ppo = PPO(train_env, model, env_bs=1000, device='cpu', lr=lr)
    
    results = []
    checkpoints = list(range(eval_freq, steps + 1, eval_freq))
    
    for checkpoint in checkpoints:
        ppo.train(eval_freq)
        
        # Evaluate
        cost_mean, cost_std = eval_model(ppo.model.actor, device='cpu', n_rollouts=10)
        
        results.append({
            'step': checkpoint,
            'cost_mean': cost_mean,
            'cost_std': cost_std,
        })
        
        print(f"  {checkpoint:>6,}: {cost_mean:>8,.0f} ± {cost_std:>5,.0f}")
    
    best = min(results, key=lambda r: r['cost_mean'])
    return best, results


def main():
    print("=" * 80)
    print("TESTING: Proper Learning Rate Scaling")
    print("=" * 80)
    
    # Get PID baseline
    print("\nPID Baseline:")
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    pid_costs = []
    for _ in range(10):
        state, _ = env.reset()
        pid.reset()
        actions, targets, actuals = [], [], []
        current = 0.0
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            target = (target_pos - pos) * 10.0
            target = np.clip(target, -1.0, 1.0)
            
            action = pid.update(target, current, state, None)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, _, terminated, truncated, info = env.step(np.array([action]))
            
            actual = info['noisy_action'] if 'noisy_action' in info else action
            actions.append(action)
            targets.append(target)
            actuals.append(actual)
            
            current = actual
            state = next_state
            if terminated or truncated:
                break
        
        cost = calculate_costs(np.array(actuals), np.array(targets), dt=0.02)
        pid_costs.append(cost['total_cost'])
    
    pid_cost = np.mean(pid_costs)
    print(f"  {pid_cost:.0f}\n")
    
    # Test different configurations
    print("=" * 80)
    print("EXPERIMENTS")
    print("=" * 80)
    
    experiments = [
        # (hidden, layers, lr)
        (256, 4, 3e-4),   # Baseline
        (512, 4, 3e-4),   # Original (fails)
        (512, 4, 1e-4),   # Fixed LR
        (512, 4, 5e-5),   # Even smaller
        (384, 4, 2e-4),   # Middle ground
    ]
    
    results = []
    
    for hidden_size, n_layers, lr in experiments:
        try:
            best, history = train_with_lr(hidden_size, n_layers, lr, steps=200000, eval_freq=50000)
            
            result = {
                'config': f"{hidden_size}x{n_layers}",
                'lr': lr,
                'best_cost': best['cost_mean'],
                'best_step': best['step'],
                'vs_pid': best['cost_mean'] / pid_cost,
            }
            results.append(result)
            
            print(f"  → Best: {best['cost_mean']:.0f} at {best['step']:,} steps ({result['vs_pid']:.2f}x vs PID)")
            
            # Check if we beat PID
            if result['vs_pid'] < 1.0:
                print(f"\n🎉🎉🎉 WE BEAT PID! 🎉🎉🎉")
                print(f"Configuration: {result['config']} with lr={lr:.0e}")
                break
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nPID Baseline: {pid_cost:.0f}\n")
    print("Config  | LR      | Best Cost | Best Step | vs PID  | Winner")
    print("-" * 80)
    
    for r in results:
        marker = "🏆" if r['vs_pid'] < 1.0 else "⭐" if r['vs_pid'] < 1.3 else "✓" if r['vs_pid'] < 2.0 else ""
        print(f"{r['config']:<7} | {r['lr']:.0e} | {r['best_cost']:>9,.0f} | "
              f"{r['best_step']:>9,} | {r['vs_pid']:>6.2f}x | {marker}")
    
    if results:
        best = min(results, key=lambda x: x['vs_pid'])
        print(f"\n🎯 BEST: {best['config']} with lr={best['lr']:.0e}")
        print(f"   Cost: {best['best_cost']:.0f} ({best['vs_pid']:.2f}x vs PID)")
        
        if best['vs_pid'] < 1.0:
            improvement = ((pid_cost - best['best_cost']) / pid_cost) * 100
            print(f"   🎉 BEATS PID by {improvement:.1f}%!")
        elif best['vs_pid'] < 1.2:
            gap = (best['best_cost'] - pid_cost)
            print(f"   💪 Very close! Only {gap:.0f} cost away from PID")


if __name__ == "__main__":
    main()

