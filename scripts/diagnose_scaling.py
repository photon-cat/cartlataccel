"""
Verify 256x4 consistency and diagnose why larger networks fail
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


def train_and_track(hidden_size, n_layers, steps=200000, eval_freq=50000):
    """Train and track metrics throughout"""
    
    print(f"\n{'='*80}")
    print(f"Training {hidden_size}x{n_layers}")
    print(f"{'='*80}")
    
    # Create environment and model
    train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000,
                        jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    model = ActorCritic(3, 
                       {"pi": [hidden_size]*n_layers, "vf": [hidden_size]*n_layers},
                       1)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}\n")
    
    # Train
    ppo = PPO(train_env, model, env_bs=1000, device='cpu', lr=3e-4)
    
    results = []
    checkpoints = list(range(eval_freq, steps + 1, eval_freq))
    
    for checkpoint in checkpoints:
        print(f"Training to {checkpoint:,}... ", end='', flush=True)
        ppo.train(eval_freq)
        
        # Evaluate
        cost_mean, cost_std = eval_model(ppo.model.actor, device='cpu', n_rollouts=10)
        
        # Check for gradient issues
        grad_norms = []
        for name, param in ppo.model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        max_grad_norm = np.max(grad_norms) if grad_norms else 0
        
        # Check weight magnitudes
        weight_norms = []
        for name, param in ppo.model.named_parameters():
            weight_norms.append(param.norm().item())
        
        avg_weight_norm = np.mean(weight_norms)
        
        result = {
            'step': checkpoint,
            'cost_mean': cost_mean,
            'cost_std': cost_std,
            'avg_grad_norm': avg_grad_norm,
            'max_grad_norm': max_grad_norm,
            'avg_weight_norm': avg_weight_norm,
        }
        results.append(result)
        
        print(f"cost={cost_mean:.0f}±{cost_std:.0f}, grad={avg_grad_norm:.4f}, weight={avg_weight_norm:.2f}")
    
    return results


def main():
    print("=" * 80)
    print("VERIFY 256x4 AND DIAGNOSE SCALING FAILURE")
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
    pid_std = np.std(pid_costs)
    print(f"  Cost: {pid_cost:.0f} ± {pid_std:.0f}\n")
    
    # Test 256x4 THREE times to verify consistency
    print("=" * 80)
    print("CONSISTENCY CHECK: 3 runs of 256x4")
    print("=" * 80)
    
    run_256x4 = []
    for run in range(3):
        print(f"\n### RUN {run + 1}/3 ###")
        results = train_and_track(256, 4, steps=200000, eval_freq=50000)
        best_cost = min(r['cost_mean'] for r in results)
        run_256x4.append(best_cost)
        print(f"Best: {best_cost:.0f} ({best_cost/pid_cost:.2f}x vs PID)")
    
    print(f"\n256x4 Results across 3 runs:")
    print(f"  Mean: {np.mean(run_256x4):.0f}")
    print(f"  Std:  {np.std(run_256x4):.0f}")
    print(f"  Min:  {np.min(run_256x4):.0f}")
    print(f"  Max:  {np.max(run_256x4):.0f}")
    print(f"  vs PID: {np.mean(run_256x4)/pid_cost:.2f}x ± {np.std(run_256x4)/pid_cost:.2f}x")
    
    # Now test 512x4 with detailed tracking
    print("\n" + "=" * 80)
    print("DETAILED DIAGNOSIS: 512x4")
    print("=" * 80)
    
    results_512 = train_and_track(512, 4, steps=200000, eval_freq=50000)
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\n256x4 Training Progression (from first run):")
    results_256 = train_and_track(256, 4, steps=200000, eval_freq=50000)
    
    print("\nStep | 256x4 Cost | 512x4 Cost | Difference")
    print("-" * 60)
    for r256, r512 in zip(results_256, results_512):
        diff = r512['cost_mean'] - r256['cost_mean']
        print(f"{r256['step']:>5,} | {r256['cost_mean']:>10,.0f} | {r512['cost_mean']:>10,.0f} | {diff:>+10,.0f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    # Check if 512x4 is overfitting
    best_256_idx = min(range(len(results_256)), key=lambda i: results_256[i]['cost_mean'])
    best_512_idx = min(range(len(results_512)), key=lambda i: results_512[i]['cost_mean'])
    
    print(f"\n256x4 best at step {results_256[best_256_idx]['step']:,}: {results_256[best_256_idx]['cost_mean']:.0f}")
    print(f"512x4 best at step {results_512[best_512_idx]['step']:,}: {results_512[best_512_idx]['cost_mean']:.0f}")
    
    # Check gradient magnitudes
    print(f"\nAverage gradient norms:")
    print(f"  256x4: {np.mean([r['avg_grad_norm'] for r in results_256]):.4f}")
    print(f"  512x4: {np.mean([r['avg_grad_norm'] for r in results_512]):.4f}")
    
    print(f"\nAverage weight norms:")
    print(f"  256x4: {np.mean([r['avg_weight_norm'] for r in results_256]):.2f}")
    print(f"  512x4: {np.mean([r['avg_weight_norm'] for r in results_512]):.2f}")
    
    # Hypothesis
    print("\n" + "=" * 80)
    print("HYPOTHESIS")
    print("=" * 80)
    
    if results_512[0]['cost_mean'] > 10000:
        print("\n❌ 512x4 is performing poorly from the start!")
        print("   Possible causes:")
        print("   1. Learning rate too high for larger network")
        print("   2. Initialization issues")
        print("   3. Optimization difficulty with more parameters")
    else:
        print("\n✓ 512x4 starts reasonably, then degrades")
        print("  This suggests OVERFITTING:")
        print("  - Larger network memorizes training data")
        print("  - Doesn't generalize to evaluation")
        print("  - Needs regularization or smaller learning rate")


if __name__ == "__main__":
    main()

