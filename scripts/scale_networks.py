"""
Scale up: Test increasingly large networks to find the limit
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


def test_network_size(hidden_size, n_layers, max_evals=300000, jerk_weight=0.005, action_weight=0.0005, 
                      lr=3e-4, device='cpu', eval_rollouts=10):
    """Test a specific network configuration"""
    
    print(f"\n{'='*80}")
    print(f"TESTING: {hidden_size} hidden units x {n_layers} layers")
    print(f"{'='*80}")
    
    # Create environments
    train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000,
                        jerk_penalty_weight=jerk_weight,
                        action_penalty_weight=action_weight)
    eval_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                       jerk_penalty_weight=jerk_weight,
                       action_penalty_weight=action_weight)
    
    # Create model
    hidden_dims_pi = [hidden_size] * n_layers
    hidden_dims_vf = [hidden_size] * n_layers
    
    model = ActorCritic(
        train_env.observation_space.shape[-1],
        {"pi": hidden_dims_pi, "vf": hidden_dims_vf},
        train_env.action_space.shape[-1]
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Create PPO trainer
    ppo = PPO(train_env, model, env_bs=1000, device=device, lr=lr)
    
    # Training checkpoints
    checkpoints = [50000, 100000, 150000, 200000, 250000, 300000]
    checkpoints = [c for c in checkpoints if c <= max_evals]
    
    results = []
    current_steps = 0
    best_cost = float('inf')
    best_checkpoint = 0
    
    start_time = time.time()
    
    for checkpoint in checkpoints:
        steps_to_train = checkpoint - current_steps
        
        print(f"\nTraining {current_steps:,} → {checkpoint:,}...", end=' ', flush=True)
        ppo.train(steps_to_train)
        current_steps = checkpoint
        
        # Evaluate
        checkpoint_results = []
        for _ in range(eval_rollouts):
            costs = rollout_and_evaluate(eval_env, ppo.model.actor, device=device)
            checkpoint_results.append(costs)
        
        avg_cost = np.mean([c['total_cost'] for c in checkpoint_results])
        
        result = {
            'timesteps': checkpoint,
            'total_cost': avg_cost,
            'lataccel_cost': np.mean([c['lataccel_cost'] for c in checkpoint_results]),
            'jerk_cost': np.mean([c['jerk_cost'] for c in checkpoint_results]),
        }
        results.append(result)
        
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_checkpoint = checkpoint
            print(f"✓ {avg_cost:.0f} ⭐ NEW BEST!")
        else:
            print(f"✓ {avg_cost:.0f}")
    
    train_time = time.time() - start_time
    
    return {
        'hidden_size': hidden_size,
        'n_layers': n_layers,
        'n_params': n_params,
        'best_cost': best_cost,
        'best_checkpoint': best_checkpoint,
        'train_time': train_time,
        'all_results': results
    }


def main():
    print("=" * 80)
    print("NETWORK SCALING EXPERIMENT")
    print("=" * 80)
    print("\nTesting increasingly large networks to find performance limits")
    print()
    
    # Get PID baseline
    print("Evaluating PID baseline...")
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    eval_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                       jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    pid_results = []
    for _ in range(10):
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
    
    pid_cost = np.mean([r['total_cost'] for r in pid_results])
    print(f"PID Baseline: {pid_cost:.2f}")
    print()
    
    # Test different network sizes
    configs = [
        # (hidden_size, n_layers)
        (256, 4),   # Already tested: 1.61x
        (512, 4),   # 4x more parameters
        (512, 6),   # Deeper
        (1024, 4),  # Even wider
        (1024, 6),  # Wide and deep
    ]
    
    all_results = []
    
    for hidden_size, n_layers in configs:
        try:
            result = test_network_size(
                hidden_size=hidden_size,
                n_layers=n_layers,
                max_evals=300000,
                jerk_weight=0.005,
                action_weight=0.0005,
                lr=3e-4,  # Same LR for all
                device='cpu',
                eval_rollouts=10
            )
            result['vs_pid'] = result['best_cost'] / pid_cost
            all_results.append(result)
            
            print(f"\nResult: {result['best_cost']:.0f} ({result['vs_pid']:.2f}x vs PID)")
            
            # Check if we beat PID!
            if result['vs_pid'] < 1.0:
                print(f"🎉🎉🎉 WE BEAT PID! 🎉🎉🎉")
                break  # Stop if we beat PID
                
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Continuing to next configuration...")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("SCALING RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"PID Baseline: {pid_cost:.2f}")
    print()
    print("Network          | Params    | Best Cost | vs PID  | Time    | Winner")
    print("-" * 80)
    
    for r in all_results:
        marker = "🏆" if r['vs_pid'] < 1.0 else "⭐" if r['vs_pid'] < 1.5 else "✓" if r['vs_pid'] < 2.0 else ""
        print(f"{r['hidden_size']:>4}x{r['n_layers']:1} layers    | {r['n_params']:>9,} | {r['best_cost']:>9,.0f} | "
              f"{r['vs_pid']:>6.2f}x | {r['train_time']:>6.1f}s | {marker}")
    
    print("=" * 80)
    
    # Find best
    if all_results:
        best = min(all_results, key=lambda x: x['vs_pid'])
        
        print(f"\nBEST CONFIGURATION:")
        print(f"  Network: {best['hidden_size']}x{best['n_layers']} layers")
        print(f"  Parameters: {best['n_params']:,}")
        print(f"  Cost: {best['best_cost']:.2f}")
        print(f"  vs PID: {best['vs_pid']:.3f}x")
        
        if best['vs_pid'] < 1.0:
            improvement = ((pid_cost - best['best_cost']) / pid_cost) * 100
            print(f"\n🎉 PPO BEAT PID BY {improvement:.1f}%! 🎉")
        elif best['vs_pid'] < 1.5:
            print(f"\n🎯 Very close to PID!")
        elif best['vs_pid'] < 2.0:
            print(f"\n✓ Within 2x of PID - competitive!")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/scaling_experiment_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, "scaling_results.json"), 'w') as f:
            json.dump({
                'pid_baseline': float(pid_cost),
                'results': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in r.items() if k != 'all_results'} 
                          for r in all_results],
                'best': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in best.items() if k != 'all_results'}
            }, f, indent=2)
        
        print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    main()

