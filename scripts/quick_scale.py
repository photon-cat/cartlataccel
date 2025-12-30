"""
Quick scaling test - just test a few key sizes efficiently
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


def quick_eval(model, device='cpu', n_rollouts=5):
    """Quick evaluation"""
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
    
    return np.mean(costs)


def quick_test(hidden_size, n_layers, steps=200000):
    """Quick test of network size"""
    
    print(f"\n{hidden_size}x{n_layers}: ", end='', flush=True)
    
    # Create environment and model
    train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000,
                        jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    model = ActorCritic(3, 
                       {"pi": [hidden_size]*n_layers, "vf": [hidden_size]*n_layers},
                       1)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} params | ", end='', flush=True)
    
    # Train
    start = time.time()
    ppo = PPO(train_env, model, env_bs=1000, device='cpu', lr=3e-4)
    ppo.train(steps)
    train_time = time.time() - start
    
    # Evaluate
    cost = quick_eval(ppo.model.actor, device='cpu', n_rollouts=5)
    
    print(f"cost={cost:.0f} | time={train_time:.1f}s")
    
    return {
        'size': f"{hidden_size}x{n_layers}",
        'params': n_params,
        'cost': cost,
        'time': train_time
    }


def main():
    print("=" * 80)
    print("QUICK SCALING TEST")
    print("=" * 80)
    print("\nTesting key network sizes at 200k steps each")
    
    # PID baseline
    print("\nPID Baseline: ", end='', flush=True)
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    pid_costs = []
    for _ in range(5):
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
    print(f"{pid_cost:.0f}")
    
    # Test configurations (most promising)
    configs = [
        (128, 3),   # Small but deep
        (256, 4),   # Known: 1.61x
        (384, 4),   # Between 256 and 512
        (512, 4),   # 4x params of 256x4
        (512, 5),   # Deeper 512
    ]
    
    results = []
    
    print("\nTraining networks...")
    for hidden_size, n_layers in configs:
        try:
            result = quick_test(hidden_size, n_layers, steps=200000)
            result['vs_pid'] = result['cost'] / pid_cost
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPID: {pid_cost:.0f}")
    print()
    print("Network  | Params     | Cost   | vs PID  | Time   | Status")
    print("-" * 80)
    
    for r in results:
        status = "🏆 BEATS PID!" if r['vs_pid'] < 1.0 else "⭐" if r['vs_pid'] < 1.5 else "✓"
        print(f"{r['size']:<8} | {r['params']:>10,} | {r['cost']:>6,.0f} | {r['vs_pid']:>6.2f}x | {r['time']:>5.0f}s | {status}")
    
    if results:
        best = min(results, key=lambda x: x['vs_pid'])
        print(f"\nBest: {best['size']} → {best['cost']:.0f} ({best['vs_pid']:.2f}x vs PID)")
        
        if best['vs_pid'] < 1.0:
            print("🎉 WE BEAT PID! 🎉")


if __name__ == "__main__":
    main()

