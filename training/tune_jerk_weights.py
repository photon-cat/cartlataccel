"""
Grid search to find optimal jerk and action penalty weights
"""

import numpy as np
import torch
import gymnasium as gym
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from model import ActorCritic
from ppo import PPO
from eval_cost import calculate_costs
import itertools

# Register environment
try:
    register(
        id='CartLatAccel-Jerk-v1',
        entry_point='jerk_env:CartLatAccelJerkEnv',
        max_episode_steps=500,
    )
except:
    pass


def quick_train_and_eval(jerk_weight, action_weight, train_steps=30000, n_eval=3, device='cpu'):
    """Quick training and evaluation"""
    # Create environment
    train_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1000,
                        jerk_penalty_weight=jerk_weight,
                        action_penalty_weight=action_weight)
    eval_env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                       jerk_penalty_weight=jerk_weight,
                       action_penalty_weight=action_weight)
    
    # Create and train model
    model = ActorCritic(
        train_env.observation_space.shape[-1],
        {"pi": [32], "vf": [32]},
        train_env.action_space.shape[-1]
    )
    ppo = PPO(train_env, model, env_bs=1000, device=device)
    ppo.train(train_steps)
    
    # Evaluate
    results = []
    for _ in range(n_eval):
        states, actions, target_lataccels, actual_lataccels = [], [], [], []
        state, _ = eval_env.reset()
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            pos_error = target_pos - pos
            target_lataccel = pos_error * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            state_tensor = torch.FloatTensor(state).to(device)
            action = ppo.model.actor.get_action(state_tensor, deterministic=True)
            
            next_state, reward, terminated, truncated, info = eval_env.step(np.array([action]))
            
            actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
            
            actions.append(action)
            target_lataccels.append(target_lataccel)
            actual_lataccels.append(actual_lataccel)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        costs = calculate_costs(np.array(actual_lataccels), np.array(target_lataccels), dt=0.02)
        results.append(costs)
    
    avg_cost = {
        'total_cost': np.mean([r['total_cost'] for r in results]),
        'lataccel_cost': np.mean([r['lataccel_cost'] for r in results]),
        'jerk_cost': np.mean([r['jerk_cost'] for r in results]),
    }
    
    return avg_cost


def main():
    print("=" * 80)
    print("GRID SEARCH: OPTIMAL JERK & ACTION PENALTY WEIGHTS")
    print("=" * 80)
    print()
    
    # Define search grid
    jerk_weights = [0.001, 0.005, 0.01, 0.05, 0.1]
    action_weights = [0.0001, 0.0005, 0.001, 0.005]
    
    print(f"Testing {len(jerk_weights)} x {len(action_weights)} = {len(jerk_weights)*len(action_weights)} configurations")
    print(f"Jerk weights: {jerk_weights}")
    print(f"Action weights: {action_weights}")
    print(f"Training steps per config: 30,000")
    print(f"Evaluation rollouts per config: 3")
    print()
    print("This will take approximately 10-15 minutes...")
    print()
    
    results = []
    
    for i, (jw, aw) in enumerate(itertools.product(jerk_weights, action_weights), 1):
        print(f"[{i}/{len(jerk_weights)*len(action_weights)}] Testing jerk={jw:.4f}, action={aw:.4f}...")
        
        try:
            cost_metrics = quick_train_and_eval(jw, aw)
            
            result = {
                'jerk_weight': jw,
                'action_weight': aw,
                **cost_metrics
            }
            results.append(result)
            
            print(f"     total_cost: {cost_metrics['total_cost']:>10,.0f}")
            
        except Exception as e:
            print(f"     ERROR: {e}")
    
    # Sort by total cost
    results.sort(key=lambda x: x['total_cost'])
    
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    print()
    print("Rank | Jerk Weight | Action Weight | Total Cost | Lataccel | Jerk Cost")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:>4} | {r['jerk_weight']:>11.4f} | {r['action_weight']:>13.4f} | "
              f"{r['total_cost']:>10,.0f} | {r['lataccel_cost']:>8.2f} | {r['jerk_cost']:>9,.0f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    best = results[0]
    print(f"\nBest Configuration:")
    print(f"  Jerk weight:   {best['jerk_weight']}")
    print(f"  Action weight: {best['action_weight']}")
    print(f"  Total cost:    {best['total_cost']:,.0f}")
    print(f"  vs PID (2,086): {best['total_cost']/2086:.1f}x worse")
    
    # Analyze trends
    high_jerk = [r for r in results if r['jerk_weight'] >= 0.01]
    low_jerk = [r for r in results if r['jerk_weight'] < 0.01]
    
    if high_jerk and low_jerk:
        avg_high = np.mean([r['total_cost'] for r in high_jerk])
        avg_low = np.mean([r['total_cost'] for r in low_jerk])
        
        print(f"\nTrend Analysis:")
        print(f"  High jerk weight (≥0.01): avg cost = {avg_high:,.0f}")
        print(f"  Low jerk weight (<0.01):  avg cost = {avg_low:,.0f}")
        
        if avg_low < avg_high:
            print(f"  → Lower jerk weights perform better")
        else:
            print(f"  → Higher jerk weights perform better")
    
    # Recommendation
    print(f"\nRecommendation:")
    print(f"  Use: --jerk_weight {best['jerk_weight']} --action_weight {best['action_weight']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

