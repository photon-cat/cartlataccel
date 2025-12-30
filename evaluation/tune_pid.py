import numpy as np
import gymnasium as gym
import gym_cartlataccel
from controllers import PIDController
from eval_cost import calculate_costs
import argparse

def rollout_controller(env, controller, max_steps=500):
  """Rollout a controller and return costs"""
  states, actions, target_lataccels, actual_lataccels = [], [], [], []
  
  state, _ = env.reset()
  controller.reset()
  current_lataccel = 0.0
  
  for step in range(max_steps):
    pos, vel, target_pos = state
    pos_error = target_pos - pos
    target_lataccel = pos_error * 10.0
    target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
    
    action = controller.update(target_lataccel, current_lataccel, state, None)
    action = np.clip(action, -1.0, 1.0)
    
    next_state, reward, terminated, truncated, info = env.step(np.array([action]))
    actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
    
    actions.append(action)
    target_lataccels.append(target_lataccel)
    actual_lataccels.append(actual_lataccel)
    
    current_lataccel = actual_lataccel
    state = next_state
    
    if terminated or truncated:
      break
  
  costs = calculate_costs(np.array(actual_lataccels), np.array(target_lataccels), dt=env.unwrapped.tau)
  return costs['total_cost']

def grid_search_pid(env, p_range, i_range, d_range, n_trials=3):
  """Grid search over PID parameters"""
  results = []
  
  for p in p_range:
    for i in i_range:
      for d in d_range:
        costs = []
        for _ in range(n_trials):
          controller = PIDController(p=p, i=i, d=d)
          cost = rollout_controller(env, controller)
          costs.append(cost)
        
        avg_cost = np.mean(costs)
        results.append({
          'p': p, 'i': i, 'd': d,
          'avg_cost': avg_cost,
          'std_cost': np.std(costs)
        })
        print(f"P={p:.3f}, I={i:.3f}, D={d:.3f} → cost={avg_cost:.2f} ± {np.std(costs):.2f}")
  
  return results

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_trials", type=int, default=3, help="Trials per config")
  parser.add_argument("--noise_mode", default=None, help="Noise mode")
  parser.add_argument("--quick", action='store_true', help="Quick search (coarse grid)")
  args = parser.parse_args()
  
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1)
  
  if args.quick:
    print("Running QUICK grid search...")
    p_range = [0.1, 0.2, 0.3]
    i_range = [0.05, 0.1, 0.15]
    d_range = [-0.1, -0.05, 0.0]
  else:
    print("Running FINE grid search...")
    p_range = np.linspace(0.15, 0.25, 5)
    i_range = np.linspace(0.08, 0.12, 5)
    d_range = np.linspace(-0.08, -0.03, 5)
  
  print(f"Testing {len(p_range) * len(i_range) * len(d_range)} configurations")
  print(f"with {args.n_trials} trials each\n")
  
  results = grid_search_pid(env, p_range, i_range, d_range, n_trials=args.n_trials)
  
  # Find best
  best = min(results, key=lambda x: x['avg_cost'])
  
  print("\n" + "=" * 60)
  print("BEST PID GAINS:")
  print(f"  P = {best['p']:.4f}")
  print(f"  I = {best['i']:.4f}")
  print(f"  D = {best['d']:.4f}")
  print(f"  Cost = {best['avg_cost']:.2f} ± {best['std_cost']:.2f}")
  print("=" * 60)
  
  # Top 5
  print("\nTop 5 configurations:")
  sorted_results = sorted(results, key=lambda x: x['avg_cost'])
  for i, r in enumerate(sorted_results[:5], 1):
    print(f"{i}. P={r['p']:.4f}, I={r['i']:.4f}, D={r['d']:.4f} → {r['avg_cost']:.2f}")
  
  env.close()

if __name__ == "__main__":
  main()

