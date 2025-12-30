import numpy as np
import gymnasium as gym
import gym_cartlataccel
from controllers import PIDController
from eval_cost import calculate_costs
import argparse

def rollout_controller(env, controller, max_steps=500):
  """
  Rollout a controller in the environment
  
  Returns:
    states, actions, target_lataccels, actual_lataccels
  """
  states = []
  actions = []
  target_lataccels = []
  actual_lataccels = []
  
  state, _ = env.reset()
  controller.reset()
  
  current_lataccel = 0.0  # start with zero acceleration
  
  for step in range(max_steps):
    # Extract position, velocity, target position from state
    pos, vel, target_pos = state
    
    # Calculate target lataccel (desired acceleration to reach target)
    # Simple target: acceleration needed to reach target position
    # For PID, we want to track the target position via acceleration control
    # Target lataccel could be computed from position error
    pos_error = target_pos - pos
    target_lataccel = pos_error * 10.0  # Simple proportional target (tunable)
    target_lataccel = np.clip(target_lataccel, -1.0, 1.0)  # clip to action space
    
    # Get controller action
    action = controller.update(target_lataccel, current_lataccel, state, None)
    action = np.clip(action, -1.0, 1.0)  # clip to action space
    
    # Step environment
    next_state, reward, terminated, truncated, info = env.step(np.array([action]))
    
    # Track actual lataccel (from info or action)
    actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
    
    states.append(state)
    actions.append(action)
    target_lataccels.append(target_lataccel)
    actual_lataccels.append(actual_lataccel)
    
    current_lataccel = actual_lataccel
    state = next_state
    
    if terminated or truncated:
      break
  
  return np.array(states), np.array(actions), np.array(target_lataccels), np.array(actual_lataccels)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--p", type=float, default=0.195, help="P gain")
  parser.add_argument("--i", type=float, default=0.100, help="I gain")
  parser.add_argument("--d", type=float, default=-0.053, help="D gain")
  parser.add_argument("--noise_mode", default=None, help="Noise mode (None, REALISTIC, HIGH)")
  parser.add_argument("--n_rollouts", type=int, default=5, help="Number of evaluation rollouts")
  parser.add_argument("--render", action='store_true', help="Render the environment")
  args = parser.parse_args()
  
  # Create environment
  render_mode = "human" if args.render else None
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1, render_mode=render_mode)
  
  # Create PID controller
  controller = PIDController(p=args.p, i=args.i, d=args.d)
  
  print(f"Evaluating PID Controller (P={args.p}, I={args.i}, D={args.d})")
  print(f"Noise mode: {args.noise_mode}")
  print(f"Running {args.n_rollouts} rollouts...\n")
  
  all_costs = []
  
  for rollout_idx in range(args.n_rollouts):
    states, actions, target_lataccels, actual_lataccels = rollout_controller(env, controller)
    
    # Calculate costs
    costs = calculate_costs(actual_lataccels, target_lataccels, dt=env.tau)
    all_costs.append(costs)
    
    print(f"Rollout {rollout_idx + 1}:")
    print(f"  lataccel_cost: {costs['lataccel_cost']:.4f}")
    print(f"  jerk_cost:     {costs['jerk_cost']:.4f}")
    print(f"  total_cost:    {costs['total_cost']:.4f}")
    print()
  
  # Calculate average costs
  avg_lataccel = np.mean([c['lataccel_cost'] for c in all_costs])
  avg_jerk = np.mean([c['jerk_cost'] for c in all_costs])
  avg_total = np.mean([c['total_cost'] for c in all_costs])
  
  print("=" * 50)
  print(f"Average over {args.n_rollouts} rollouts:")
  print(f"  lataccel_cost: {avg_lataccel:.4f}")
  print(f"  jerk_cost:     {avg_jerk:.4f}")
  print(f"  total_cost:    {avg_total:.4f}")
  print("=" * 50)
  
  env.close()

if __name__ == "__main__":
  main()

