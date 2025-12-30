import numpy as np
import torch
import gymnasium as gym
import gym_cartlataccel
import argparse
from model import ActorCritic
from eval_cost import calculate_costs
from ppo import PPO

def rollout_ppo_with_costs(env, model, max_steps=500, device='cpu'):
  """
  Rollout PPO model and track costs
  
  Returns:
    states, actions, target_lataccels, actual_lataccels
  """
  states = []
  actions = []
  target_lataccels = []
  actual_lataccels = []
  
  state, _ = env.reset()
  current_lataccel = 0.0
  
  for step in range(max_steps):
    state_tensor = torch.FloatTensor(state).to(device)
    action = model.get_action(state_tensor, deterministic=True)
    
    # Extract position, velocity, target from state
    pos, vel, target_pos = state
    
    # Calculate target lataccel (same as PID for fair comparison)
    pos_error = target_pos - pos
    target_lataccel = pos_error * 10.0
    target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
    
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Track actual lataccel
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
  parser.add_argument("--model_path", type=str, default=None, help="Path to saved PPO model")
  parser.add_argument("--noise_mode", default=None, help="Noise mode (None, REALISTIC, HIGH)")
  parser.add_argument("--n_rollouts", type=int, default=5, help="Number of evaluation rollouts")
  parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
  parser.add_argument("--render", action='store_true', help="Render the environment")
  parser.add_argument("--train_first", action='store_true', help="Train a new model first")
  parser.add_argument("--max_evals", type=int, default=10000, help="Max training evals if training")
  args = parser.parse_args()
  
  # Create environment
  render_mode = "human" if args.render else None
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1, render_mode=render_mode)
  
  # Load or train model
  if args.train_first:
    print("Training PPO model first...")
    train_env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1000)
    model = ActorCritic(train_env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, train_env.action_space.shape[-1])
    ppo = PPO(train_env, model, env_bs=1000, device=args.device)
    best_model, hist = ppo.train(args.max_evals)
    print("Training complete!\n")
  elif args.model_path:
    print(f"Loading model from {args.model_path}")
    best_model = torch.load(args.model_path, map_location=args.device)
  else:
    print("Training a quick PPO model...")
    train_env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1000)
    model = ActorCritic(train_env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, train_env.action_space.shape[-1])
    ppo = PPO(train_env, model, env_bs=1000, device=args.device)
    best_model, hist = ppo.train(10000)
    print("Training complete!\n")
  
  print(f"Evaluating PPO Model")
  print(f"Noise mode: {args.noise_mode}")
  print(f"Running {args.n_rollouts} rollouts...\n")
  
  all_costs = []
  
  for rollout_idx in range(args.n_rollouts):
    states, actions, target_lataccels, actual_lataccels = rollout_ppo_with_costs(
      env, best_model, max_steps=500, device=args.device
    )
    
    # Calculate costs
    costs = calculate_costs(actual_lataccels, target_lataccels, dt=env.unwrapped.tau)
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

