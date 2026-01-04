#!/usr/bin/env python3
"""
train_ppo.py - Train PPO on CartLatAccel

Usage:
  python train_ppo.py
  python train_ppo.py --max_evals 50000 --render
"""
import argparse
import os
import time
import gymnasium as gym
import numpy as np
import torch
import gym_cartlataccel
from model import ActorCritic
from ppo import PPO

def main():
  parser = argparse.ArgumentParser(description='Train PPO on CartLatAccel')
  parser.add_argument('--max_evals', type=int, default=30000)
  parser.add_argument('--env_bs', type=int, default=1000)
  parser.add_argument('--noise', type=float, default=0.5, help='Noise level 0-1 (default: 0.5)')
  parser.add_argument('--render', action='store_true', help='Show eval rollout after training')
  parser.add_argument('--out', type=str, default='models/ppo.pt', help='Output model path')
  args = parser.parse_args()

  print(f'Training PPO with max_evals={args.max_evals}')
  start = time.time()

  # Training env (batched, no render)
  env = gym.make('CartLatAccel-v1', noise=args.noise, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {'pi': [32], 'vf': [32]}, env.action_space.shape[-1])
  ppo = PPO(env, model, env_bs=args.env_bs)
  best_model, hist = ppo.train(args.max_evals)
  train_time = time.time() - start

  print(f'Training done in {train_time:.2f}s')

  # Save model
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  torch.save(best_model, args.out)
  print(f'Model saved to {args.out}')

  # Eval rollout
  render_mode = 'human' if args.render else None
  eval_env = gym.make('CartLatAccel-v1', noise=args.noise, env_bs=1, render_mode=render_mode)
  states, actions, rewards, dones, next_state = ppo.rollout(eval_env, best_model, max_steps=500, deterministic=True, device=ppo.device)
  
  print(f'\n=== Eval ===')
  print(f'Reward: {sum(rewards):.3f}')
  print(f'Mean |action|: {np.mean(np.abs(actions)):.3f}')

  eval_env.close()

if __name__ == '__main__':
  main()

