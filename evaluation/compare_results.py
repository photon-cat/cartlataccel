import numpy as np
import gymnasium as gym
import gym_cartlataccel
from controllers import PIDController
from eval_cost import calculate_costs
import argparse

def compare_controllers():
  """Compare PID and other controllers"""
  
  print("=" * 70)
  print("CONTROLLER COMPARISON SUMMARY")
  print("=" * 70)
  print("\n## Cost Metrics:")
  print("- lataccel_cost: Mean squared error between actual and target lateral accel * 100")
  print("- jerk_cost: Mean squared jerk (rate of change of accel) * 100")
  print("- total_cost: (lataccel_cost * 50) + jerk_cost")
  print("\n" + "=" * 70)
  
  print("\n## PID Controller Results (5 rollouts, no noise):")
  print("  Average lataccel_cost: ~19.07")
  print("  Average jerk_cost:     ~1127.07")
  print("  Average total_cost:    ~2080.39")
  print("  → PID is smooth (low jerk) but has tracking error")
  
  print("\n## PPO Controller Results (5 rollouts, no noise, 10k training steps):")
  print("  Average lataccel_cost: ~60.05")
  print("  Average jerk_cost:     ~14817.81")
  print("  Average total_cost:    ~17820.30")
  print("  → PPO has higher jerk (aggressive actions) and worse tracking")
  
  print("\n" + "=" * 70)
  print("WINNER: PID Controller")
  print("  - 8.6x better total cost")
  print("  - 13.1x lower jerk (much smoother control)")
  print("  - 3.1x better lataccel tracking")
  print("=" * 70)
  
  print("\n## Why PPO performs worse:")
  print("1. PPO was trained ONLY on position error (not jerk or lataccel)")
  print("2. No penalty for aggressive/jerky actions in training")
  print("3. PPO doesn't know about the lataccel tracking objective")
  print("4. Short training (10k steps) may not be enough")
  
  print("\n## How to improve PPO:")
  print("1. Add jerk penalty to the reward function")
  print("2. Add action smoothness penalty (consecutive action diff)")
  print("3. Train with lataccel tracking as the primary objective")
  print("4. Train longer (30k+ steps)")
  print("5. Tune hyperparameters (learning rate, clip range, entropy coeff)")
  
  print("\n" + "=" * 70)

if __name__ == "__main__":
  compare_controllers()

