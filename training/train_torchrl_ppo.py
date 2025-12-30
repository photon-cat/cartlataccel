"""
TorchRL PPO Training for CartLatAccel Environment

Adapts the official TorchRL PPO tutorial to work with CartLatAccel-v1
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import gymnasium as gym
import gym_cartlataccel
from eval_cost import calculate_costs

import argparse


def create_env(device="cpu", noise_mode=None):
    """Create and configure the CartLatAccel environment"""
    # Create base gym environment
    base_env = GymEnv("CartLatAccel-v1", device=device, 
                      noise_mode=noise_mode, env_bs=1)
    
    # Add transforms
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    
    # Initialize normalization statistics
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    
    print("Environment created successfully!")
    print(f"Observation spec: {env.observation_spec}")
    print(f"Action spec: {env.action_spec}")
    
    # Check environment specs
    check_env_specs(env)
    
    return env


def create_policy_network(env, num_cells=256, device="cpu"):
    """Create policy network (actor)"""
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )
    
    return policy_module


def create_value_network(env, num_cells=256, device="cpu"):
    """Create value network (critic)"""
    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
    
    return value_module


def evaluate_policy(env, policy_module, device="cpu", n_rollouts=5):
    """Evaluate policy with cost metrics"""
    all_costs = []
    all_rewards = []
    
    for _ in range(n_rollouts):
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # Rollout for 500 steps
            rollout_data = env.rollout(500, policy_module)
            
            # Extract data
            actions = rollout_data["action"].cpu().numpy()
            states = rollout_data["observation"].cpu().numpy()
            rewards = rollout_data["next", "reward"].cpu().numpy()
            
            # Calculate target lataccels (same as in eval_pid.py)
            target_lataccels = []
            actual_lataccels = []
            
            for i in range(len(states)):
                if len(states[i].shape) > 1:
                    state = states[i][0]
                    action = actions[i][0]
                else:
                    state = states[i]
                    action = actions[i]
                
                # Extract pos, vel, target_pos from state
                if len(state) >= 3:
                    pos, vel, target_pos = state[0], state[1], state[2]
                    pos_error = target_pos - pos
                    target_lataccel = pos_error * 10.0
                    target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
                    target_lataccels.append(target_lataccel)
                    actual_lataccels.append(action)
            
            if len(target_lataccels) > 0:
                costs = calculate_costs(
                    np.array(actual_lataccels), 
                    np.array(target_lataccels), 
                    dt=0.02
                )
                all_costs.append(costs)
                all_rewards.append(rewards.mean())
    
    # Aggregate results
    if len(all_costs) > 0:
        avg_results = {
            'lataccel_cost': np.mean([c['lataccel_cost'] for c in all_costs]),
            'jerk_cost': np.mean([c['jerk_cost'] for c in all_costs]),
            'total_cost': np.mean([c['total_cost'] for c in all_costs]),
            'reward': np.mean(all_rewards),
        }
    else:
        avg_results = {
            'lataccel_cost': 0,
            'jerk_cost': 0,
            'total_cost': 0,
            'reward': 0,
        }
    
    return avg_results


def train_ppo_torchrl(args):
    """Train PPO using TorchRL pipeline"""
    
    # Setup
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/torchrl_ppo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"TorchRL PPO Training - {timestamp}")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Total frames: {args.total_frames}")
    print(f"Frames per batch: {args.frames_per_batch}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Log directory: {log_dir}")
    print("=" * 80)
    print()
    
    # Create environment
    print("Creating environment...")
    env = create_env(device=device, noise_mode=args.noise_mode)
    
    # Create policy and value networks
    print("Creating policy and value networks...")
    policy_module = create_policy_network(env, num_cells=args.num_cells, device=device)
    value_module = create_value_network(env, num_cells=args.num_cells, device=device)
    
    print("Testing policy and value networks...")
    print("Policy output:", policy_module(env.reset()))
    print("Value output:", value_module(env.reset()))
    
    # Create data collector
    print(f"Creating data collector (frames_per_batch={args.frames_per_batch})...")
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        split_trajs=False,
        device=device,
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    # Create advantage module (GAE)
    advantage_module = GAE(
        gamma=args.gamma,
        lmbda=args.lmbda,
        value_network=value_module,
        average_gae=True,
        device=device,
    )
    
    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_bonus=bool(args.entropy_eps),
        entropy_coef=args.entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    
    # Optimizer
    optim = torch.optim.Adam(loss_module.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args.total_frames // args.frames_per_batch, 0.0
    )
    
    # Training loop
    print("\nStarting training loop...")
    logs = defaultdict(list)
    benchmark_log = []
    pbar = tqdm(total=args.total_frames)
    eval_str = ""
    frames_collected = 0
    start_time = time.time()
    
    # Initial evaluation
    if args.eval_interval > 0:
        print("Initial evaluation...")
        eval_results = evaluate_policy(env, policy_module, device=device, n_rollouts=args.eval_rollouts)
        eval_results['timesteps'] = 0
        eval_results['wall_time'] = 0
        benchmark_log.append(eval_results)
        print(f"Initial total_cost: {eval_results['total_cost']:.2f}\n")
    
    for i, tensordict_data in enumerate(collector):
        frames_collected += tensordict_data.numel()
        
        # PPO training epochs
        for _ in range(args.num_epochs):
            # Compute advantage
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            
            for _ in range(args.frames_per_batch // args.sub_batch_size):
                subdata = replay_buffer.sample(args.sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                # Backward pass
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
                optim.step()
                optim.zero_grad()
        
        # Logging
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])
        
        pbar.update(tensordict_data.numel())
        
        # Periodic evaluation
        if args.eval_interval > 0 and (i + 1) % args.eval_interval == 0:
            print(f"\nEvaluating at {frames_collected} frames...")
            eval_results = evaluate_policy(env, policy_module, device=device, n_rollouts=args.eval_rollouts)
            eval_results['timesteps'] = frames_collected
            eval_results['wall_time'] = time.time() - start_time
            benchmark_log.append(eval_results)
            
            eval_str = (
                f"total_cost: {eval_results['total_cost']:>8.2f}, "
                f"reward: {eval_results['reward']:>7.3f}"
            )
            
            # Save checkpoint
            checkpoint_path = os.path.join(log_dir, f"model_step_{frames_collected:07d}.pt")
            torch.save(policy_module.state_dict(), checkpoint_path)
        
        cum_reward_str = f"reward={logs['reward'][-1]:.4f}"
        pbar.set_description(f"{eval_str}, {cum_reward_str}")
        
        scheduler.step()
    
    pbar.close()
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_results = evaluate_policy(env, policy_module, device=device, n_rollouts=args.eval_rollouts * 2)
    final_results['timesteps'] = frames_collected
    final_results['wall_time'] = total_time
    benchmark_log.append(final_results)
    
    # Save results
    log_path = os.path.join(log_dir, "benchmark_log.json")
    with open(log_path, 'w') as f:
        json.dump(benchmark_log, f, indent=2)
    
    summary = {
        'config': vars(args),
        'timestamp': timestamp,
        'total_time': total_time,
        'total_frames': frames_collected,
        'benchmark_log': benchmark_log,
    }
    
    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save final model
    model_path = os.path.join(log_dir, "final_model.pt")
    torch.save(policy_module.state_dict(), model_path)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total frames: {frames_collected:,}")
    print(f"\nFinal Results:")
    print(f"  Total cost:    {final_results['total_cost']:,.2f}")
    print(f"  Lataccel cost: {final_results['lataccel_cost']:.2f}")
    print(f"  Jerk cost:     {final_results['jerk_cost']:,.2f}")
    print(f"  Reward:        {final_results['reward']:.3f}")
    print(f"\nResults saved to: {log_dir}")
    print("=" * 80)
    
    return log_dir, benchmark_log


def main():
    parser = argparse.ArgumentParser(description="TorchRL PPO for CartLatAccel")
    
    # Environment
    parser.add_argument("--noise_mode", default=None, help="Noise mode (None, REALISTIC, HIGH)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    
    # Network architecture
    parser.add_argument("--num_cells", type=int, default=256, help="Hidden layer size")
    
    # Training
    parser.add_argument("--total_frames", type=int, default=50000, help="Total training frames")
    parser.add_argument("--frames_per_batch", type=int, default=1000, help="Frames per batch")
    parser.add_argument("--sub_batch_size", type=int, default=64, help="Sub-batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Epochs per batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # PPO hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_eps", type=float, default=1e-4, help="Entropy coefficient")
    
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate every N batches (0 to disable)")
    parser.add_argument("--eval_rollouts", type=int, default=3, help="Rollouts per evaluation")
    
    args = parser.parse_args()
    
    log_dir, benchmark_log = train_ppo_torchrl(args)
    
    # Print comparison with PID
    if len(benchmark_log) > 0:
        best = min(benchmark_log, key=lambda x: x['total_cost'])
        final = benchmark_log[-1]
        
        print("\n" + "=" * 80)
        print("COMPARISON WITH BASELINES")
        print("=" * 80)
        print(f"PID baseline:         total_cost = 2,080.39")
        print(f"TorchRL PPO (best):   total_cost = {best['total_cost']:,.2f} at {best['timesteps']} steps")
        print(f"TorchRL PPO (final):  total_cost = {final['total_cost']:,.2f}")
        print(f"\nPID is {best['total_cost']/2080.39:.1f}x better than TorchRL PPO's best")
        print("=" * 80)


if __name__ == "__main__":
    main()

