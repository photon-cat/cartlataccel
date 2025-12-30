"""
Deep dive: Why can't PPO beat PID?
Let's find the bottlenecks and fix them
"""

import numpy as np
import torch
import gymnasium as gym
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from model import ActorCritic
from ppo import PPO
from controllers import PIDController
from eval_cost import calculate_costs

# Register environment
try:
    register(
        id='CartLatAccel-Jerk-v1',
        entry_point='jerk_env:CartLatAccelJerkEnv',
        max_episode_steps=500,
    )
except:
    pass


def analyze_controller_behavior(controller_name, model_or_controller, is_ppo=True, device='cpu'):
    """Analyze what the controller is actually doing"""
    
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1, 
                   jerk_penalty_weight=0.01, action_penalty_weight=0.001)
    
    state, _ = env.reset()
    if not is_ppo:
        model_or_controller.reset()
    
    # Track detailed metrics
    states = []
    actions = []
    position_errors = []
    velocity_tracking = []
    action_magnitudes = []
    action_changes = []
    rewards = []
    
    prev_action = 0
    
    for step in range(500):
        pos, vel, target_pos = state[0], state[1], state[2]
        pos_error = abs(pos - target_pos)
        
        if is_ppo:
            state_tensor = torch.FloatTensor(state).to(device)
            action = model_or_controller.get_action(state_tensor, deterministic=True)
        else:
            # PID needs target lataccel
            target_lataccel = (target_pos - pos) * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            current_lataccel = prev_action  # Approximate
            action = model_or_controller.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, terminated, truncated, info = env.step(np.array([action]))
        
        states.append(state)
        actions.append(action)
        position_errors.append(pos_error)
        velocity_tracking.append(abs(vel))
        action_magnitudes.append(abs(action))
        action_changes.append(abs(action - prev_action))
        rewards.append(reward)
        
        prev_action = action
        state = next_state
        
        if terminated or truncated:
            break
    
    # Calculate statistics
    actions = np.array(actions)
    
    analysis = {
        'controller': controller_name,
        'mean_pos_error': np.mean(position_errors),
        'max_pos_error': np.max(position_errors),
        'mean_velocity': np.mean(velocity_tracking),
        'mean_action': np.mean(action_magnitudes),
        'max_action': np.max(action_magnitudes),
        'mean_action_change': np.mean(action_changes),
        'max_action_change': np.max(action_changes),
        'action_std': np.std(actions),
        'total_reward': np.sum(rewards),
        'actions_near_limit': np.sum(np.abs(actions) > 0.9) / len(actions),  # % at limits
        'action_diversity': len(np.unique(np.round(actions, 2))) / len(actions),  # Unique actions
    }
    
    return analysis


def main():
    print("=" * 80)
    print("WHY CAN'T PPO BEAT PID? - DEEP ANALYSIS")
    print("=" * 80)
    print()
    
    # Load best PPO model
    print("Loading best PPO model (200k steps)...")
    try:
        import glob
        model_files = glob.glob("logs/1m_training_*/model_step_0200000.pt")
        if model_files:
            model_path = model_files[0]
            print(f"Found: {model_path}")
        else:
            print("ERROR: No model found. Please run train_1m.py first")
            return
    except:
        print("ERROR: Could not find model")
        return
    
    # Create PPO model and load weights
    ppo_model = ActorCritic(3, {"pi": [32], "vf": [32]}, 1)
    ppo_model.actor.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Create PID controller
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    
    print("\nAnalyzing controller behaviors...")
    print()
    
    # Analyze both controllers
    ppo_analysis = analyze_controller_behavior("PPO (200k)", ppo_model.actor, is_ppo=True)
    pid_analysis = analyze_controller_behavior("PID", pid, is_ppo=False)
    
    # Display comparison
    print("=" * 80)
    print("BEHAVIORAL COMPARISON")
    print("=" * 80)
    print()
    print("Metric                    | PPO (200k)  | PID         | Winner")
    print("-" * 80)
    
    metrics = [
        ('mean_pos_error', 'Mean Position Error', 'lower'),
        ('max_pos_error', 'Max Position Error', 'lower'),
        ('mean_action', 'Mean Action Magnitude', None),
        ('max_action', 'Max Action', None),
        ('mean_action_change', 'Mean Action Change', 'lower'),
        ('max_action_change', 'Max Action Change', 'lower'),
        ('action_std', 'Action Std Dev', None),
        ('total_reward', 'Total Reward', 'higher'),
        ('actions_near_limit', '% Actions Near Limit', None),
        ('action_diversity', 'Action Diversity', None),
    ]
    
    for key, label, preference in metrics:
        ppo_val = ppo_analysis[key]
        pid_val = pid_analysis[key]
        
        if preference == 'lower':
            winner = "PPO" if ppo_val < pid_val else "PID"
        elif preference == 'higher':
            winner = "PPO" if ppo_val > pid_val else "PID"
        else:
            winner = "-"
        
        print(f"{label:25} | {ppo_val:>11.4f} | {pid_val:>11.4f} | {winner}")
    
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    
    # Identify problems
    problems = []
    
    if ppo_analysis['mean_pos_error'] > pid_analysis['mean_pos_error'] * 1.5:
        problems.append("❌ PPO has significantly worse position tracking")
        print(f"1. Position Tracking: PPO error is {ppo_analysis['mean_pos_error']/pid_analysis['mean_pos_error']:.2f}x worse")
    
    if ppo_analysis['action_std'] < 0.1:
        problems.append("❌ PPO actions lack diversity (too conservative)")
        print(f"2. Action Diversity: PPO std={ppo_analysis['action_std']:.4f} is very low")
    
    if ppo_analysis['actions_near_limit'] < 0.01:
        problems.append("❌ PPO never uses full action space")
        print(f"3. Action Range: PPO only uses {ppo_analysis['actions_near_limit']*100:.1f}% of action space")
    
    if ppo_analysis['mean_action'] < pid_analysis['mean_action'] * 0.5:
        problems.append("❌ PPO is too timid (actions too small)")
        print(f"4. Action Magnitude: PPO uses {ppo_analysis['mean_action']/pid_analysis['mean_action']:.2f}x smaller actions")
    
    print()
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()
    
    if len(problems) == 0:
        print("No obvious behavioral problems found!")
        print("The issue might be:")
        print("  - Network capacity (32 hidden units is small)")
        print("  - Training time (might need more than 200k steps)")
        print("  - Reward shaping (penalties might be too strong)")
    else:
        print("IDENTIFIED PROBLEMS:")
        for i, p in enumerate(problems, 1):
            print(f"{i}. {p}")
        
        print()
        print("LIKELY ROOT CAUSES:")
        print()
        
        if "conservative" in str(problems) or "timid" in str(problems):
            print("➜ JERK PENALTY IS TOO STRONG")
            print("  Current: jerk_weight=0.01, action_weight=0.001")
            print("  Try: jerk_weight=0.001, action_weight=0.0001")
            print()
        
        if "diversity" in str(problems) or "action space" in str(problems):
            print("➜ NETWORK IS TOO SMALL OR UNDERTRAINED")
            print("  Current: 32 hidden units, 200k steps")
            print("  Try: 128 hidden units, 500k+ steps")
            print()
        
        if "tracking" in str(problems):
            print("➜ REWARD BALANCE IS OFF")
            print("  Position error might be underweighted")
            print("  Try: Reduce jerk penalty OR increase position weight")
            print()
    
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    print("Try these in order:")
    print()
    print("1. REDUCE PENALTY WEIGHTS (Most likely to help)")
    print("   python train_1m.py --jerk_weight 0.001 --action_weight 0.0001 --max_evals 500000")
    print()
    print("2. INCREASE NETWORK SIZE")
    print("   Modify model.py: Change hidden size from 32 → 128")
    print("   python train_1m.py --max_evals 500000")
    print()
    print("3. DIFFERENT REWARD STRUCTURE")
    print("   Try lataccel tracking as primary reward instead of position")
    print()
    print("4. USE A DIFFERENT ALGORITHM")
    print("   SAC or TD3 might work better for continuous control")
    print()


if __name__ == "__main__":
    main()

