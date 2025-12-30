"""
Modified CartLatAccel environment with jerk-aware rewards
This should prevent PPO from self-destructing with more training
"""

import numpy as np
import gymnasium as gym
from gym_cartlataccel.env_v1 import CartLatAccelEnv


class CartLatAccelJerkEnv(CartLatAccelEnv):
    """
    CartLatAccel environment with jerk penalty in reward
    
    This prevents RL agents from learning overly aggressive control
    by penalizing rapid changes in acceleration (jerk).
    """
    
    def __init__(self, 
                 render_mode: str = None, 
                 noise_mode: str = None, 
                 moving_target: bool = True, 
                 env_bs: int = 1,
                 jerk_penalty_weight: float = 0.01,
                 action_penalty_weight: float = 0.001):
        """
        Args:
            jerk_penalty_weight: Weight for jerk penalty in reward
            action_penalty_weight: Weight for action magnitude penalty
        """
        super().__init__(render_mode, noise_mode, moving_target, env_bs)
        
        self.jerk_penalty_weight = jerk_penalty_weight
        self.action_penalty_weight = action_penalty_weight
        
        # Track previous action for jerk calculation
        self.prev_action = None
        
    def reset(self, seed=None, options=None):
        """Reset environment and previous action"""
        self.prev_action = None
        return super().reset(seed, options)
    
    def step(self, action):
        """Step with jerk-aware reward"""
        x = self.state[:,0]
        v = self.state[:,1]
        action = action.squeeze()
        action = np.clip(action, -self.max_u, self.max_u)
        noisy_action = self.noise_model.add_lat_noise(self.curr_step, action)

        new_a = noisy_action * self.force_mag
        new_x = 0.5 * new_a * self.tau**2 + v * self.tau + x
        new_x = np.clip(new_x, -self.max_x, self.max_x)
        new_v = new_a * self.tau + v
        new_x_target = self.x_targets[:, self.curr_step]

        self.state = np.stack([new_x, new_v, new_x_target], axis=1).astype(np.float32)

        # Position error (original reward)
        error = abs(new_x - new_x_target)
        position_reward = -error/self.max_x
        
        # Jerk penalty (rate of change of acceleration)
        if self.prev_action is not None:
            jerk = (action - self.prev_action) / self.tau
            jerk_penalty = -self.jerk_penalty_weight * (jerk ** 2)
        else:
            jerk_penalty = 0.0
        
        # Action magnitude penalty (discourage large actions)
        action_penalty = -self.action_penalty_weight * (action ** 2)
        
        # Combined reward
        reward = position_reward + jerk_penalty + action_penalty
        
        # Store for next step
        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        self.curr_step += 1
        truncated = self.curr_step >= self.max_episode_steps
        info = {
            "action": action, 
            "noisy_action": noisy_action, 
            "x": new_x, 
            "x_target": new_x_target,
            "position_reward": position_reward,
            "jerk_penalty": jerk_penalty,
            "action_penalty": action_penalty,
        }
        
        if self.bs == 1:
            return self.state[0], reward[0], False, truncated, info
        return self.state, reward, False, truncated, info


# Register the new environment
from gymnasium.envs.registration import register

register(
    id='CartLatAccel-Jerk-v1',
    entry_point='jerk_env:CartLatAccelJerkEnv',
    max_episode_steps=500,
)

