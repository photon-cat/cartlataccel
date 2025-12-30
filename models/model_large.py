"""
Bigger network version of ActorCritic
256 hidden units instead of 32
"""

import torch
import torch.nn as nn
import numpy as np


class ActorCriticLarge(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim):
        super().__init__()
        
        # Much larger actor network
        self.actor = Actor(obs_dim, hidden_dims["pi"], action_dim)
        
        # Much larger critic network  
        self.critic = Critic(obs_dim, hidden_dims["vf"])


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dims, action_dim):
        super().__init__()
        
        # Build deep, wide network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Add normalization for stability
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        mean = self.net(obs)
        mean = torch.tanh(mean)  # Keep in [-1, 1]
        return mean
    
    def get_action(self, obs, deterministic=False):
        mean = self.forward(obs)
        
        if deterministic:
            return mean.detach().cpu().numpy()
        
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        
        return action.detach().cpu().numpy()
    
    def get_logprob(self, obs, actions):
        mean = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.net(obs)

