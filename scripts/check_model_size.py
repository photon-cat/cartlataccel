"""
Count parameters in actor vs critic for 256x4 network
"""

import torch
from model import ActorCritic

# Create the optimal 256x4 model
model = ActorCritic(
    obs_dim=3,
    hidden_sizes={"pi": [256, 256, 256, 256], "vf": [256, 256, 256, 256]},
    act_dim=1
)

# Count parameters
actor_params = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
critic_params = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 80)
print("MODEL SIZE BREAKDOWN - 256x4 Configuration")
print("=" * 80)
print()

print("Actor (Policy Network):")
print(f"  Architecture: 3 → [256, 256, 256, 256] → 1")
print(f"  Parameters: {actor_params:,}")
print(f"  Percentage: {actor_params/total_params*100:.1f}%")
print()

print("Critic (Value Network):")
print(f"  Architecture: 3 → [256, 256, 256, 256] → 1")
print(f"  Parameters: {critic_params:,}")
print(f"  Percentage: {critic_params/total_params*100:.1f}%")
print()

print("Total:")
print(f"  Parameters: {total_params:,}")
print()

# Detailed layer breakdown
print("=" * 80)
print("DETAILED LAYER BREAKDOWN")
print("=" * 80)
print()

print("Actor Layers:")
for name, param in model.actor.named_parameters():
    print(f"  {name:30s}: {param.numel():>8,} params  {list(param.shape)}")

print()
print("Critic Layers:")
for name, param in model.critic.named_parameters():
    print(f"  {name:30s}: {param.numel():>8,} params  {list(param.shape)}")

print()
print("=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print()
print(f"Actor and Critic are ALMOST THE SAME SIZE!")
print(f"  Actor:  {actor_params:,} parameters")
print(f"  Critic: {critic_params:,} parameters")
print(f"  Ratio:  {actor_params/critic_params:.2f}:1")
print()
print("The only difference is the actor has an extra log_std parameter (1 value)")
print("for controlling the exploration noise.")

