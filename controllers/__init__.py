"""
Controllers for CartLatAccel environment.

Each controller implements the BaseController interface:
  - update(target_lataccel, current_lataccel, state, future_plan) -> action
  - reset()
"""

import numpy as np


class BaseController:
  """Base controller interface."""
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Compute control action.
    
    Args:
      target_lataccel: Target lateral acceleration (or position for position control)
      current_lataccel: Current lateral acceleration (or position)
      state: Full state [pos, vel, target_pos]
      future_plan: Future trajectory (optional)
    
    Returns:
      action: Control action in [-1, 1]
    """
    raise NotImplementedError
  
  def reset(self):
    """Reset controller state."""
    pass


from controllers.pid import Controller as PIDController
from controllers.ppo import Controller as PPOController
from controllers.poly import Controller as PolyController

