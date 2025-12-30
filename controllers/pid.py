from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  PID controller for position tracking.
  
  Usage: action = controller.update(target_pos, current_pos, state, future_plan)
  """
  def __init__(self,):
    # Default gains for position control
    self.p = 2.0
    self.i = 0.01
    self.d = 0.5
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Compute PID control action.
    
    Args:
      target_lataccel: Target position (or lataccel if tracking acceleration)
      current_lataccel: Current position (or lataccel)
      state: Full state [pos, vel, target_pos]
      future_plan: Future trajectory (unused)
    
    Returns:
      action: Control signal (acceleration)
    """
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff

  def reset(self):
    self.error_integral = 0
    self.prev_error = 0

