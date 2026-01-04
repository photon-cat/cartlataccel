from . import BaseController
import numpy as np

class Controller(BaseController):
  """PID controller for position tracking"""
  def __init__(self, p=0.07, i=0.00, d=0.1):
    self.p = p
    self.i = i
    self.d = d
    self.reset()

  def act(self, obs):
    pos, vel, target_pos = obs[0], obs[1], obs[2]
    error = target_pos - pos
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    action = self.p * error + self.i * self.error_integral + self.d * error_diff
    return np.clip(action, -1.0, 1.0)

  def reset(self):
    self.error_integral = 0
    self.prev_error = 0
