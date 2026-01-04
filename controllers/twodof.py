import numpy as np
from . import BaseController

class Controller(BaseController):
  """
  2-DOF Controller: Feedforward (target velocity) + Feedback (PID)
  
  Architecture:
    action = feedforward + PID(error)
    
  The feedforward predicts what velocity we SHOULD have to follow the trajectory.
  The PID corrects for errors when the prediction is wrong.
  
  What happens with model mismatch:
  
  UNDER-CONTROL (feedforward too weak):
    - Cart lags behind target
    - Position error builds up
    - P term increases correction → catches up
    - I term accumulates → eliminates steady-state error
    
  OVER-CONTROL (feedforward too aggressive):  
    - Cart overshoots target
    - Position error goes negative
    - P term reduces/reverses correction
    - D term (velocity damping) prevents oscillation
    
  With noise/disturbances:
    - Feedforward doesn't know about disturbances
    - PID reacts to resulting errors
    - I term handles persistent bias (e.g., constant wind)
    - D term damps out oscillations from noise
  """
  
  def __init__(self, force_mag=50.0, tau=0.02, max_rel_vel=5.0, kp=40.0, ki=0.01, kd=0.5):
    """
    Args:
      force_mag: action-to-acceleration scaling (action=1 → 50 m/s²)
      tau: simulation timestep (0.02s = 50Hz)
      max_rel_vel: max allowed |cart_vel - target_vel| to prevent runaway
      kp: proportional gain - how aggressively to correct position error
          higher = faster response but may overshoot
      ki: integral gain - accumulates error over time to eliminate steady-state offset
          higher = eliminates bias faster but may cause windup/oscillation
      kd: derivative gain - damps velocity error to prevent overshoot
          higher = more damping but slower response
    """
    self.force_mag = force_mag
    self.tau = tau
    self.max_rel_vel = max_rel_vel
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.reset()
  
  def set_trajectory(self, trajectory):
    """Store the full target trajectory for lookahead"""
    self.trajectory = trajectory
  
  def act(self, obs):
    """
    Compute control action given current observation.
    
    Args:
      obs: [pos, vel, target_pos] - current cart state
      
    Returns:
      action in [-1, 1] - commanded acceleration / force_mag
    """
    pos, vel, target_pos = obs[0], obs[1], obs[2]
    
    # =========================================================================
    # EXACT MODEL INVERSION
    # =========================================================================
    # The env rewards us for: |pos_after_action - traj[curr_step]|
    # So we want pos after this action to equal traj[step]
    # 
    # Physics: x_next = x + v*τ + 0.5*a*τ²
    # Solve for a: a = 2*(x_target - x - v*τ) / τ²
    #
    # Note: target_pos in obs is traj[step-1] (previous step's target)
    # We need traj[step] which is the NEXT target
    
    if self.trajectory is not None and self.step < len(self.trajectory):
      # This is the target we'll be compared against AFTER this step
      target_for_reward = self.trajectory[self.step]
    else:
      target_for_reward = target_pos
    
    # Exact model inversion: what accel to reach target_for_reward?
    accel = 2 * (target_for_reward - pos - vel * self.tau) / (self.tau ** 2)
    
    # =========================================================================
    # VELOCITY LIMITING (catch-up from bad initial state)
    # =========================================================================
    # Compute target velocity from trajectory
    if self.trajectory is not None and self.step + 1 < len(self.trajectory):
      target_vel = (self.trajectory[self.step + 1] - self.trajectory[self.step]) / self.tau
    else:
      target_vel = 0.0
    
    # Limit relative velocity to prevent wild oscillations
    predicted_vel = vel + accel * self.tau
    rel_vel = predicted_vel - target_vel
    if abs(rel_vel) > self.max_rel_vel:
      clamped_vel = target_vel + np.sign(rel_vel) * self.max_rel_vel
      accel = (clamped_vel - vel) / self.tau
    
    # Convert to action (normalize by force_mag)
    action = accel / self.force_mag
    self.step += 1
    
    # Clamp to valid action range
    return np.clip(action, -1.0, 1.0)
  
  def reset(self):
    """Reset controller state for new episode"""
    self.step = 0
    self.trajectory = None
    self.prev_error = 0.0
    self.error_integral = 0.0  # reset integral windup
