import numpy as np

def calculate_costs(actual_lataccel, target_lataccel, dt=0.02):
  """
  Calculate lataccel_cost and jerk_cost for a rollout
  
  Args:
    actual_lataccel: array of actual lateral accelerations over time
    target_lataccel: array of target lateral accelerations over time  
    dt: timestep (default 0.02s)
    
  Returns:
    dict with lataccel_cost, jerk_cost, and total_cost
  """
  steps = len(actual_lataccel)
  
  # lataccel_cost: mean squared error * 100
  lataccel_cost = np.sum((actual_lataccel - target_lataccel) ** 2) / steps * 100
  
  # jerk_cost: mean squared jerk * 100
  # jerk = (accel_t - accel_{t-1}) / dt
  jerk = np.diff(actual_lataccel) / dt
  jerk_cost = np.sum(jerk ** 2) / (steps - 1) * 100
  
  # total_cost: weighted combination
  total_cost = (lataccel_cost * 50) + jerk_cost
  
  return {
    'lataccel_cost': lataccel_cost,
    'jerk_cost': jerk_cost,
    'total_cost': total_cost
  }

def evaluate_rollout(states, actions, target_lataccels, dt=0.02):
  """
  Evaluate a rollout with the cost metrics
  
  Args:
    states: array of states from rollout
    actions: array of actions (lataccels) from rollout
    target_lataccels: array of target lataccels
    dt: timestep
    
  Returns:
    dict with cost metrics
  """
  actual_lataccel = np.array(actions).squeeze()
  target_lataccel = np.array(target_lataccels).squeeze()
  
  costs = calculate_costs(actual_lataccel, target_lataccel, dt)
  
  return costs

