class BaseController:
  """Base controller interface"""
  def act(self, obs):
    """Given observation [pos, vel, target_pos], return action"""
    raise NotImplementedError
  
  def reset(self):
    """Reset controller state between episodes"""
    pass

