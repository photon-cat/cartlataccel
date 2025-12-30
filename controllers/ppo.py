from . import BaseController
import numpy as np
import torch


class Controller(BaseController):
  """
  PPO (Vanilla) neural network controller.
  
  Loads a trained actor network and uses it for control.
  """
  def __init__(self, model_path=None, device='cpu'):
    self.device = device
    self.model = None
    
    if model_path:
      self.load(model_path)
  
  def load(self, model_path):
    """Load trained model from path."""
    self.model = torch.load(model_path, map_location=self.device, weights_only=False)
    self.model.eval()
    print(f"Loaded PPO model from {model_path}")
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Compute action from neural network.
    
    Note: PPO uses full state directly, ignores target_lataccel/current_lataccel.
    """
    if self.model is None:
      raise RuntimeError("No model loaded. Call load() or pass model_path to __init__")
    
    # State is [pos, vel, target_pos]
    if isinstance(state, np.ndarray):
      state_tensor = torch.FloatTensor(state).to(self.device)
    else:
      state_tensor = state.to(self.device)
    
    with torch.no_grad():
      action = self.model.get_action(state_tensor, deterministic=True)
    
    if isinstance(action, torch.Tensor):
      action = action.cpu().numpy()
    
    # Flatten if needed
    if hasattr(action, '__len__') and len(action) == 1:
      action = float(action[0])
    else:
      action = float(action)
    
    return np.clip(action, -1.0, 1.0)
  
  def reset(self):
    """No state to reset for feedforward network."""
    pass

