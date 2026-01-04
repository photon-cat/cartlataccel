import torch
import numpy as np
from . import BaseController

class Controller(BaseController):
  """PPO controller - loads trained model"""
  def __init__(self, model_path='models/ppo.pt'):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = torch.load(model_path, map_location=self.device, weights_only=False)
    self.model.eval()

  def act(self, obs):
    with torch.no_grad():
      obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
      action = self.model(obs_tensor).cpu().numpy().flatten()[0]
    return action

  def reset(self):
    pass

