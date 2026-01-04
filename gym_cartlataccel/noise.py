import numpy as np

CAMERA_FPS = 50

class SimNoise:
  """
  Simulates realistic control noise with:
  - Action delay (lag)
  - Action scaling noise
  - Position drift noise
  - Temporal correlation (smooth noise over time)
  
  Args:
    n: number of timesteps
    fps: simulation frequency
    noise: noise scale 0-1 (default 0.5 = REALISTIC, 0=none, 1=high)
    seed: random seed
  """
  def __init__(self, n, fps, noise=0.5, seed=42):
    self.n = n
    self.fps = fps

    # Scale params so noise=0.5 matches old REALISTIC mode
    # REALISTIC was: lag=0.5, t_corr=[0.5,5], dy=1, lat_action=0.3
    self.lag = noise * 1.0              # 0.5 -> 0.5s delay
    self.t_corr = 0.5 + noise * 9.0     # 0.5 -> 5s window
    self.dy_scale = noise * 2.0         # 0.5 -> 1.0 drift
    self.action_scale = noise * 0.6     # 0.5 -> 0.3 (30% action noise)
    
    # Generate correlated noise sequences
    np.random.seed(seed)
    self.t_noise_corr = np.random.uniform(0.5, self.t_corr)
    self.dy_noise = np.random.uniform(-self.dy_scale, self.dy_scale, size=self.n)
    self.dy_noise = self._correlate(self.dy_noise)
    self.lat_action_noise = np.random.normal(0.0, self.action_scale, size=self.n)
    self.lat_action_noise = self._correlate(self.lat_action_noise)

    self.reset(seed)  

  def reset(self, seed=None):
    if seed is not None:
      np.random.seed(seed)
    # Initialize lag buffer
    lag_frames = int(self.lag * self.fps) if self.lag > 0 else 0
    self.lats = [0.0] * lag_frames

  def _correlate(self, noise_samples):
    """Smooth noise over time using convolution"""
    t_frames = max(1, int(self.t_noise_corr * CAMERA_FPS))
    kernel = np.ones(t_frames) / np.sqrt(t_frames)
    return np.convolve(noise_samples, kernel, mode='same')

  def add_lat_noise(self, step, action):
    # Scale action by multiplicative noise
    scaled_action = (1 + self.lat_action_noise[step]) * action
    # Add position drift
    drift = self.dy_noise[step] / self.fps
    # Apply lag
    self.lats.append(scaled_action)
    return self.lats.pop(0) + drift
