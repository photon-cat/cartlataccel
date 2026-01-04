import numpy as np
from . import BaseController

"""
pure ff expecting perfect model

"""

class Controller(BaseController):
    def __init__(self, force_mag=50.0, tau=0.02):
        self.force_mag = force_mag
        self.tau = tau

    def act(self, obs):
        # obs = [pos_t, vel_t, pos_{t+1}]
        pos, vel, next_pos = obs

        # desired delta position
        delta_x = next_pos - pos

        # exact acceleration to achieve delta_x in one timestep
        accel = 2.0 * (delta_x - vel * self.tau) / (self.tau ** 2)

        # normalize to action space
        action = accel / self.force_mag
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        pass
