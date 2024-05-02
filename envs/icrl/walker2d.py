### Adapted from: https://github.com/Guiliang/ICRL-benchmarks-public

import numpy as np
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv


# =========================================================================== #
#                    Walker With Global Postion Coordinates                   #
# =========================================================================== #

class Walker2dWall(Walker2dEnv):

    def __init__(self, *args, **kwargs):
        super(Walker2dWall, self).__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        rem_reward = reward = self._forward_reward_weight * info['x_velocity']
        reward = rem_reward + self._forward_reward_weight * abs(info['x_velocity'])

        return observation, reward, terminated, truncated, info
