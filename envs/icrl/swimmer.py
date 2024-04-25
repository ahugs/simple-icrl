### Adapted from: https://github.com/Guiliang/ICRL-benchmarks-public

import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv

# =========================================================================== #
#                   Swimmer                                                   #
# =========================================================================== #

class SwimmerWall(SwimmerEnv):

    def __init__(self, *args, **kwargs):
        super(SwimmerWall, self).__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = abs(info['forward_reward']) + info['reward_ctrl']
        return observation, reward, terminated, truncated, info 
