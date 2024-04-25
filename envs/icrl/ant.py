### Adapted from: https://github.com/Guiliang/ICRL-benchmarks-public


import numpy as np
import os
from gymnasium.envs.mujoco.ant_v4 import AntEnv


###############################################################################
# ANT WALL ENVIRONMENTS
###############################################################################

class AntWall(AntEnv):
    def __init__(self, *args, **kwargs):
       super(AntWall, self).__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = info['distance_from_origin'] + info['reward_survive']
        return observation, reward, terminated, truncated, info