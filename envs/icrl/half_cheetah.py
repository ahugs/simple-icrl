### Adapted from: https://github.com/Guiliang/ICRL-benchmarks-public


import os

import gymnasium
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

# ========================================================================== #
# CHEETAH WITH TORQUE CONSTRAINT
# ========================================================================== #

ACTION_TORQUE_THRESHOLD = 0.5
VIOLATIONS_ALLOWED = 100


class HalfCheetahWall(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args,  **kwargs)

    def step(self, action):
        obs, _, term, trun, info = super().step(action)
        reward_ctrl = info['reward_ctrl']
        reward_run = info['reward_run']

        return obs, abs(reward_run) + reward_ctrl, term, trun, info
