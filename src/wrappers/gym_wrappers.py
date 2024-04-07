#!/usr/bin/env python3

from typing import SupportsFloat

import gymnasium as gym
import numpy as np


class NoRewardEnv(gym.RewardWrapper):
    """sets the reward to 0.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> np.ndarray:
        """Set reward to 0."""
        return np.zeros_like(reward)
