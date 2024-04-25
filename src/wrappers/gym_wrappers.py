#!/usr/bin/env python3

from typing import SupportsFloat, Callable

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
    

class CostWrapper(gym.Wrapper):
    """Sets the reward to -cost.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env, cost_fn: Callable) -> None:
        super().__init__(env)
        self.cost_fn = cost_fn

    def step(self, action: np.ndarray):
        obs, reward, term, trunc, info = self.env.step(action)
        cost = self.cost_fn(obs=obs, acs=action, info=info)
        info = {'cost': cost, **info}
        return obs, reward, term, trunc, info