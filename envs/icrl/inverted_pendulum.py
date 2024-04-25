### Adapted from: https://github.com/Guiliang/ICRL-benchmarks-public

import os

import gymnasium
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv

# =========================================================================== #
#                         Inverted Pendulum Wall                              #
# =========================================================================== #


class InvertedPendulumWall(InvertedPendulumEnv):

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))
        if xposafter <= -0.01:
            reward = 1
        elif xposafter >= 0:
            reward = 0.1
        else:
            reward = (-xposafter/0.01)*0.9+0.1
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {'x_position': xposafter}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
