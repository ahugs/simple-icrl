from typing import Any

import numpy as np
import torch

from tianshou.env.utils import gym_new_venv_step_type
from tianshou.env.venvs import GYM_RESERVED_KEYS, BaseVectorEnv
from tianshou.env.venv_wrappers import VectorEnvWrapper
from tianshou.utils import RunningMeanStd


class VectorEnvNormRew(VectorEnvWrapper):
    """A reward normalization wrapper for vectorized environments.

    :param update_rew_rms: whether to update rew_rms. Default to True.
    """

    def __init__(self, venv: BaseVectorEnv, update_rew_rms: bool = True) -> None:
        super().__init__(venv)
        # initialize observation running mean/std
        self.update_rew_rms = update_rew_rms
        self.rew_rms = RunningMeanStd()

    def step(
        self,
        action: np.ndarray | torch.Tensor,
        id: int | list[int] | np.ndarray | None = None,
    ) -> gym_new_venv_step_type:
        obs, rew, term, trunc, info = self.venv.step(action, id)
        if self.rew_rms and self.update_rew_rms:
            self.rew_rms.update(rew)
        return obs, self._norm_rew(rew), term, trunc, info

    def _norm_rew(self, rew: np.ndarray) -> np.ndarray:
        if self.rew_rms:
            return self.rew_rms.norm(rew)  # type: ignore
        return rew

    def set_rew_rms(self, rew_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.rew_rms = rew_rms

    def get_rew_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.rew_rms