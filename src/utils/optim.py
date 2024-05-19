import numpy as np
import torch
from torch import optim, nn


class LagrangianOptimizer(object):
    """
    Simple lagrangian multiplier Adam optimizer

    :param float lr: learning rate of the optimizer
    :param float init: initial value of the lagrangian multiplier

    """

    def __init__(self, lr=10, init=0):
        self.lagrangian = nn.Parameter(torch.tensor(init, dtype=torch.float32, requires_grad=True))
        self.lr = lr
        self.optimizer = optim.Adam([self.lagrangian], lr=lr)

    def step(self, value: float, threshold: float) -> None:
        """Optimize the multiplier by one step

        :param float value: the current value estimation
        :param float threshold: the threshold of the value
        """

        self.loss = -self.lagrangian *  np.mean(value - threshold) 

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_lag(self) -> float:
        """Get the lagrangian multiplier."""
        return self.lagrangian.item()

    def state_dict(self) -> dict:
        """Get the parameters of this lagrangian optimizer"""
        params = {
            "lagrangian": self.lagrangian.item()
        }
        return params

    def load_state_dict(self, params: dict) -> None:
        """Load the parameters to continue training"""
        self.lagrangian = nn.Parameter(torch.tensor(params["lagrangian"], dtype=torch.float32, requires_grad=True))
        self.optimizer = optim.Adam([self.lagrangian], lr=self.lr)