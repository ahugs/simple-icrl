from abc import ABC
from collections.abc import Sequence
from typing import Any, List, Union, Optional, Type
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.common import (
    MLP,
    Net,
    TLinearLayer,
    TActionShape,
    get_output_dim,
)
from tianshou.utils.net.continuous import ActorProb
from fsrl.utils.net.continuous import DoubleCritic


class Reward(nn.Module, ABC):
    """Simple reward network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.
    :param linear_layer: use this module as linear layer.
    :param output_activation: use this output activation function
    :param flatten_input: whether to flatten input data for the last layer.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        hidden_sizes: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        output_activation: nn.Module = nn.Identity,
        output_scaling: float = 1.0,
        clip_range: List[float] = [-np.inf, np.inf],
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )
        self.output_activation = output_activation()
        self.output_scaling = output_scaling
        self.clip_range = clip_range

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        values_B, hidden_BH = self.preprocess(obs)
        return torch.clamp(self.output_scaling * self.output_activation(self.last(values_B)), 
                           min=self.clip_range[0], max=self.clip_range[1])
    

class DoubleCritic(DoubleCritic):

    def __init__(
        self,
        preprocess_net1: nn.Module,
        preprocess_net2: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.linear_layer = linear_layer
        self.flatten_input = flatten_input
        self.preprocess_net_output_dim = preprocess_net_output_dim
        super(DoubleCritic, self).__init__(preprocess_net1,
                                           preprocess_net2,
                                           hidden_sizes,
                                           device,
                                           preprocess_net_output_dim,
                                           linear_layer,
                                           flatten_input)

    def reinitialize_last_layer(self):
        input_dim = getattr(self.preprocess1, "output_dim", self.preprocess_net_output_dim)
        self.last1 = MLP(
            input_dim,  # type: ignore
            1,
            self.hidden_sizes,
            device=self.device,
            linear_layer=self.linear_layer,
            flatten_input=self.flatten_input,
        )
        self.last2 = deepcopy(self.last1)


class ActorProb(ActorProb):

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.preprocess_net_output_dim = preprocess_net_output_dim
        super(ActorProb, self).__init__(preprocess_net,
                                           action_shape,
                                           hidden_sizes,
                                           max_action,
                                           device,
                                           unbounded,
                                           conditioned_sigma,
                                           preprocess_net_output_dim)

    def reinitialize_last_layer(self):
        input_dim = get_output_dim(self.preprocess, self.preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim, self.hidden_sizes, device=self.device)


