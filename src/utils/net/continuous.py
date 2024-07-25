from abc import ABC
from collections.abc import Sequence
from typing import Any, List, Union, Optional, Type, Callable
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
from tianshou.utils.net.continuous import ActorProb, Critic
from fsrl.utils.net.continuous import DoubleCritic

def reset_child_params(module):
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_child_params(layer)

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
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
        assert not (output_scaling != 1.0 and output_transform is not None)
        if output_scaling != 1.0:
            self.output_transform = lambda x: x * output_scaling
        elif output_transform is not None:
            self.output_transform = output_transform
        else: 
            self.output_transform = nn.Identity()
        if clip_range is None:
            self.clip_range = [-np.inf, np.inf]
        else:
            self.clip_range = clip_range

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        values_B, hidden_BH = self.preprocess(obs)
        output = self.output_transform(self.output_activation(self.last(values_B)))
        return torch.clamp(output, min=self.clip_range[0], max=self.clip_range[1])
    

class DoubleCritic(DoubleCritic):

    def reinitialize_last_layer(self):
        reset_child_params(self.last1)
        reset_child_params(self.last2)


class ActorProb(ActorProb):

    def reinitialize_last_layer(self):
        reset_child_params(self.mu)
        try:
            reset_child_params(self.sigma)
        except:
            reset_child_params(self.sigma_param)


class Critic(Critic):

    def reinitialize_last_layer(self):
        reset_child_params(self.last)



