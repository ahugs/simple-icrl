import numpy as np
import torch
from dataclasses import dataclass

from tianshou.data import ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import BasePolicy, SACPolicy, PPOPolicy
from tianshou.policy.modelfree.sac import SACTrainingStats
from fsrl.policy import SACLagrangian, PPOLagrangian
from typing import TypeVar, Any


@dataclass(kw_only=True)
class ICLSACTrainingStats(SACTrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha: float | None = None
    alpha_loss: float | None = None
    average_costs: float | None = None


TICLSACTrainingStats = TypeVar("TICLSACTrainingStats", bound=ICLSACTrainingStats)


class ICLBasePolicy(BasePolicy):

    def __init__(self, reward_net, *args, **kwargs):
        super(ICLBasePolicy, self).__init__(*args, **kwargs)
        self.constraint_net = reward_net

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        # Add new reward to the batch
        batch_size = batch.obs_next.shape[0]
        input_size = next(self.reward_net.parameters()).size()
        concat = False
        if batch.obs.shape[-1] < input_size[-1]:
            concat = True
        input = np.concatenate(
            [batch.obs, batch.act if concat else []],
            axis=1,
        )

        with torch.no_grad():
            batch.cost = (
                self.constraint_net(input).reshape(batch_size, -1).detach().cpu().numpy()
            )

        nstep_indices = [indices]
        for _ in range(self.estimation_step - 1):
            nstep_indices.append(buffer.next(nstep_indices[-1]))
        nstep_indices = np.unique(np.stack(nstep_indices).flatten())
        input = np.concatenate(
            [
                buffer.obs[nstep_indices, :],
                buffer.act[nstep_indices, :] if concat else [],
            ],
            axis=1,
        )

        with torch.no_grad():
            buffer.cost[nstep_indices] = (
                self.constraint_net(input).reshape(len(nstep_indices)).detach().cpu().numpy()
            )

        batch = super().process_fn(batch, buffer, indices)
        return batch

    def learn(
        self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any
    ) -> TIRLSACTrainingStats:
        average_costs = batch.cost.mean().item()
        stats = super().learn(batch, *args, **kwargs)
        return ICLSACTrainingStats(
            **{
                **stats._get_self_dict(),
                "average_costss": average_costs,
            }
        )

    def update_constraint_net(self, constraint_net):
        self.constraint_net = constraint_net


class ICLSACPolicy(ICLBasePolicy, SACLagrangian):
    def __init__(self, *args, **kwargs):
        super(ICLSACPolicy, self).__init__(*args, **kwargs)


class ICLPPOPolicy(ICLBasePolicy, PPOLagrangian):
    def __init__(self, *args, **kwargs):
        super(ICLPPOPolicy, self).__init__(*args, **kwargs)
