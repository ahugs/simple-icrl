import numpy as np
import torch
from dataclasses import dataclass

from tianshou.data import ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy.modelfree.sac import SACTrainingStats, SACPolicy
from tianshou.utils import RunningMeanStd
from fsrl.policy import SACLagrangian, PPOLagrangian, BasePolicy
from typing import TypeVar, Any, List, Dict
import hydra

from src.utils.optim import LagrangianOptimizer
@dataclass(kw_only=True)
class ICRLSACTrainingStats(SACTrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha: float | None = None
    alpha_loss: float | None = None
    average_costs: float | None = None


TICRLSACTrainingStats = TypeVar("TICRLSACTrainingStats", bound=ICRLSACTrainingStats)


class ICRLBasePolicy(BasePolicy):

    def __init__(self, constraint_net, lagrangian, lagrangian_optim=None, *args, normalize_reward=False, **kwargs):
        critics = kwargs.pop("critics")
        critics = list(critics)
        alpha = kwargs.pop("alpha")
        try:
            len(alpha)
            alpha = tuple(alpha)
        except:
            pass
        super(ICRLBasePolicy, self).__init__(*args, critics=critics, alpha=alpha, **kwargs)
        self.constraint_net = constraint_net
        self.lagrangian = lagrangian
        if normalize_reward:
            self.rms = RunningMeanStd()
        else:
            self.rms = None

        if self.use_lagrangian and lagrangian_optim is not None:
            assert len(
                self.cost_limit
            ) == (self.critics_num - 1), "cost_limit must has equal len of critics_num"
            self.lag_optims = [
                hydra.utils.instantiate(lagrangian_optim) for _ in range(self.critics_num - 1)
            ]

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        # Add new constraint to the batch
        batch_size = batch.obs_next.shape[0]
        input_size = next(self.constraint_net.parameters()).size()
        concat = False
        if batch.obs.shape[-1] < input_size[-1]:
            concat = True
        input = np.concatenate(
            [batch.obs, batch.act if concat else []],
            axis=1,
        )

        with torch.no_grad():
            batch.info.cost = batch.info.cost.astype(np.float32)
            batch.info.orig_cost = batch.info.cost.copy()
            if self.rms is not None:
                batch.info.orig_rew = batch.rew.copy()
                self.rms.update(batch.rew)
                batch.rew = self.rms.norm(batch.rew)
            batch.info.cost = (
                self.constraint_net(input).reshape(batch_size, -1).detach().cpu().numpy().squeeze()
            )

        nstep_indices = [indices]
        for _ in range(self._n_step - 1):
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
            buffer.info.cost = buffer.info.cost.astype(np.float32)
            buffer.info.orig_cost = buffer.info.cost.copy()
            if self.rms is not None:
                buffer.info.orig_rew = buffer.rew.copy()
                buffer.rew[nstep_indices] = self.rms.norm(buffer.rew[nstep_indices])
            buffer.info.cost[nstep_indices] = (
                self.constraint_net(input).reshape(len(nstep_indices)).detach().cpu().numpy().squeeze()
            )

        batch = super().process_fn(batch, buffer, indices)
        return batch
    
    def post_process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        
        batch.info.cost[:] = batch.info.orig_cost
        buffer.info.cost[:] = buffer.info.orig_cost
        if self.rms is not None:
            batch.rew[:] = batch.info.orig_rew
            buffer.rew[:] = buffer.info.orig_rew

        return batch

    def pre_update_fn(self, stats_train: Dict, **kwarg: Any) -> None:
        if self.lagrangian is not None:
            for optim in self.lag_optims:
                optim.lagrangian = self.lagrangian
            return 
        else:
            cost_values = stats_train["learned_cost"]
            self.update_lagrangian(cost_values)
            return

    def update_constraint_net(self, constraint_net):
        self.constraint_net = constraint_net

    def reinitialize_last_layers(self):
        self.actor.reinitialize_last_layer()
        for critic in self.critics[1:]:
            critic.reinitialize_last_layer()


class ICRLSACPolicy(ICRLBasePolicy, SACLagrangian):
    def __init__(self, *args, **kwargs):
        super(ICRLSACPolicy, self).__init__(*args, **kwargs)

class ICRLPPOPolicy(ICRLBasePolicy, PPOLagrangian):
    def __init__(self, *args, **kwargs):
        super(ICRLPPOPolicy, self).__init__(*args, **kwargs)
