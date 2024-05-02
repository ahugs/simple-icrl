import numpy as np
import torch
from dataclasses import dataclass

from tianshou.data import ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import BasePolicy, SACPolicy, PPOPolicy
from tianshou.policy.modelfree.sac import SACTrainingStats

from typing import TypeVar, Any


@dataclass(kw_only=True)
class IRLSACTrainingStats(SACTrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha: float | None = None
    alpha_loss: float | None = None
    average_rewards: float | None = None


TIRLSACTrainingStats = TypeVar("TIRLSACTrainingStats", bound=IRLSACTrainingStats)


class IRLBasePolicy(BasePolicy):

    def __init__(self, reward_net, *args,  additive_reward=False, **kwargs):
        super(IRLBasePolicy, self).__init__(*args, **kwargs)
        self.reward_net = reward_net
        self.additive_reward = additive_reward

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
            orig_rew = batch.rew
            batch.rew = (
                self.reward_net(input).reshape(batch_size, -1).detach().cpu().numpy().squeeze()
            )
            if self.additive_reward:
                batch.rew += orig_rew
            batch.info.orig_rew = orig_rew

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
            orig_rew = buffer.rew
            buffer.rew[nstep_indices] = (
                self.reward_net(input).reshape(len(nstep_indices)).detach().cpu().numpy().squeeze()
            )
            if self.additive_reward:
                buffer.rew[nstep_indices] += orig_rew[nstep_indices]
            buffer.info.orig_rew = orig_rew

        batch = super().process_fn(batch, buffer, indices)
        return batch

    def post_process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:
        
        batch.rew[:] = batch.info.orig_rew
        buffer.rew[:] = buffer.info.orig_rew

        return batch


    def learn(
        self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any
    ) -> TIRLSACTrainingStats:
        average_rewards = batch.rew.mean().item()
        stats = super().learn(batch, *args, **kwargs)
        return IRLSACTrainingStats(
            **{
                **stats._get_self_dict(),
                "average_rewards": average_rewards,
            }
        )

    def update_reward_net(self, reward_net):
        self.reward_net = reward_net

    def pre_update_fn(self, stats_train, batch_size, buffer, update_per_step):
        return
    
    def post_update_fn(self, stats_train):
        return


class IRLSACPolicy(IRLBasePolicy, SACPolicy):
    def __init__(self, *args, **kwargs):
        super(IRLSACPolicy, self).__init__(*args, **kwargs)


class IRLPPOPolicy(IRLBasePolicy, PPOPolicy):
    def __init__(self, *args, **kwargs):
        super(IRLPPOPolicy, self).__init__(*args, **kwargs)
