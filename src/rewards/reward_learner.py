import torch
from tqdm import tqdm
import numpy as np


class RewardLearner:

    def __init__(
        self,
        net,
        optim,
        expert_buffer,
        steps_per_epoch=5,
        batch_size=64,
        learn_true_rewards=False,
        regularization_coeff=1,
    ):
        self.net = net
        self.optim = optim
        self.expert_buffer = expert_buffer
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.regularization_coeff = regularization_coeff
        self.learn_true_rewards = learn_true_rewards

    def update(self, learner_buffer):
        self.net.train()

        for _ in (pbar := tqdm(range(self.steps_per_epoch))):

            expert_batch, _ = self.expert_buffer.sample(self.batch_size)
            learner_batch, _ = learner_buffer.sample(self.batch_size)

            input_size = next(self.net.parameters()).size()
            concat = False
            if expert_batch.obs.shape[-1] < input_size[-1]:
                concat = True
            expert_input = np.concatenate(
                [expert_batch.obs, expert_batch.act if concat else []],
                axis=1,
            )
            learner_input = np.concatenate(
                [learner_batch.obs, learner_batch.act if concat else []],
                axis=1,
            )
            learner = self.net(learner_input)
            expert = self.net(expert_input)

            if self.learn_true_rewards:
                loss = torch.nn.MSELoss()(learner, torch.FloatTensor(learner_batch.rew).to(learner.device))
                loss += torch.nn.MSELoss()(expert, torch.FloatTensor(expert_batch.rew).to(expert.device))
            else:
                loss = learner.mean() - expert.mean()
                num_data = expert.shape[0] + learner.shape[0]
                loss += (
                    self.regularization_coeff
                    * (torch.sum(expert**2) + torch.sum(learner**2))
                    / num_data
                )
            pbar.set_description(f"Reward Loss {loss.item()}")

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.net.eval()
        return {'loss': loss.item(),
                'learner_reward': learner.mean().item(),
                'expert_reward': expert.mean().item()}
