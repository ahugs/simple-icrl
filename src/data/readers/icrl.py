import os
import pickle
import numpy as np
from tianshou.data import ReplayBuffer, Batch


class ICRLReader:
    """
    Read data from HdF5 file (in D4RL format) and add to ReplayBuffer.
    """
    def __init__(self, num_episodes: int = None):
        self.num_episodes = num_episodes

    def load(self, path: str,) -> ReplayBuffer:

        file_names = sorted(os.listdir(path))
        if self.num_episodes is None or self.num_episodes > len(file_names):
            num_episodes = len(file_names)
        else:
            num_episodes = self.num_episodes

        obs_all = []
        acs_all = []
        rew_all = []
        obs_next_all = []
        truncated_all = []
        terminated_all = []


        for i in range(num_episodes):
            # file_name = sample_names[i]
            file_name = file_names[i]
            with open(os.path.join(path, file_name), "rb") as f:
                data = pickle.load(f)
            obs = data['original_observations']
            acs = data['actions']
            if 'reward' in data.keys():
                rew = data['reward']
            else:
                rew = np.zeros(obs.shape[0])

            obs_next = obs[1:, ...]
            obs = obs[:-1, ...]
            acs = acs[:-1, ...]
            rew = rew[:-1, ...]
            terminated = np.zeros_like(rew)
            truncated = np.zeros_like(rew)
            truncated[-1] = 1

            obs_all.append(obs)
            acs_all.append(acs)
            rew_all.append(rew)
            obs_next_all.append(obs_next)
            terminated_all.append(terminated)
            truncated_all.append(truncated)

        obs = np.concatenate(obs_all, axis=0)
        acs = np.concatenate(acs_all, axis=0)
        rew = np.concatenate(rew_all, axis=0)
        obs_next = np.concatenate(obs_next_all, axis=0)
        terminated = np.concatenate(terminated_all, axis=0)
        truncated = np.concatenate(truncated_all, axis=0)

        dataset_size = rew.size

        expert_buffer = ReplayBuffer(dataset_size)

        for i in range(dataset_size):
            expert_buffer.add(
                Batch(
                    obs=obs[i],
                    act=acs[i],
                    rew=rew[i],
                    terminated=terminated[i],
                    truncated=truncated[i],
                    obs_next=obs_next[i],
                )
            )

        return expert_buffer
