import h5py
import numpy as np
from tianshou.data import ReplayBuffer, Batch


class D4RLReader:
    """
    Read data from HdF5 file (in D4RL format) and add to ReplayBuffer.
    """
    def __init__(self, num_episodes: int = None):
        self.num_episodes = num_episodes

    def load(self, filename: str,) -> ReplayBuffer:
        with h5py.File(filename, "r") as dataset:
            obs = dataset["observations"][:]
            act = dataset["actions"][:]
            rew = dataset["rewards"][:]
            terminated = dataset["terminals"][:]
            truncated = dataset["timeouts"][:]
            obs_next = dataset["next_observations"][:]
        if self.num_episodes is not None:
            end_index = np.where((terminated + truncated).cumsum() >= self.num_episodes)[0][0] + 1
            obs = obs[:end_index, ...]
            act = act[:end_index, ...]
            rew = rew[:end_index, ...]
            terminated = terminated[:end_index, ...]
            truncated = truncated[:end_index, ...]
            obs_next = obs_next[:end_index, ...]
        
        dataset_size = rew.size

        expert_buffer = ReplayBuffer(dataset_size)

        for i in range(dataset_size):
            expert_buffer.add(
                Batch(
                    obs=obs[i],
                    act=act[i],
                    rew=rew[i],
                    terminated=terminated[i],
                    truncated=truncated[i],
                    obs_next=obs_next[i],
                )
            )
        return expert_buffer
