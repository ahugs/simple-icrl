import h5py
import numpy as np
from tianshou.data import ReplayBuffer, Batch


class DummyReader:
    """
    Read data from HdF5 file (in D4RL format) and add to ReplayBuffer.
    """
    def __init__(self, action_shape, obs_shape, num_episodes: int = None):
        self.num_episodes = num_episodes
        self.action_shape = action_shape
        self.obs_shape = obs_shape

    def load(self) -> ReplayBuffer:

        expert_buffer = ReplayBuffer(100)

        for i in range(100):
            expert_buffer.add(
                Batch(
                    obs=np.zeros(self.obs_shape),
                    act=np.zeros(self.action_shape),
                    rew=np.array([0]),
                    terminated=np.array([0]),
                    truncated=np.array([0]),
                    obs_next=np.zeros(self.obs_shape),
                )
            )
        return expert_buffer
