import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.space_info import SpaceInfo
from tianshou.data import ReplayBuffer, Collector

@hydra.main(version_base=None, config_path="../config", config_name="config_collect")
def train(conf: DictConfig) -> None:

    logger = hydra.utils.instantiate(conf.logger, _partial_=True)
    logger = logger(config=OmegaConf.to_container(conf, resolve=True))
    logger.load(SummaryWriter(conf.log_path))

    env = gym.make(conf.task)

    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    action_space = env.action_space
    min_action = space_info.action_info.min_action
    max_action = space_info.action_info.max_action
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    print("Action range:", min_action, max_action)

    # seed
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    #env.seed(conf.seed)

    actor = hydra.utils.instantiate(
        conf.policy.actor,
        action_shape=action_shape,
        preprocess_net={"state_shape": state_shape, "device": conf.device},
        device=conf.device,
    ).to(conf.device)
    critic = hydra.utils.instantiate(
        conf.policy.critic,
        preprocess_net={
            "state_shape": state_shape,
            "action_shape": action_shape,
            "device": conf.device,
        },
        device=conf.device,
    ).to(conf.device)
    actor_optim = hydra.utils.instantiate(conf.policy.actor_optim, actor.parameters())
    critic_optim = hydra.utils.instantiate(
        conf.policy.critic_optim, critic.parameters()
    )

    # TODO: Add automatic alpha tuning
    policy = hydra.utils.instantiate(
        conf.policy,
        action_space=action_space,
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
    )

    policy.load_state_dict(torch.load(conf.policy_path))

    buffer: ReplayBuffer = ReplayBuffer(conf.buffer_size)
    collector = Collector(policy, env, buffer)

    collector.collect(n_step=conf.buffer_size, reset_before_collect=True)

    buffer.save_hdf5(f"{conf.log_path}/{conf.task}.hdf5")

    


if __name__ == "__main__":
    train()
