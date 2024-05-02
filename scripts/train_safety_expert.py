import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.space_info import SpaceInfo
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs
from mujoco_env import make_mujoco_env
from tianshou.data import ReplayBuffer, VectorReplayBuffer, Collector

from src.wrappers.gym_wrappers import CostWrapper


@hydra.main(version_base=None, config_path="../config", config_name="config_safety_expert")
def train(conf: DictConfig) -> None:

    logger = hydra.utils.instantiate(conf.logger, _partial_=True)
    logger = logger(config=OmegaConf.to_container(conf, resolve=True))
    logger.load(SummaryWriter(conf.log_path))

    env, train_envs, test_envs = make_mujoco_env(
        conf.task,
        conf.seed,
        conf.train_env_num,
        conf.test_env_num,
        obs_norm=False,
    )

    if np.isscalar(conf.cost_limit):
        cost_dim = 1
    else:
        cost_dim = len(conf.cost_limit)

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
    train_envs.seed(conf.seed)
    test_envs.seed(conf.seed)

    actor = hydra.utils.instantiate(
        conf.policy.actor,
        action_shape=action_shape,
        preprocess_net={"state_shape": state_shape, "device": conf.device},
        device=conf.device,
    ).to(conf.device)

    critics = []
    for critic in conf.policy.critics:
        critic = hydra.utils.instantiate(
            critic,
            preprocess_net={
                "state_shape": state_shape,
                "action_shape": action_shape,
                "device": conf.device,
            },
            device=conf.device,
        ).to(conf.device)
        critics.append(critic)

    actor_optim = hydra.utils.instantiate(conf.policy.actor_optim, actor.parameters())
    critic_optim = hydra.utils.instantiate(
        conf.policy.critic_optim, torch.nn.ModuleList(critics).parameters()
    )

    # TODO: Add automatic alpha tuning
    policy = hydra.utils.instantiate(
        conf.policy,
        action_space=action_space,
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
    )

    train_buffer: ReplayBuffer
    if conf.train_env_num > 1:
        train_buffer = VectorReplayBuffer(conf.train_buffer_size, len(train_envs))
    else:
        train_buffer = ReplayBuffer(conf.train_buffer_size)
    train_collector = Collector(
        policy, train_envs, train_buffer, exploration_noise=True
    )

    test_buffer: ReplayBuffer
    if conf.test_env_num > 1:
        test_buffer = VectorReplayBuffer(conf.test_buffer_size, len(train_envs))
    else:
        test_buffer = ReplayBuffer(conf.test_buffer_size)
    test_collector = Collector(policy, test_envs, test_buffer)

    def save_best_fn(policy) -> None:
        torch.save(policy.state_dict(), conf.log_path + "/policy.pth")

    trainer = hydra.utils.instantiate(
        conf.trainer,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector if conf.trainer.episode_per_test > 0 else None,
        logger=logger,
        save_best_fn=save_best_fn,
    )

    train_collector.collect(n_step=conf.warmstart_timesteps, random=True, reset_before_collect=True)
    trainer.run()


if __name__ == "__main__":
    train()
