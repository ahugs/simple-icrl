import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.space_info import SpaceInfo
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs
from tianshou.data import ReplayBuffer, VectorReplayBuffer, Collector

from src.wrappers.gym_wrappers import NoRewardEnv
from mujoco_env import make_mujoco_env
import envs


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(conf: DictConfig) -> None:
    logger = hydra.utils.instantiate(conf.logger, _partial_=True)
    logger = logger(config=OmegaConf.to_container(conf, resolve=True))
    logger.load(SummaryWriter(conf.log_path))

    # env = gym.make(conf.task)
    env, train_envs, test_envs = make_mujoco_env(
        conf.task,
        conf.seed,
        conf.train_env_num,
        conf.test_env_num,
        obs_norm=False,
    )
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

    reward_net = hydra.utils.instantiate(
        conf.reward.net,
        preprocess_net={
            "state_shape": state_shape,
            "action_shape": action_shape,
            "device": conf.device,
        },
        device=conf.device,
    ).to(conf.device)

    policy = conf.policy.copy()

    if not np.isscalar(conf.policy.alpha):
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=conf.device)
        alpha_optim = hydra.utils.instantiate(conf.policy.alpha.optim, [log_alpha])
        alpha = (target_entropy, log_alpha, alpha_optim)
        del policy.alpha
    else:
        alpha = conf.policy.alpha

    policy = hydra.utils.instantiate(
        policy,
        reward_net=reward_net,
        action_space=action_space,
        actor=actor,
        critic=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        alpha=alpha
    )

    reward_optim = hydra.utils.instantiate(conf.reward.optim, reward_net.parameters())
    expert_buffer = hydra.utils.instantiate(conf.data).load(
        conf.expert_data_path
    )
    irl = hydra.utils.instantiate(
        conf.reward, optim=reward_optim, net=reward_net, expert_buffer=expert_buffer
    )
    train_buffer: ReplayBuffer
    if conf.train_env_num > 1:
        train_buffer = VectorReplayBuffer(conf.train_buffer_size, len(train_envs))
    else:
        train_buffer = ReplayBuffer(conf.train_buffer_size)
    train_collector = hydra.utils.instantiate(
        conf.collector, policy=policy, env=train_envs, buffer=train_buffer
    )

    test_buffer: ReplayBuffer
    if conf.test_env_num > 1:
        test_buffer = VectorReplayBuffer(conf.test_buffer_size, len(test_envs))
    else:
        test_buffer = ReplayBuffer(conf.test_buffer_size)
    test_collector = hydra.utils.instantiate(
        conf.collector, policy=policy, env=test_envs, buffer=test_buffer
    )
    trainer = hydra.utils.instantiate(
        conf.trainer,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector if conf.trainer.episode_per_test > 0 else None,
        logger=logger,
    )

    train_collector.collect(n_step=conf.warmstart_timesteps, random=True)
    for epoch_stat in trainer:
        trainer.policy.update_reward_net(irl.net)
        test_collector.policy.eval()
        test_collector.collect(n_episode=5)
        stats = irl.update(test_collector.buffer)
        logger.write("reward/env_step", trainer.env_step, logger.prepare_dict_for_logging(stats, parent_key="reward"))
        test_collector.reset()


if __name__ == "__main__":
    train()
