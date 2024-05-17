import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.space_info import SpaceInfo
from src.envs.mujoco_env import make_mujoco_env
from tianshou.data import ReplayBuffer, VectorReplayBuffer, Collector
import icrl


@hydra.main(version_base=None, config_path="../config", config_name="config_icrl")
def train(conf: DictConfig) -> None:

    import icrl 
    
    logger = hydra.utils.instantiate(conf.logger, _partial_=True)
    logger = logger(
        config=OmegaConf.to_container(conf, resolve=True), log_dir=conf.log_path
    )

    cost_fn = hydra.utils.instantiate(conf.cost_fn, _partial_=True)
    cost_wrapper = hydra.utils.instantiate(
        {'_target_': 'src.wrappers.gym_wrappers.CostWrapper'}, cost_fn=cost_fn, _partial_=True
    )

    env, train_envs, test_envs = make_mujoco_env(
        conf.task,
        conf.seed,
        conf.train_env_num,
        conf.test_env_num,
        obs_norm=conf.normalize_obs,
        rew_norm=False,
        wrappers=[cost_wrapper],
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

    critics = []
    for critic in conf.policy.critics:
        if 'critic' in conf.policy.keys():
            critic = OmegaConf.merge(critic, conf.policy.critic)
        critic = hydra.utils.instantiate(
            critic,
            preprocess_net1={
                "state_shape": state_shape,
                "action_shape": action_shape,
                "device": conf.device,
            },
            preprocess_net2={
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

    constraint_net = hydra.utils.instantiate(
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

    if 'critic' in conf.policy.keys():
        del policy.critic
    policy = hydra.utils.instantiate(
        policy,
        constraint_net=constraint_net,
        action_space=action_space,
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        alpha=alpha
    )

    constraint_optim = hydra.utils.instantiate(conf.reward.optim, constraint_net.parameters())
    constraint_lr_scheduler = hydra.utils.instantiate(conf.reward.lr_scheduler, constraint_optim)

    expert_buffer = hydra.utils.instantiate(conf.data).load(conf.expert_data_path)
    irl = hydra.utils.instantiate(
        conf.reward, optim=constraint_optim, net=constraint_net, expert_buffer=expert_buffer,
        lr_scheduler=constraint_lr_scheduler
    )

    train_buffer: ReplayBuffer
    if conf.train_env_num > 1:
        train_buffer = VectorReplayBuffer(conf.train_buffer_size, len(train_envs))
    else:
        train_buffer = ReplayBuffer(conf.train_buffer_size)
    train_collector = hydra.utils.instantiate(
        conf.train_collector, policy=policy, env=train_envs, buffer=train_buffer
    )

    test_buffer: ReplayBuffer
    if conf.test_env_num > 1:
        test_buffer = VectorReplayBuffer(conf.test_buffer_size, len(test_envs))
    else:
        test_buffer = ReplayBuffer(conf.test_buffer_size)
    test_collector = hydra.utils.instantiate(
        conf.test_collector, policy=policy, env=test_envs, buffer=test_buffer
    )

    trainer = hydra.utils.instantiate(
        conf.trainer,
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector if conf.trainer.episode_per_test > 0 else None,
        logger=logger,
    )

    train_collector.collect(n_episode=conf.warmstart_episodes, random=True)
    for epoch_stat in trainer:
        trainer.policy.update_constraint_net(irl.net)
        test_collector.policy.eval()
        stats = irl.update(test_collector.buffer)
        logger.store(tab="reward", **stats)
        logger.write(
            trainer.env_step,
        )
        test_collector.reset()
        if conf.reset_policy:
            trainer.policy.reinitialize_last_layers()

if __name__ == "__main__":
    train()
