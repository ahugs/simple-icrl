from gymnasium.envs.registration import register

#
# Modular RL
#

# Cheetah

register(
    id='modular_rl/cheetah-7-full-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_7_full.xml'}
)

register(
    id='modular_rl/cheetah-6-front-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_6_front.xml'}
)

register(
    id='modular_rl/cheetah-6-back-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_6_back.xml'}
)

register(
    id='modular_rl/cheetah-5-front-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_5_front.xml'}
)

register(
    id='modular_rl/cheetah-5-balanced-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_5_balanced.xml'}
)

register(
    id='modular_rl/cheetah-5-back-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_5_back.xml'}
)

register(
    id='modular_rl/cheetah-4-front-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_4_front.xml'}
)

register(
    id='modular_rl/cheetah-4-back-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_4_back.xml'}
)

register(
    id='modular_rl/cheetah-4-allfront-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_4_allfront.xml'}
)

register(
    id='modular_rl/cheetah-4-allback-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_4_allback.xml'}
)

register(
    id='modular_rl/cheetah-3-front-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_3_front.xml'}
)

register(
    id='modular_rl/cheetah-3-balanced-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_3_balanced.xml'}
)

register(
    id='modular_rl/cheetah-3-back-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_3_back.xml'}
)

register(
    id='modular_rl/cheetah-2-front-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_2_front.xml'}
)

register(
    id='modular_rl/cheetah-2-back-v0',
    entry_point='envs.modular_rl.cheetah:ModularCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={'xml_file': 'assets/cheetah_2_back.xml'}
)
