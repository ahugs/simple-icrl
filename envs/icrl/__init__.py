from gymnasium.envs.registration import register


# =========================================================================== #
#                              Inverted Pendulum                              #
# =========================================================================== #

PENDULUM_LEN = 100

register(
    id="InvertedPendulumWall-v0",
    entry_point="icrl.inverted_pendulum:InvertedPendulumWall",
    max_episode_steps=PENDULUM_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                   Cheetah                                   #
# =========================================================================== #

CHEETAH_LEN = 1000

register(
    id="HalfCheetahWall-v0",
    entry_point="icrl.half_cheetah:HalfCheetahWall",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'exclude_current_positions_from_observation': False}
)

# =========================================================================== #
#                                   Ant                                       #
# =========================================================================== #

ANT_LEN = 500

register(
    id="AntWall-v0",
    entry_point="icrl.ant:AntWall",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'exclude_current_positions_from_observation': False,
            'use_contact_forces': True}
)


# =========================================================================== #
#                                  Swimmer                                    #
# =========================================================================== #

SWIMMER_LEN = 500

register(
    id="SwimmerWall-v0",
    entry_point="icrl.swimmer:SwimmerWall",
    max_episode_steps=SWIMMER_LEN,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'exclude_current_positions_from_observation': False}
)

# =========================================================================== #
#                                   Walker                                    #
# =========================================================================== #

WALKER_LEN = 500

register(
    id="Walker2dWall-v0",
    entry_point="icrl.walker2d:Walker2dWall",
    max_episode_steps=WALKER_LEN,
    reward_threshold=None,
    nondeterministic=False,
    kwargs={'exclude_current_positions_from_observation': False}
)