from gymnasium.envs.registration import register


# =========================================================================== #
#                              Inverted Pendulum                              #
# =========================================================================== #

PENDULUM_LEN = 100

register(
    id="InvertedPendulumWall-v0",
    entry_point="envs.icrl.inverted_pendulum:InvertedPendulumWall",
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
    entry_point="envs.icrl.half_cheetah:HalfCheetahWall",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)