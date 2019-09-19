import itertools
from gym.envs.registration import register
import numpy as np

register(
    id='flowers-Walker-continuous-v0',
    entry_point='gym_flowers.envs:BipedalWalkerContinuous',
    max_episode_steps=2000,
    reward_threshold=300,
)

# register(
#     id='flowers-Walker-movement-v0',
#     entry_point='gym_flowers.envs.box2d:BipedalWalkerMovement',
#     max_episode_steps=2000,
#     reward_threshold=300,
# )