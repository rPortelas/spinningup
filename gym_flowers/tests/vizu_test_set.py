
import pickle
import gym
import gym_flowers
import time
import numpy as np

env = gym.make('flowers-Walker-continuous-v0') #4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
env.env.my_init({'leg_size':'default'})
env.seed(564564)



test_env_list = pickle.load( open("stump_height0_6.0stump_width0_6.0stump_rot0_6.28obstacle_spacing0_6.0.pkl", "rb" ))

for kwargs in test_env_list:
    if kwargs['stump_height'] is not None:
        random_stump_h = [kwargs['stump_height'], 0.1]
    if kwargs['stump_width'] is not None:
        random_stump_w = [kwargs['stump_width'], 0.1]
    if kwargs['stump_rot'] is not None:
        random_stump_r = [kwargs['stump_rot'], 0.1]
    if kwargs['tunnel_height'] is not None:
        kwargs['tunnel_height'] = [kwargs['tunnel_height'], 0.1]
    random_ob_spacing = kwargs['obstacle_spacing']

    env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                     stump_width=random_stump_w, stump_rot=random_stump_r,
                                     tunnel_height=None, obstacle_spacing=random_ob_spacing,
                                     gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                     step_number=kwargs['step_number'], env_param_input=0)


    env.reset()
    img = env.render(mode='rgb_array')
    time.sleep(0.8)
