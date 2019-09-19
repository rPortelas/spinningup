import gym
import gym_flowers
import time
import numpy as np

#https://github.com/dgriff777/a3c_continuous/blob/master/environment.py


def time_test(ntest=20):
    start = time.time()
    for i in range(ntest):
        env.reset()
        for j in range(2000):
            env.step(env.action_space.sample())
    end = time.time()
    print("{} episodes --> {} secs".format(ntest,end-start))


def _get_env_params(self, env_params):
    v = []
    if self.roughness:
        v.append(self.roughness)
    if self.stump_height:
        v.extend(self.stump_height)
    if self.gap_width:
        v.extend(self.gap_width)
    if self.step_height and self.step_number:
        v.extend(self.step_height + [self.step_number])
    return v

class MaxMinFilter():
    def __init__(self, env_params_dict=None):
        self.mx_d = 3.15
        self.mn_d = -3.15
        self.new_maxd = 1.0
        self.new_mind = -1.0

        #parameterized env specific
        self.env_p_dict = env_params_dict
        if env_params_dict:
            assert(env_params_dict['roughness'] is None
                   and env_params_dict['gap_width'] is None
                   and env_params_dict['step_height'] is None
                   and env_params_dict['step_number'] is None
                   and env_params_dict['stump_height'] is not None)
            self.min_stump = env_params_dict['stump_height'][0]
            self.max_stump = env_params_dict['stump_height'][1]

    def __call__(self, x):
        new_env_params = []
        if self.env_p_dict:
            env_params = x[0:2].clip(self.min_stump, self.max_stump)
            new_env_params = (((env_params - self.min_stump) * (self.new_maxd - self.new_mind)
                        ) / (self.max_stump - self.min_stump)) + self.new_mind
            x = x[2:]
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / (self.mx_d - self.mn_d)) + self.new_mind
        print(type(new_env_params))
        print(new_obs)
        return np.concatenate((new_env_params, new_obs))

env = gym.make('flowers-Walker-v2') #4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
env.seed(564564)

env_kwargs = {'roughness': None,
              'stump_height': None,
              'gap_width': None,
              'step_height': None,
              'step_number': None}
env.env.set_environment(roughness=None, stump_height=[2.0,0.1], gap_width=None, step_height=None, tunnel_height=None, step_number=None)
time_test()
# for i in range(20):
#     env.reset()
#     env.render()
#     time.sleep(0.1)

# done = False
# norm = MaxMinFilter()
# for j in range(1):
#     start = time.time()
#     env.reset()
#     for i in range(2000):
#         o, r, d, _ = env.step(np.random.rand(4))
#         env.render()
#         n_o = norm(o)
#         print("bef {} af: {}".format(len(o),len(n_o)))
#         print("bef {} af: {}".format(o, n_o))
#     end = time.time()
#     #print(end-start)


