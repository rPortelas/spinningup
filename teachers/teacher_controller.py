import numpy as np
import pickle
import os
import copy
from teachers.algos.sagg_iac import SAGG_IAC
from teachers.algos.riac import RIAC
from teachers.algos.alp_gmm import ALPGMM
from teachers.algos.covar_gmm import CovarGMM
from teachers.algos.random_teacher import RandomTeacher
from teachers.algos.oracle_teacher import OracleTeacher
from teachers.utils.test_utils import get_test_set_name
from collections import OrderedDict

def get_sorted2d_params(v_min, v_max, eps=1e-3):
    random_2dparams = np.random.uniform(v_min, v_max, 2)
    random_2dparams.sort()
    if np.abs(random_2dparams[1] - random_2dparams[0]) < eps:
        random_2dparams[1] += eps
    return random_2dparams.tolist()

def get_mu_sigma(v_min, v_max): #  assumes sigma has same bounds as mu
    random_2dparams = np.random.uniform(v_min, v_max, 2)
    return random_2dparams.tolist() #  returning mu and sigma

def param_vec_to_param_dict(param_env_bounds, param):
    param_dict = OrderedDict()
    cpt = 0
    for i,(name, bounds) in enumerate(param_env_bounds.items()):
        if len(bounds) == 2:
            param_dict[name] = param[i]
            cpt += 1
        elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
            nb_dims = bounds[2]
            param_dict[name] = param[i:i+nb_dims]
            cpt += nb_dims
    #print('reconstructed param vector {}\n into {}'.format(param, param_dict)) #todo remove
    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):  # needs param_env_bounds for order reference
    param_vec = []
    for name, bounds in param_env_bounds.items():
        #print(param_dict[name])
        param_vec.append(param_dict[name])
    return np.array(param_vec, dtype=np.float32)



class EnvParamsSelector(object):
    def __init__(self, env_babbling, nb_test_episodes, param_env_bounds, seed=None, teacher_params={}):
        self.env_babbling = env_babbling
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)

        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
            if len(bounds) == 2:
                mins.append(bounds[0])
                maxs.append(bounds[1])
            elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                mins.extend([bounds[0]] * bounds[2])
                maxs.extend([bounds[1]] * bounds[2])
            else:
                print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                exit(1)
        #print(mins)
        #print(maxs) # todo remove

        # setup goals generator
        if env_babbling == 'oracle':
            self.goal_generator = OracleTeacher(mins, maxs, teacher_params['window_step_vector'], seed=seed)
        elif env_babbling == 'random':
            self.goal_generator = RandomTeacher(mins, maxs, seed=seed)
        elif env_babbling == 'riac':
            self.goal_generator = RIAC(mins, maxs, seed=seed, params=teacher_params)
        elif env_babbling == 'gmm':
            self.goal_generator = ALPGMM(mins, maxs, seed=seed, params=teacher_params)
        elif env_babbling == 'bmm':
            self.goal_generator = CovarGMM(mins, maxs, seed=seed, params=teacher_params)
        else:
            print('Unknown env babbling')
            raise NotImplementedError

        self.test_mode = "fixed_set"
        if self.test_mode == "fixed_set":
            name = get_test_set_name(self.param_env_bounds)
            self.test_env_list = pickle.load( open("teachers/test_sets/"+name+".pkl", "rb" ) )
            print('fixed set of {} goals loaded: {}'.format(len(self.test_env_list),name))

        #data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if self.env_babbling != 'oracle':
            reward = np.interp(reward, (-150, 350), (0, 1))
            self.env_train_norm_rewards.append(reward)
        self.goal_generator.update(self.env_params_train[-1], reward)

    def record_test_episode(self, reward, ep_len):
        self.env_test_rewards.append(reward)
        self.env_test_len.append(ep_len)

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_params_test': self.env_params_test,
                         'env_test_rewards': self.env_test_rewards,
                         'env_test_len': self.env_test_len,
                         'env_param_bounds': list(self.param_env_bounds.items())}
            dump_dict = self.goal_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def set_env_params(self, env):
        params = copy.copy(self.goal_generator.sample_goal())
        assert type(params[0]) == np.float32
        self.env_params_train.append(params)
        param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
        env.env.set_environment(**param_dict)
        return params

    def set_test_env_params(self, test_env):
        self.test_ep_counter += 1
        if self.test_mode == "fixed_set":
            test_param_dict = self.test_env_list[self.test_ep_counter-1]
        else:
            raise NotImplementedError

        #print('test param dict is: {}'.format(test_param_dict))
        test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
        #print('test param vector is: {}'.format(test_param_vec))

        self.env_params_test.append(test_param_vec)
        test_env.env.set_environment(**test_param_dict)

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0