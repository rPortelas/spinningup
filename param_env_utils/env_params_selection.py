import numpy as np
import pickle
import os
import copy
from param_env_utils.active_goal_sampling import SAGG_IAC
from param_env_utils.imgep_utils.riac import RIAC
from param_env_utils.imgep_utils.gmm import InterestGMM
from param_env_utils.imgep_utils.baranes_gmm import BaranesGMM
from param_env_utils.test_utils import get_test_set_name


def get_sorted2d_params(v_min, v_max, eps=1e-3):
    random_2dparams = np.random.uniform(v_min, v_max, 2)
    random_2dparams.sort()
    if np.abs(random_2dparams[1] - random_2dparams[0]) < eps:
        random_2dparams[1] += eps
    return random_2dparams.tolist()

def get_mu_sigma(v_min, v_max): #  assumes sigma has same bounds as mu
    random_2dparams = np.random.uniform(v_min, v_max, 2)
    return random_2dparams.tolist() #  returning mu and sigma

class BaselineGoalGenerator(object):
    def __init__(self, env_babbling, train_env_kwargs):
        self.env_babbling = env_babbling
        self.train_env_kwargs = train_env_kwargs
        if env_babbling == 'oracle':

            self.min_stump_height = 0.0
            self.max_stump_height = 0.5

            self.min_poly_offset = 0.0
            self.max_poly_offset = 0.66

            self.min_ob_spacing = 5
            self.max_ob_spacing = 6

            self.min_seq = 0
            self.max_seq = 1

            self.mutation = 0.1

            self.mutation_rate = 50 #mutate each 50 episodes
            self.mutation_thr = 230 #reward threshold

    def sample_goal(self, kwargs):
        #print(kwargs)
        params = copy.copy(self.train_env_kwargs)
        random_stump_h = None
        random_tunnel_h = None
        random_ob_spacing = None
        random_stump_w = None
        random_stump_r = None
        random_poly_shape = None
        random_stump_seq = None
        if self.env_babbling == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = [np.random.uniform(kwargs['stump_height'][0], kwargs['stump_height'][1]), 0.1]
            if kwargs['stump_width'] is not None:
                random_stump_w = [np.random.uniform(kwargs['stump_width'][0], kwargs['stump_width'][1]), 0.1]
            if kwargs['stump_rot'] is not None:
                random_stump_r = [np.random.uniform(kwargs['stump_rot'][0], kwargs['stump_rot'][1]), 0.1]
            if kwargs['obstacle_spacing'] is not None:
                random_ob_spacing = get_mu_sigma(kwargs['obstacle_spacing'][0], kwargs['obstacle_spacing'][1])[0]
            if kwargs['poly_shape'] is not None:
                random_poly_shape = np.random.uniform(kwargs['poly_shape'][0],
                                                      kwargs['poly_shape'][1], 12+kwargs['nb_rand_dim']).tolist()
            if kwargs['stump_seq'] is not None:
                random_stump_seq = np.random.uniform(kwargs['stump_seq'][0],
                                                      kwargs['stump_seq'][1], 10).tolist()
        elif self.env_babbling == "oracle":
            if kwargs['stump_height'] is not None:
                random_stump_h = [np.random.uniform(self.min_stump_height, self.max_stump_height), 0.1]
            # if kwargs['tunnel_height'] is not None:
            #     random_tunnel_h = get_mu_sigma(self.min_tunnel_height, self.max_tunnel_height)
            #     random_tunnel_h[1] = self.oracle_std
            if kwargs['obstacle_spacing'] is not None:
                random_ob_spacing = np.random.uniform(self.min_ob_spacing, self.max_ob_spacing)
            if kwargs['poly_shape'] is not None:
                random_poly_shape = np.random.uniform(self.min_poly_offset, self.max_poly_offset, 12).tolist()
            if kwargs['stump_seq'] is not None:
                random_stump_seq = np.random.uniform(self.min_seq, self.max_seq, 10).tolist()
        # if (kwargs['stump_height'] is not None) and (kwargs['tunnel_height'] is not None): #if multi dim, fix std
        #     random_stump_h = [random_stump_h[0], 0.3]
        #     random_tunnel_h = [random_tunnel_h[0], 0.3]
        params['stump_height'] = random_stump_h
        params['stump_width'] = random_stump_w
        params['stump_rot'] = random_stump_r
        params['tunnel_height'] = random_tunnel_h
        params['obstacle_spacing'] = random_ob_spacing
        params['poly_shape'] = random_poly_shape
        params['stump_seq'] = random_stump_seq
        return params

    def update(self, goal, reward, env_train_rewards):
        if self.env_babbling == 'oracle':
            if (len(env_train_rewards) % self.mutation_rate) == 0:
                mean_ret = np.mean(env_train_rewards[-50:])
                if mean_ret > self.mutation_thr:
                    if self.train_env_kwargs['stump_height'] is not None:
                        self.min_stump_height = min(self.min_stump_height + self.mutation, self.train_env_kwargs['stump_height'][1] - 0.5)
                        self.max_stump_height = min(self.max_stump_height + self.mutation, self.train_env_kwargs['stump_height'][1])
                        print('mut stump: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_stump_height,
                                                                          self.max_stump_height))
                    if self.train_env_kwargs['obstacle_spacing'] is not None:
                        self.min_ob_spacing = max(0, self.min_ob_spacing - (self.mutation * 2))
                        self.max_ob_spacing = max(1, self.max_ob_spacing - (self.mutation * 2))
                        print('mut ob_spacing: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_ob_spacing,
                                                                           self.max_ob_spacing))
                    if self.train_env_kwargs['poly_shape'] is not None:
                        self.min_poly_offset = min(self.min_poly_offset + self.mutation, self.train_env_kwargs['poly_shape'][1] - 0.66)
                        self.max_poly_offset = min(self.max_poly_offset + self.mutation, self.train_env_kwargs['poly_shape'][1])
                    if self.train_env_kwargs['stump_seq'] is not None:
                        self.min_seq = min(self.min_seq + self.mutation, self.train_env_kwargs['stump_seq'][1] - 1)
                        self.max_seq = min(self.max_seq + self.mutation, self.train_env_kwargs['stump_seq'][1])

    def dump(self, dump_dict):
        return dump_dict

class EnvParamsSelector(object):
    def __init__(self, env_babbling, nb_test_episodes, train_env_kwargs, seed=None, teacher_params={}):
        self.env_babbling = env_babbling
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        self.train_env_kwargs = copy.deepcopy(train_env_kwargs)
        if train_env_kwargs['stump_height'] is not None:
            self.min_stump_height = train_env_kwargs['stump_height'][0]
            self.max_stump_height = train_env_kwargs['stump_height'][1]
        if train_env_kwargs['stump_width'] is not None:
            self.min_stump_width = train_env_kwargs['stump_width'][0]
            self.max_stump_width = train_env_kwargs['stump_width'][1]
        if train_env_kwargs['stump_rot'] is not None:
            self.min_stump_rot = train_env_kwargs['stump_rot'][0]
            self.max_stump_rot = train_env_kwargs['stump_rot'][1]
        if train_env_kwargs['tunnel_height'] is not None:
            self.min_tunnel_height = train_env_kwargs['tunnel_height'][0]
            self.max_tunnel_height = train_env_kwargs['tunnel_height'][1]
        if train_env_kwargs['obstacle_spacing'] is not None:
            self.min_ob_spacing = train_env_kwargs['obstacle_spacing'][0]
            self.max_ob_spacing = train_env_kwargs['obstacle_spacing'][1]

        # figure out parameters boundaries
        mins, maxs = None, None
        if (train_env_kwargs['stump_height'] is not None) and (train_env_kwargs['tunnel_height'] is not None):
            mins = np.array([self.min_stump_height, self.min_tunnel_height])
            maxs = np.array([self.max_stump_height, self.max_tunnel_height])
        elif (train_env_kwargs['stump_height'] is not None) \
                and (train_env_kwargs['obstacle_spacing'] is not None)\
                and (train_env_kwargs['stump_width'] is not None) \
                and (train_env_kwargs['stump_rot'] is not None):
            mins = np.array([self.min_stump_height, self.min_stump_width, self.min_stump_rot, self.min_ob_spacing])
            maxs = np.array([self.max_stump_height, self.max_stump_width, self.max_stump_rot, self.max_ob_spacing])
        elif (train_env_kwargs['stump_height'] is not None) and (train_env_kwargs['obstacle_spacing'] is not None):
            mins = np.array([self.min_stump_height, self.min_ob_spacing])
            maxs = np.array([self.max_stump_height, self.max_ob_spacing])
        elif train_env_kwargs['stump_height'] is not None:
            mins = np.array([self.min_stump_height] * 2)
            maxs = np.array([self.max_stump_height] * 2)
        elif train_env_kwargs['tunnel_height'] is not None:
            mins = np.array([self.min_tunnel_height] * 2)
            maxs = np.array([self.max_tunnel_height] * 2)
        elif train_env_kwargs['poly_shape'] is not None:
            mins = np.array([train_env_kwargs['poly_shape'][0]] * (12 + train_env_kwargs['nb_rand_dim']))
            maxs = np.array([train_env_kwargs['poly_shape'][1]] * (12 + train_env_kwargs['nb_rand_dim']))
        elif train_env_kwargs['stump_seq'] is not None:
            mins = np.array([train_env_kwargs['stump_seq'][0]] * (10))
            maxs = np.array([train_env_kwargs['stump_seq'][1]] * (10))
        else:
            print('Unknown parameters')
            raise NotImplementedError

        # setup goals generator
        if env_babbling == 'oracle' or env_babbling == 'random':
            self.goal_generator = BaselineGoalGenerator(env_babbling, self.train_env_kwargs)
        # elif env_babbling == 'sagg_iac':
        #     self.goal_generator = SAGG_IAC(mins, maxs, seed=seed)
        elif env_babbling == 'riac':
            self.goal_generator = RIAC(mins, maxs, seed=seed, params=teacher_params)
        elif env_babbling == 'gmm':
            self.goal_generator = InterestGMM(mins, maxs, seed=seed, params=teacher_params)
        elif env_babbling == 'bmm':
            self.goal_generator = BaranesGMM(mins, maxs, seed=seed)
        else:
            print('Unknown env babbling')
            raise NotImplementedError

        self.test_mode = "fixed_set" #"levels"
        if self.test_mode == "fixed_set":
            name = get_test_set_name(self.train_env_kwargs)
            self.test_env_list = pickle.load( open("param_env_utils/test_sets/"+name+".pkl", "rb" ) )
            print('fixed set of {} goals loaded: {}'.format(len(self.test_env_list),name))



        #data recording
        self.env_params_train = {'stump_hs':[], 'stump_ws':[], 'stump_rs':[],
                                 'tunnel_hs':[], 'ob_sps':[], 'poly_ss':[], 'seqs':[]}
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []

        self.env_params_test = {'stump_hs':[], 'stump_ws':[], 'stump_rs':[],
                                'tunnel_hs':[], 'ob_sps':[], 'poly_ss':[], 'seqs':[]}
        self.env_test_rewards = []
        self.env_test_len = []

    def get_env_params_vec(self, all_env_params):
        params = []
        if all_env_params['stump_hs'][-1] is not None:
            params += all_env_params['stump_hs'][-1]
        if all_env_params['stump_ws'][-1] is not None:
            params += all_env_params['stump_ws'][-1]
        if all_env_params['tunnel_hs'][-1] is not None:
            params += all_env_params['tunnel_hs'][-1]
        if (all_env_params['stump_hs'][-1] is not None) and (all_env_params['tunnel_hs'][-1] is not None):
            params = [all_env_params['stump_hs'][-1][0], all_env_params['tunnel_hs'][-1][0]]
        if (all_env_params['stump_hs'][-1] is not None) and (all_env_params['ob_sps'][-1] is not None):
            params = [all_env_params['stump_hs'][-1][0], all_env_params['ob_sps'][-1]]
        if (all_env_params['stump_hs'][-1] is not None) \
            and (all_env_params['stump_ws'][-1] is not None) \
            and (all_env_params['stump_rs'][-1] is not None) \
            and (all_env_params['ob_sps'][-1] is not None):
            params = [all_env_params['stump_hs'][-1][0], all_env_params['stump_ws'][-1][0],
                      all_env_params['stump_rs'][-1][0], all_env_params['ob_sps'][-1]]
        if all_env_params['poly_ss'][-1] is not None:
            params = all_env_params['poly_ss'][-1]
        if all_env_params['seqs'][-1] is not None:
            params = all_env_params['seqs'][-1]
        return np.array(params)

    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if (self.env_babbling == 'bmm') or (self.env_babbling == 'gmm') or (self.env_babbling == 'riac'):
            reward = np.interp(reward, (-150, 350), (0, 1))
            self.env_train_norm_rewards.append(reward)
        self.goal_generator.update(self.get_env_params_vec(self.env_params_train),
                                   reward, self.env_train_rewards)


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
                         'env_test_len': self.env_test_len}
            dump_dict = self.goal_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def set_env_params(self, env, kwargs):
        params = self.goal_generator.sample_goal(kwargs)
        if (self.env_babbling == 'gmm') \
            or (self.env_babbling == 'riac')\
            or (self.env_babbling == 'bmm'):
            algo_params = copy.copy(params)
            params = {'tunnel_height':None, 'stump_height':None, 'stump_width':None,
                      'stump_rot':None, 'obstacle_spacing':None, 'poly_shape':None, 'stump_seq':None}
            if (kwargs['stump_height'] is not None) and (kwargs['tunnel_height'] is not None):
                params['stump_height'] = [algo_params[0], 0.3]
                params['tunnel_height'] = [algo_params[1], 0.3]
            elif (kwargs['stump_height'] is not None) \
                    and (kwargs['obstacle_spacing'] is not None) \
                    and (kwargs['stump_width'] is not None) \
                    and (kwargs['stump_rot'] is not None):
                params['stump_height'] = [algo_params[0], 0.1]
                params['stump_width'] = [algo_params[1], 0.1]
                params['stump_rot'] = [algo_params[2], 0.1]
                params['obstacle_spacing'] = algo_params[3]
            elif (kwargs['stump_height'] is not None) and (kwargs['obstacle_spacing'] is not None):
                params['stump_height'] = [algo_params[0], 0.1]
                params['obstacle_spacing'] = algo_params[1]
            elif kwargs['stump_height'] is not None:
                params['stump_height'] = algo_params
            elif kwargs['tunnel_height'] is not None:
                params['tunnel_height'] = algo_params
            elif kwargs['poly_shape'] is not None:
                params['poly_shape'] = algo_params
            elif kwargs['stump_seq'] is not None:
                params['stump_seq'] = algo_params
            else:
                raise NotImplementedError
        self.env_params_train['stump_hs'].append(params['stump_height'])
        self.env_params_train['stump_ws'].append(params['stump_width'])
        self.env_params_train['stump_rs'].append(params['stump_rot'])
        self.env_params_train['tunnel_hs'].append(params['tunnel_height'])
        self.env_params_train['ob_sps'].append(params['obstacle_spacing'])
        self.env_params_train['poly_ss'].append(params['poly_shape'])
        self.env_params_train['seqs'].append(params['stump_seq'])
        env.env.set_environment(roughness=kwargs['roughness'], stump_height=params['stump_height'],
                                stump_width=params['stump_width'], stump_rot=params['stump_rot'],
                                obstacle_spacing=params['obstacle_spacing'],
                                tunnel_height=params['tunnel_height'], poly_shape=params['poly_shape'],
                                stump_seq=params['stump_seq'],
                                gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])
        return params

    def set_test_env_params(self, test_env, kwargs):
        self.test_ep_counter += 1
        #print("test_ep_nb:{}".format(self.test_ep_counter))

        epsilon = 1e-03
        random_stump_h = None
        random_tunnel_h = None
        random_stump_r = None
        random_stump_w = None
        random_ob_spacing = None
        random_poly_shape = None
        random_stump_seq = None

        if self.test_mode == "fixed_set":
            env_args = self.test_env_list[self.test_ep_counter-1]
            if kwargs['stump_height'] is not None:
                random_stump_h = [env_args['stump_height'], 0.1]
            if kwargs['stump_width'] is not None:
                random_stump_w = [env_args['stump_width'], 0.1]
            if kwargs['stump_rot'] is not None:
                random_stump_r = [env_args['stump_rot'], 0.1]
            if kwargs['tunnel_height'] is not None:
                random_tunnel_h = [env_args['tunnel_height'], 0.1]
            if kwargs['obstacle_spacing'] is not None:
                random_ob_spacing = env_args['obstacle_spacing']
            if 'poly_shape' in kwargs and kwargs['poly_shape'] is not None:
                random_poly_shape = env_args['poly_shape']
            if 'stump_seq' in kwargs and kwargs['stump_seq'] is not None:
                random_stump_seq = env_args['stump_seq']

        # elif self.test_mode == "levels":
        #     nb_levels = 3
        #     step = self.nb_test_episodes // nb_levels
        #     step_levels = np.arange(step, self.nb_test_episodes + step, step)
        #     current_level = -1
        #     for i in range(nb_levels):
        #         if self.test_ep_counter <= step_levels[i]:
        #             current_level = i
        #             break
        #     # if (kwargs['stump_height'] is not None) and (kwargs['tunnel_height'] is not None):
        #     #     pass
        #     if kwargs['stump_height'] is not None:
        #         max_stump_height = kwargs['stump_height'][1]
        #         stumph_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
        #         random_stump_h = get_mu_sigma(stumph_levels[current_level][0], stumph_levels[current_level][1])
        #         random_stump_h[1] = 0.1
        #     if kwargs['stump_width'] is not None:
        #         max_stump_width = kwargs['stump_width'][1]
        #         stumpw_levels = [[0.5, 1.0], [1.0, 1.5], [1.5, 2.]]
        #         random_stump_w = get_mu_sigma(stumpw_levels[current_level][0], stumpw_levels[current_level][1])
        #         random_stump_w[1] = 0.1
        #     if kwargs['tunnel_height'] is not None:
        #         max_tunnel_height = kwargs['tunnel_height'][1]
        #         assert(max_tunnel_height == 2)
        #         tunnel_levels = [[1.0, 1.3], [1.3, 1.6], [1.6, 1.9]]
        #         tunnel_levels.reverse() #shorter is harder
        #         random_tunnel_h = get_mu_sigma(tunnel_levels[current_level][0], tunnel_levels[current_level][1])
        #         random_tunnel_h[1] = 0.1
        #     if kwargs['obstacle_spacing'] is not None:
        #         spacing_levels = [[5, 8], [2, 5], [0, 2]]
        #         random_ob_spacing = get_mu_sigma(spacing_levels[current_level][0], spacing_levels[current_level][1])[0]

            if (kwargs['tunnel_height'] is not None) and (kwargs['obstacle_spacing'] is not None):
                # reduced std when both
                random_stump_h[1] = 0.1
        else:
            raise NotImplementedError


        self.env_params_test['stump_hs'].append(random_stump_h)
        self.env_params_test['stump_ws'].append(random_stump_w)
        self.env_params_test['tunnel_hs'].append(random_tunnel_h)
        self.env_params_test['stump_rs'].append(random_stump_r)
        self.env_params_test['ob_sps'].append(random_ob_spacing)
        self.env_params_test['poly_ss'].append(random_poly_shape)
        self.env_params_test['seqs'].append(random_stump_seq)
        test_env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                     stump_width=random_stump_w, stump_rot=random_stump_r, poly_shape=random_poly_shape,
                                     stump_seq=random_stump_seq,
                                     tunnel_height=random_tunnel_h, obstacle_spacing=random_ob_spacing,
                                     gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                     step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0