import numpy as np
import pickle
import copy
from param_env_utils.active_goal_sampling import SAGG_IAC
from param_env_utils.imgep_utils.gmm import InterestGMM


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

            self.min_tunnel_height = 1.5
            self.max_tunnel_height = 2.0

            self.min_ob_spacing = 5
            self.max_ob_spacing = 6

            self.mutation = 0.1
            self.oracle_std = 0.3

            self.mutation_rate = 50 #mutate each 50 episodes
            self.mutation_thr = 230 #reward threshold

    def sample_goal(self, kwargs):
        #print(kwargs)
        params = copy.copy(self.train_env_kwargs)
        random_stump_h = None
        random_tunnel_h = None
        random_ob_spacing = None
        if self.env_babbling == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = get_mu_sigma(kwargs['stump_height'][0], kwargs['stump_height'][1])
            if kwargs['tunnel_height'] is not None:
                random_tunnel_h = get_mu_sigma(kwargs['tunnel_height'][0], kwargs['tunnel_height'][1])
            if kwargs['obstacle_spacing'] is not None:
                random_ob_spacing = get_mu_sigma(kwargs['obstacle_spacing'][0], kwargs['obstacle_spacing'][1])[0]
        elif self.env_babbling == "oracle":
            if kwargs['stump_height'] is not None:
                random_stump_h = get_mu_sigma(self.min_stump_height, self.max_stump_height)
                random_stump_h[1] = self.oracle_std
            if kwargs['tunnel_height'] is not None:
                random_tunnel_h = get_mu_sigma(self.min_tunnel_height, self.max_tunnel_height)
                random_tunnel_h[1] = self.oracle_std
            if kwargs['obstacle_spacing'] is not None:
                random_ob_spacing = get_mu_sigma(self.min_ob_spacing, self.max_ob_spacing)[0]
        if (kwargs['stump_height'] is not None) and (kwargs['tunnel_height'] is not None): #if multi dim, fix std
            random_stump_h = [random_stump_h[0], 0.3]
            random_tunnel_h = [random_tunnel_h[0], 0.3]
        if (kwargs['stump_height'] is not None) and (kwargs['obstacle_spacing'] is not None):
            random_stump_h = [random_stump_h[0], 0.1]
        params['stump_height'] = random_stump_h
        params['tunnel_height'] = random_tunnel_h
        params['obstacle_spacing'] = random_ob_spacing
        return params

    def update(self, goal, reward, env_train_rewards):
        if self.env_babbling == 'oracle':
            if (len(env_train_rewards) % self.mutation_rate) == 0:
                mean_ret = np.mean(env_train_rewards[-50:])
                if mean_ret > self.mutation_thr:
                    if self.train_env_kwargs['stump_height'] is not None:
                        self.min_stump_height += self.mutation
                        self.max_stump_height += self.mutation
                        print('mut stump: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_stump_height,
                                                                          self.max_stump_height))
                    if self.train_env_kwargs['tunnel_height'] is not None:
                        self.min_tunnel_height -= self.mutation/2
                        self.max_tunnel_height -= self.mutation/2
                        print('mut tunnel: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_tunnel_height,
                                                                           self.max_tunnel_height))
                    if self.train_env_kwargs['obstacle_spacing'] is not None:
                        self.min_ob_spacing -= self.mutation * 3
                        self.max_ob_spacing -= self.mutation * 3
                        print('mut ob_spacing: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_ob_spacing,
                                                                           self.max_ob_spacing))

    def dump(self, dump_dict):
        return dump_dict

class EnvParamsSelector(object):
    def __init__(self, env_babbling, nb_test_episodes, train_env_kwargs):
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
                and (train_env_kwargs['stump_width'] is not None):
            mins = np.array([self.min_stump_height, self.min_stump_width, self.min_ob_spacing])
            maxs = np.array([self.max_stump_height, self.max_stump_width, self.max_ob_spacing])
        elif (train_env_kwargs['stump_height'] is not None) and (train_env_kwargs['obstacle_spacing'] is not None):
            mins = np.array([self.min_stump_height, self.min_ob_spacing])
            maxs = np.array([self.max_stump_height, self.max_ob_spacing])
        elif train_env_kwargs['stump_height'] is not None:
            mins = np.array([self.min_stump_height] * 2)
            maxs = np.array([self.max_stump_height] * 2)
        elif train_env_kwargs['tunnel_height'] is not None:
            mins = np.array([self.min_tunnel_height] * 2)
            maxs = np.array([self.max_tunnel_height] * 2)
        else:
            print('Unknown parameters')
            raise NotImplementedError

        # setup goals generator
        if env_babbling == 'oracle' or env_babbling == 'random':
            self.goal_generator = BaselineGoalGenerator(env_babbling, self.train_env_kwargs)
        elif env_babbling == 'sagg_iac':
            self.goal_generator = SAGG_IAC(mins, maxs)
        elif env_babbling == 'gmm':
            self.goal_generator = InterestGMM(mins, maxs)
        else:
            print('Unknown env babbling')
            raise NotImplementedError



        #data recording
        self.env_params_train = {'stump_hs':[], 'stump_ws':[], 'tunnel_hs':[], 'ob_sps':[]}
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []

        self.env_params_test = {'stump_hs':[], 'stump_ws':[], 'tunnel_hs':[], 'ob_sps':[]}
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
            and (all_env_params['ob_sps'][-1] is not None):
            params = [all_env_params['stump_hs'][-1][0], all_env_params['stump_ws'][-1][0], all_env_params['ob_sps'][-1]]
        return np.array(params)

    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if (self.env_babbling == 'sagg_iac') or (self.env_babbling == 'gmm'):
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
        if (self.env_babbling == 'sagg_iac') or (self.env_babbling == 'gmm'):
            algo_params = copy.copy(params)
            params = {'tunnel_height':None, 'stump_height':None, 'stump_width':None, 'obstacle_spacing':None}
            if (kwargs['stump_height'] is not None) and (kwargs['tunnel_height'] is not None):
                params['stump_height'] = [algo_params[0], 0.3]
                params['tunnel_height'] = [algo_params[1], 0.3]
            elif (kwargs['stump_height'] is not None) and (kwargs['obstacle_spacing'] is not None) and (kwargs['stump_width'] is not None):
                params['stump_height'] = [algo_params[0], 0.1]
                params['stump_width'] = [algo_params[1], 0.1]
                params['obstacle_spacing'] = algo_params[2]
            elif (kwargs['stump_height'] is not None) and (kwargs['obstacle_spacing'] is not None):
                params['stump_height'] = [algo_params[0], 0.1]
                params['obstacle_spacing'] = algo_params[1]
            elif kwargs['stump_height'] is not None:
                params['stump_height'] = algo_params
            elif kwargs['tunnel_height'] is not None:
                params['tunnel_height'] = algo_params
            else:
                raise NotImplementedError
        self.env_params_train['stump_hs'].append(params['stump_height'])
        self.env_params_train['stump_ws'].append(params['stump_width'])
        self.env_params_train['tunnel_hs'].append(params['tunnel_height'])
        self.env_params_train['ob_sps'].append(params['obstacle_spacing'])
        env.env.set_environment(roughness=kwargs['roughness'], stump_height=params['stump_height'],
                                stump_width=params['stump_width'], obstacle_spacing=params['obstacle_spacing'],
                                tunnel_height=params['tunnel_height'],
                                gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])
        return params

    def set_test_env_params(self, test_env, kwargs):
        self.test_ep_counter += 1
        #print("test_ep_nb:{}".format(self.test_ep_counter))

        epsilon = 1e-03
        random_stump_h = None
        random_tunnel_h = None
        random_stump_w = None
        random_ob_spacing = None
        test_mode = "levels"
        if test_mode == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = get_mu_sigma(kwargs['stump_height'][0], kwargs['stump_height'][1])
        elif test_mode == "levels":
            nb_levels = 3
            step = self.nb_test_episodes // nb_levels
            step_levels = np.arange(step, self.nb_test_episodes + step, step)
            current_level = -1
            for i in range(nb_levels):
                if self.test_ep_counter <= step_levels[i]:
                    current_level = i
                    break

            # if (kwargs['stump_height'] is not None) and (kwargs['tunnel_height'] is not None):
            #     pass
            if kwargs['stump_height'] is not None:
                max_stump_height = kwargs['stump_height'][1]
                stumph_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
                random_stump_h = get_mu_sigma(stumph_levels[current_level][0], stumph_levels[current_level][1])
                random_stump_h[1] = 0.1
            if kwargs['stump_width'] is not None:
                max_stump_width = kwargs['stump_width'][1]
                stumpw_levels = [[0.5, 1.0], [1.0, 1.5], [1.5, 2.]]
                random_stump_w = get_mu_sigma(stumpw_levels[current_level][0], stumpw_levels[current_level][1])
                random_stump_w[1] = 0.1
            if kwargs['tunnel_height'] is not None:
                max_tunnel_height = kwargs['tunnel_height'][1]
                assert(max_tunnel_height == 2)
                tunnel_levels = [[1.0, 1.3], [1.3, 1.6], [1.6, 1.9]]
                tunnel_levels.reverse() #shorter is harder
                random_tunnel_h = get_mu_sigma(tunnel_levels[current_level][0], tunnel_levels[current_level][1])
                random_tunnel_h[1] = 0.1
            if kwargs['obstacle_spacing'] is not None:
                spacing_levels = [[5, 8], [2, 5], [0, 2]]
                random_ob_spacing = get_mu_sigma(spacing_levels[current_level][0], spacing_levels[current_level][1])[0]

            if (kwargs['tunnel_height'] is not None) and (kwargs['obstacle_spacing'] is not None):
                # reduced std when both
                random_stump_h[1] = 0.1


        self.env_params_test['stump_hs'].append(random_stump_h)
        self.env_params_test['stump_ws'].append(random_stump_w)
        self.env_params_test['tunnel_hs'].append(random_tunnel_h)
        self.env_params_test['ob_sps'].append(random_ob_spacing)
        test_env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h, stump_width=random_stump_w,
                                     tunnel_height=random_tunnel_h, obstacle_spacing=random_ob_spacing,
                                     gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                     step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0