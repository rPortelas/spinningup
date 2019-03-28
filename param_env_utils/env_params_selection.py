import numpy as np
import pickle
from param_env_utils.active_goal_sampling import SAGG_RIAC


def get_sorted2d_params(min, max, eps=1e-3):
    random_2dparams = np.random.uniform(min, max, 2)
    random_2dparams.sort()
    if np.abs(random_2dparams[1] - random_2dparams[0]) < eps:
        random_2dparams[1] += eps
    return random_2dparams.tolist()

class BaselineGoalGenerator(object):
    def __init__(self, env_babbling):
        self.env_babbling = env_babbling
        if env_babbling == 'oracle':
            self.min_stump_height = 0.0
            self.max_stump_height = 0.5
            self.mutation = 0.1
            self.mutation_rate = 50 #mutate each 50 episodes
            self.mutation_thr = 230 #reward threshold

    def sample_goal(self, kwargs):
        params = []
        random_stump_h = None
        if self.env_babbling == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = get_sorted2d_params(kwargs['stump_height'][0], kwargs['stump_height'][1])
                params += random_stump_h
                # print(random_stump_h)
        elif self.env_babbling == "oracle":
            if kwargs['stump_height'] is not None:
                random_stump_h = get_sorted2d_params(self.min_stump_height, self.max_stump_height)
                params += random_stump_h
        return params

    def update(self, goal, reward, env_train_rewards):
        if self.env_babbling == 'oracle':
            if (len(env_train_rewards) % self.mutation_rate) == 0:
                mean_ret = np.mean(env_train_rewards[-50:])
                if mean_ret > self.mutation_thr:
                    self.min_stump_height += self.mutation
                    self.max_stump_height += self.mutation
                print('mut step: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_stump_height,
                                                                 self.max_stump_height))
    def dump(self, dump_dict):
        return dump_dict

class EnvParamsSelector(object):
    def __init__(self, env_babbling, nb_test_episodes, train_env_kwargs):
        self.env_babbling = env_babbling
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        self.train_env_kwargs = train_env_kwargs
        self.min_stump_height = train_env_kwargs['stump_height'][0]
        self.max_stump_height = train_env_kwargs['stump_height'][1]

        if env_babbling == 'oracle' or env_babbling == 'random':
            self.goal_generator = BaselineGoalGenerator(env_babbling)

        elif env_babbling == 'sagg_iac':
            self.goal_generator = SAGG_RIAC(np.array([self.min_stump_height]*2),
                                            np.array([self.max_stump_height]*2))


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
        train_rewards = self.env_train_rewards
        if self.env_babbling == 'sagg_iac':
            reward = np.interp(reward, (-200, 300), (0, 1))
            self.env_train_norm_rewards.append(reward)
            train_rewards = self.env_train_norm_rewards[-1]
        self.goal_generator.update(np.array(self.env_params_train[-1]),
                                   reward, train_rewards)


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
        self.env_params_train.append(params)
        if self.env_babbling == 'sagg_iac':
            params.sort()
            if np.abs(params[0] - params[1]) < 1e-3: # TODO WILL NOT WORK WHEN CHANGING ENV
                params[1] += 1e-3
        env.env.set_environment(roughness=kwargs['roughness'], stump_height=params, #WILL NOT WORK WHEN CHANGING ENV
                                gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])
        return params

    def set_test_env_params(self, test_env, kwargs):
        self.test_ep_counter += 1
        #print("test_ep_nb:{}".format(self.test_ep_counter))

        epsilon = 1e-03
        params = []
        test_mode = "levels"
        if test_mode == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = get_sorted2d_params(kwargs['stump_height'][0], kwargs['stump_height'][1])
        elif test_mode == "levels":
            nb_levels = 3
            step = self.nb_test_episodes // nb_levels
            step_levels = np.arange(step, self.nb_test_episodes + step, step)
            current_level = -1
            for i in range(nb_levels):
                if self.test_ep_counter <= step_levels[i]:
                    current_level = i
                    break

            if kwargs['stump_height'] is not None:
                max_stump_height = kwargs['stump_height'][1]
                assert (max_stump_height == 2)
                stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
                random_stump_h = get_sorted2d_params(stump_levels[current_level][0], stump_levels[current_level][1])

        params += random_stump_h
        #print(random_stump_h)
        test_env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                     gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                     step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0
        self.env_params_test.append(params)
        return params