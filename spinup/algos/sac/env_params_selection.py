import numpy as np
import pickle

class EnvParamsSelector(object):
    def __init__(self, env_babbling, nb_test_episodes):
        self.env_babbling = env_babbling
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03

        if env_babbling == 'oracle':
            self.min_stump_height = 0.0
            self.max_stump_height = 0.5
            self.mutation = 0.1
            self.mutation_rate = 50 #mutate each 50 episodes
            self.mutation_thr = 230 #reward threshold

        #data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_len = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if (len(self.env_train_rewards) % self.mutation_rate) == 0:
            mean_ret = np.mean(self.env_train_rewards[-50:])
            if mean_ret > self.mutation_thr:
                self.min_stump_height += self.mutation
                self.max_stump_height += self.mutation
            print('mut step: mean_ret:{} aft:({},{})'.format(mean_ret, self.min_stump_height, self.max_stump_height))


    def record_test_episode(self, reward, ep_len):
        self.env_test_rewards.append(reward)
        self.env_test_len.append(ep_len)

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump({'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_params_test': self.env_params_test,
                         'env_test_rewards': self.env_test_rewards,
                         'env_test_len': self.env_test_len}, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_random_stump_params(self, min_stump_height, max_stump_height):
        random_stump_height = np.random.uniform(min_stump_height, max_stump_height,2)
        random_stump_height.sort()
        if np.abs(random_stump_height[1] - random_stump_height[0]) < self.eps:
            random_stump_height[1] += self.eps
        return random_stump_height.tolist()

    def set_env_params(self, env, kwargs):
        params = []
        random_stump_h = None
        if self.env_babbling == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = self.get_random_stump_params(kwargs['stump_height'][0],kwargs['stump_height'][1])
                params += random_stump_h
                #print(random_stump_h)
        elif self.env_babbling == "oracle":
            if kwargs['stump_height'] is not None:
                random_stump_h = self.get_random_stump_params(self.min_stump_height,self.max_stump_height)
                params += random_stump_h
        env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])
        self.env_params_train.append(params)
        return params

    def set_test_env_params(self, test_env, kwargs):
        self.test_ep_counter += 1
        #print("test_ep_nb:{}".format(self.test_ep_counter))

        epsilon = 1e-03
        params = []
        test_mode = "levels"
        if self.env_babbling is not "none":
            if test_mode == "random":
                if kwargs['stump_height'] is not None:
                    random_stump_h = self.get_random_stump_params(kwargs['stump_height'][0], kwargs['stump_height'][1])
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
                    random_stump_h = self.get_random_stump_params(stump_levels[current_level][0], stump_levels[current_level][1])

            params += random_stump_h
            #print(random_stump_h)
            test_env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                         gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                         step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0
        self.env_params_test.append(params)
        return params