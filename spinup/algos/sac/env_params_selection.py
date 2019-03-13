import numpy as np

class EnvParamsSelector(object):
    def __init__(self, env_babbling, nb_test_episodes):
        self.env_babbling = env_babbling
        self.nb_test_episodes = nb_test_episodes
        self.test_ep_counter = 0
        self.eps= 1e-03
        pass

    def get_random_stump_params(self, min_stump_height, max_stump_height):
        random_stump_height = np.random.uniform(min_stump_height, max_stump_height,2)
        random_stump_height.sort()
        if np.abs(random_stump_height[1] - random_stump_height[0]) < self.eps:
            random_stump_height[1] += self.eps
        return random_stump_height.tolist()

    def set_env_params(self, env, kwargs):
        params = []
        if self.env_babbling == "random":
            if kwargs['stump_height'] is not None:
                random_stump_h = self.get_random_stump_params(kwargs['stump_height'][0],kwargs['stump_height'][1])
                params += random_stump_h
                #print(random_stump_h)
            env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                    gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                    step_number=kwargs['step_number'], env_param_input=kwargs['env_param_input'])
        return params

    def set_test_env_params(self, test_env, kwargs):
        self.test_ep_counter += 1
        #print("test_ep_nb:{}".format(self.test_ep_counter))

        epsilon = 1e-03
        params = []
        test_mode = "levels"
        if self.env_babbling == "random":
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
        return params