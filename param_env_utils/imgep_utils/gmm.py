import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture as GMM
from sklearn.datasets import make_moons
import numpy as np
import copy
from gym.spaces import Box
from param_env_utils.imgep_utils.dataset import BufferedDataset
from param_env_utils.imgep_utils.gep_utils import proportional_choice
from param_env_utils.imgep_utils.dwgmm import DimensionallyWeightedGMM

class EmpiricalLearningProgress():
    def __init__(self, goal_size):
        self.interest_knn = BufferedDataset(1, goal_size, buffer_size=500, lateness=0)
        #self.window_size = 1000

    def get_lp(self, goal, competence):
        interest = 0
        if len(self.interest_knn) > 5:
            # compute learning progre   ss for new goal
            dist, idx = self.interest_knn.nn_y(goal)
            # closest_previous_goal = previous_goals[idx]
            closest_previous_goal = self.interest_knn.get_y(idx[0])
            closest_previous_goal_competence = self.interest_knn.get_x(idx[0])
            # print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)

            # compute Progress as absolute difference in competence
            progress = closest_previous_goal_competence - competence
            interest = np.abs(progress)

        # add to database
        self.interest_knn.add_xy(competence, goal)
        return interest

class InterestGMM():
    def __init__(self, mins, maxs, n_components=None, seed=None, params=dict(), random_goal_ratio=0.2,
                 gmm_fitness_fun='bic'):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)
        self.mins = mins
        self.maxs = maxs

        self.normalize_reward = False if "normalize reward" not in params else params["normalize reward"]
        if self.normalize_reward:
            self.max_reward = np.max(np.array(maxs) - np.array(mins)) # reward is scaled according to largest goal space

        self.potential_ks = np.arange(1,11,1) if "potential_ks" not in params else params["potential_ks"]
        self.warm_start = False if "warm_start" not in params else params["warm_start"]
        self.gmm_fitness_fun = "bic" if "gmm_fitness_fun" not in params else params["gmm_fitness_fun"]
        self.use_weighted_gmm = False if "weighted_gmm" not in params else True
        self.nb_em_init = 1 if "nb_em_init" not in params else params['nb_em_init']
        # print(self.warm_start)
        # print(self.gmm_fitness_fun)
        # print(self.potential_ks[0])
        # print(self.potential_ks[-1])
        # print(self.use_weighted_gmm)
        self.random_goal_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)
        self.lp_computer = EmpiricalLearningProgress(len(mins))
        self.goals = []
        self.lps = []
        self.goals_lps = []
        self.fit_rate = 250 if "fit_rate" not in params else params['fit_rate']
        self.nb_random = self.fit_rate
        self.random_goal_ratio = random_goal_ratio
        self.window = self.fit_rate

        # init GMMs
        self.potential_gmms = [self.init_gmm(k) for k in self.potential_ks]

        # boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'goals_lps': [], 'episodes': [],
              'comp_grids': [], 'comp_xs': [], 'comp_ys': []}

    def init_gmm(self, nb_gaussians):
        if self.use_weighted_gmm:
            return DimensionallyWeightedGMM(n_components=nb_gaussians, covariance_type='full', random_state=self.seed,
                   warm_start=self.warm_start)
        else:
            return GMM(n_components=nb_gaussians, covariance_type='full', random_state=self.seed,
                                            warm_start=self.warm_start, n_init=self.nb_em_init)


    def get_nb_gmm_params(self, gmm):
        # assumes full covariance
        # see https://stats.stackexchange.com/questions/229293/the-number-of-parameters-in-gaussian-mixture-model
        nb_gmms = gmm.get_params()['n_components']
        d = len(self.mins)
        params_per_gmm = (d*d - d)/2 + 2*d + 1
        return nb_gmms * params_per_gmm - 1


    def update(self, goals, competences,all_rewards=None):
        if not isinstance(competences, list):
            competences = [competences]
        if (not isinstance(goals[0], list)) and (not isinstance(goals[0], np.ndarray)):
            goals = [goals]
        for g, c in zip(goals, competences):
            self.goals.append(g)
            if self.normalize_reward:
                c = np.interp(c,[0,1],[0,self.max_reward])
            self.lps.append(self.lp_computer.get_lp(g, c))
            self.goals_lps.append(np.array(g.tolist()+[self.lps[-1]]))

        #re-fit
        if len(self.goals) >= self.nb_random:
            if (len(self.goals) % self.fit_rate) == 0:
                #print(np.array(self.goals_lps).shape)
                #print(np.array(self.goals_lps))
                cur_goals_lps = np.array(self.goals_lps[-self.window:])
                self.potential_gmms = [g.fit(cur_goals_lps) for g in self.potential_gmms]#  fit all

                if self.gmm_fitness_fun == 'bic':
                    fitnesses = [m.bic(cur_goals_lps) for m in self.potential_gmms]
                elif self.gmm_fitness_fun == 'aic':
                    fitnesses = [m.aic(cur_goals_lps) for m in self.potential_gmms]
                elif self.gmm_fitness_fun == 'aicc':
                    n = self.fit_rate
                    fitnesses = []
                    for l, m in enumerate(self.potential_gmms):
                        k = self.get_nb_gmm_params(m)
                        penalty = (2*k*(k+1)) / (n-k-1)
                        fitnesses.append(m.aic(cur_goals_lps) + penalty)
                else:
                    raise NotImplementedError
                    exit(1)
                #plt.plot(self.potential_ks, bics, label='BIC')
                #plt.show()
                self.gmm = self.potential_gmms[np.argmin(fitnesses)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['goals_lps'] = self.goals_lps
                self.bk['episodes'].append(len(self.goals))

    def sample_goal(self, kwargs=None, n_samples=1):
        new_goals = []
        if (len(self.goals) < self.nb_random) or (np.random.random() < self.random_goal_ratio):  # random goals until enough data is collected
            new_goals = [self.random_goal_generator.sample() for _ in range(n_samples)]
        else:
            self.lp_means = []
            self.lp_stds = []
            for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                self.lp_means.append(pos[-1])
                self.lp_stds.append(covar[-1,-1])
            for _ in range(n_samples):
                # sample gaussian
                idx = proportional_choice(self.lp_means, eps=0.0)
                # sample goal in gaussian, without forgetting to remove learning progress dimension
                new_goal = np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
                new_goals.append(np.clip(new_goal, self.mins, self.maxs))

        if n_samples == 1:
            return new_goals[0].tolist()
        else:
            return np.array(new_goals)

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict