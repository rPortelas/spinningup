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



class BaranesGMM():
    def __init__(self, mins, maxs, n_components=None, seed=None, random_goal_ratio=0.2):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(np.random.randint(42,424242))
        self.mins = mins
        self.maxs = maxs
        self.random_goal_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)
        self.goals = []
        self.goals_times_comps = []
        self.fit_rate = 250
        self.nb_random = 250
        #self.window = 250
        self.random_goal_ratio = random_goal_ratio
        self.potential_ks = np.arange(1,11,1)
        self.all_times = np.arange(0, 1, 1/self.fit_rate)

        # boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'goals_lps': [], 'episodes': [],
              'comp_grids': [], 'comp_xs': [], 'comp_ys': []}

    def update(self, goal, competence,all_rewards=None):
        current_time = self.all_times[len(self.goals) % self.fit_rate]
        self.goals.append(goal)
        self.goals_times_comps.append(np.array(goal.tolist() + [current_time] + [competence]))

        #re-fit
        if len(self.goals) >= self.nb_random:
            if (len(self.goals) % self.fit_rate) == 0:
                #print(np.array(self.goals_lps).shape)
                #print(np.array(self.goals_lps))
                cur_goals_times_comps = np.array(self.goals_times_comps[-self.fit_rate:])
                potential_gmms = [GMM(n_components=k, covariance_type='full', random_state=self.seed) for k in self.potential_ks]
                potential_gmms = [g.fit(cur_goals_times_comps) for g in potential_gmms]#  fit all

                bics = [m.bic(cur_goals_times_comps) for m in potential_gmms]
                #plt.plot(self.potential_ks, bics, label='BIC')
                #plt.show()
                self.gmm = potential_gmms[np.argmin(bics)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['goals_lps'] = self.goals_times_comps
                self.bk['episodes'].append(len(self.goals))

    def sample_goal(self, kwargs=None, n_samples=1):
        #print(len(self.goals))
        new_goals = []
        if (len(self.goals) < self.nb_random) or (np.random.random() < self.random_goal_ratio):  # random goals until enough data is collected
            new_goals = [self.random_goal_generator.sample() for _ in range(n_samples)]
        else:
            self.times_comps_covars = []
            for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                self.times_comps_covars.append(max(0,covar[-2,-1]))  # heart of the swarm: TxC covariance as competence progress
            for _ in range(n_samples):
                # sample the gaussian according to its interest, defined as the absolute value of competence progress
                idx = proportional_choice(np.abs(self.times_comps_covars), eps=0.1)
                # sample goal in gaussian, without forgetting to remove learning progress dimension
                new_goal = np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-2]
                new_goals.append(np.clip(new_goal, self.mins, self.maxs))

        if n_samples == 1:
            return new_goals[0].tolist()
        else:
            return np.array(new_goals)

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict