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

class EmpiricalLearningProgress():
    def __init__(self, goal_size):
        self.interest_knn = BufferedDataset(1, goal_size, buffer_size=500, lateness=0)
        #self.window_size = 1000

    def get_lp(self, goal, competence):
        interest = 0
        if len(self.interest_knn) > 5:
            # compute learning progress for new goal
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
    def __init__(self, mins, maxs, n_components=None, seed=None):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(np.random.randint(42,424242))
        self.mins = mins
        self.maxs = maxs
        self.random_goal_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)
        self.lp_computer = EmpiricalLearningProgress(len(mins))
        self.goals = []
        self.lps = []
        self.goals_lps = []
        self.fit_rate = 250
        self.nb_random = 250
        self.random_goal_ratio = 0.1
        self.window = 250
        self.potential_ks = np.arange(1,11,1)

        # boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'goals_lps': [], 'episodes': [],
              'comp_grids': [], 'comp_xs': [], 'comp_ys': []}

    def update(self, goals, competences,all_rewards=None):
        if not isinstance(competences, list):
            competences = [competences]
        if (not isinstance(goals[0], list)) and (not isinstance(goals[0], np.ndarray)):
            goals = [goals]
        for g, c in zip(goals, competences):
            self.goals.append(g)
            self.lps.append(self.lp_computer.get_lp(g, c))
            self.goals_lps.append(np.array(g.tolist()+[self.lps[-1]]))

        #re-fit
        if len(self.goals) >= self.nb_random:
            if (len(self.goals) % self.fit_rate) == 0:
                #print(np.array(self.goals_lps).shape)
                #print(np.array(self.goals_lps))
                cur_goals_lps = np.array(self.goals_lps[-self.window:])
                potential_gmms = [GMM(n_components=k, covariance_type='full', random_state=self.seed) for k in self.potential_ks]
                potential_gmms = [g.fit(cur_goals_lps) for g in potential_gmms]#  fit all

                bics = [m.bic(cur_goals_lps) for m in potential_gmms]
                #plt.plot(self.potential_ks, bics, label='BIC')
                #plt.show()
                self.gmm = potential_gmms[np.argmin(bics)]

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
                idx = proportional_choice(self.lp_means, eps=0.1)
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








if __name__=="__main__":
    # Generate some data
    X, y_true = make_blobs(n_samples=400, centers=4,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting
    rng = np.random.RandomState(13)
    X_stretched = np.dot(X, rng.randn(2, 2))
    Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)

    gmm16 = InterestGMM([0,0],[1,1], n_components=16)
    #plot_gmm(gmm16, Xmoon, label=False)
    #plt.show(block=False)
    #plt.pause(1.0)
    gmm16.update(Xmoon, [0.]*len(Xmoon))
    Xnew = gmm16.sample_goal(n_samples=400)
    plt.scatter(Xnew[:, 0], Xnew[:, 1])
    plt.show(block=False)
    plt.pause(2.0)