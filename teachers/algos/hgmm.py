import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import copy
from gym.spaces import Box
from teachers.utils.dataset import BufferedDataset
from teachers.algos.gep_utils import proportional_choice
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema

def get_most_prob_value(values, likelihoods):
    return values[np.argmax(likelihoods)]

def get_groups(data, plot=True):
    bandwidth = 1.06*np.std(data)*(len(data)**(-1/5))  # Rosenblats rule of thumb
    s = np.linspace(0,np.max(data),100)
    kde = KernelDensity(kernel='gaussian', bandwidth=max(bandwidth,1e-10)).fit(data)
    e = kde.score_samples(s.reshape(-1,1))
    e = np.exp(e)
    mins, maxs = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    groups = []
    groups_idxs = []
    groups_maxs = []  # maximum likelihood point
    if len(mins) == 1:
        groups.append(data[data < s[mins[0]]])
        groups_idxs.append(np.where(data < s[mins[0]])[0])
        groups_maxs.append(get_most_prob_value(s[:mins[0] + 1], e[:mins[0] + 1]))

        groups.append(data[data >= s[mins[0]]])
        groups_idxs.append(np.where(data >= s[mins[0]])[0])
        groups_maxs.append(get_most_prob_value(s[mins[0]:], e[mins[0]:]))
    elif len(mins) == 0:
        groups = [data]
        groups_idxs.append(np.arange(0,len(data),1))
        groups_maxs.append(get_most_prob_value(s,e))
    else:
        for i in range(len(mins)):
            min_lp = s[mins[i]]
            if i == 0: # first one
                groups.append(data[data < min_lp])
                groups_idxs.append(np.where(data < min_lp)[0])
                groups_maxs.append(get_most_prob_value(s[:mins[0] + 1], e[:mins[0] + 1]))

                next_mi = s[mins[i + 1]]
                groups.append(data[(data >= min_lp) * (data < next_mi)])
                groups_idxs.append(np.where((data >= min_lp) * (data < next_mi))[0])
                groups_maxs.append(get_most_prob_value(s[mins[i]:mins[i + 1] + 1], e[mins[i]:mins[i + 1] + 1]))

            elif i == len(mins)-1:  # last one
                groups.append(data[data >= min_lp])
                groups_idxs.append(np.where(data >= min_lp)[0])
                groups_maxs.append(get_most_prob_value(s[mins[i]:], e[mins[i]:]))
            else:
                next_mi = s[mins[i+1]]
                groups.append(data[(data >= min_lp) * (data < next_mi)])
                groups_idxs.append(np.where((data >= min_lp) * (data < next_mi))[0])
                groups_maxs.append(get_most_prob_value(s[mins[i]:mins[i + 1] + 1], e[mins[i]:mins[i + 1] + 1]))


    if plot:
        plt.plot(s, e)
        print(groups_maxs)
        print([len(g) for g in groups])
        print([g[-5:] for g in groups])
        plt.plot(
             s[maxs], e[maxs], 'go',
             s[mins], e[mins], 'ro')
        for i in range(len(mins)):
            if i == 0:  # first one
                plt.plot(s[:mins[i] + 1], e[:mins[i] + 1])
            elif i == len(mins) - 1:  # last one
                plt.plot(s[mins[i]:], e[mins[i]:])
            else:
                plt.plot(s[mins[i]:mins[i+1] + 1], e[mins[i]:mins[i+1] + 1])
        for i,d in enumerate(data):
            plt.plot(d,0.01,'bo', markersize=10)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

    for k,(idx_group, group)in enumerate(zip(groups_idxs, groups)):
        for idx, d in zip(idx_group, group):
            if data[idx] != d:
                print(data[idx])
                print(d)
                print('sorcery')
            assert data[idx] == d
    print('ALP space : bins in [0 , {}], bandwidth={}, clusters={}'.format(np.max(data), bandwidth, [len(c) for c in groups]))
    return groups_idxs, groups, groups_maxs

class MultivariateGaussian():
    def __init__(self, mus, stds):
        self.mus = mus
        self.stds = stds

        # book-keeping stuff:
        self.weights_ = ['single_gaussian']
        self.means_ = [mus]
        self.covariances_ = [np.diag(stds)]



    def sample(self):
        return ([np.random.normal(self.mus, self.stds)], None)

class EmpiricalLearningProgress():
    def __init__(self, goal_size):
        self.interest_knn = BufferedDataset(1, goal_size, buffer_size=2000, lateness=0)
        #self.window_size = 1000

    def get_lp(self, goal, competence):
        interest = 0
        if len(self.interest_knn) > 5:
            # compute learning progre   ss for new goal
            dist, idx = self.interest_knn.nn_y(goal)
            # closest_previous_goal = previous_tasks[idx]
            closest_previous_goal = self.interest_knn.get_y(idx[0])
            closest_previous_goal_competence = self.interest_knn.get_x(idx[0])
            # print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)

            # compute Progress as absolute difference in competence
            progress = closest_previous_goal_competence - competence
            interest = np.abs(progress)

        # add to database
        self.interest_knn.add_xy(competence, goal)
        return interest

class HGMM():
    def __init__(self, mins, maxs, n_components=None, seed=None, params=dict(), random_task_ratio=0.2,
                 gmm_fitness_fun='bic'):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)
        self.mins = mins
        self.maxs = maxs

        self.k_max = 5
        self.warm_start = False if "warm_start" not in params else params["warm_start"]
        self.gmm_fitness_fun = "bic" if "gmm_fitness_fun" not in params else params["gmm_fitness_fun"]
        self.use_weighted_gmm = False if "weighted_gmm" not in params else True
        self.nb_em_init = 1 if "nb_em_init" not in params else params['nb_em_init']
        self.multiply_lp = False if "multiply_lp" not in params else params['multiply_lp']

        self.random_task_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)
        self.lp_computer = EmpiricalLearningProgress(len(mins))
        self.mutation_noise = 1 / 20 * (np.array(self.maxs) - np.array(self.mins))
        self.tasks = []
        self.lps = []
        self.fit_rate = 250 if "fit_rate" not in params else params['fit_rate']
        self.nb_random = self.fit_rate
        self.random_task_ratio = random_task_ratio
        self.window = self.fit_rate

        self.previous_task = -1

        # boring book-keeping
        self.bk = {'clusters_weights': [], 'clusters_covariances': [], 'clusters_means': [], 'clusters_lps': [],
                   'clusters_tasks':[], 'tasks': [], 'episodes': [], 'lps':[]}

    def init_gmm(self, nb_gaussians):
        return GMM(n_components=nb_gaussians, covariance_type='full', warm_start=self.warm_start, n_init=self.nb_em_init)

    def update(self, task, competence,all_rewards=None):
        self.tasks.append(task)
        self.lps.append(self.lp_computer.get_lp(task, competence))

        # fitting time
        if len(self.tasks) >= self.nb_random:
            if (len(self.tasks) % self.fit_rate) == 0:

                # STEP 1 - Classification of tasks solely based on their ALP value (using 1-D Kernel Density)
                cur_lps = self.lps[-self.window:]
                cur_tasks = self.tasks[-self.window:]
                clusters_idx, clusters, self.clusters_alp = get_groups(np.array(cur_lps).reshape(-1,1), plot=False)
                clusters_tasks = []
                for cluster in clusters_idx:
                    clusters_tasks.append([])
                    for task_idx in cluster:
                        clusters_tasks[-1].append(cur_tasks[task_idx])

                # STEP 2 - Fit GMMs on the task dimensions of each group
                self.clusters_gmms = []
                for tasks in clusters_tasks:
                    tasks = np.array(tasks)
                    nb_tasks = len(tasks)
                    if nb_tasks == 1:  # single gaussian since only one task in group
                        self.clusters_gmms.append(MultivariateGaussian(tasks[0], self.mutation_noise))
                    else:
                        potential_ks = np.arange(1,min(self.k_max+1, nb_tasks),1)
                        potential_gmms = [self.init_gmm(k) for k in potential_ks]
                        potential_gmms = [g.fit(tasks) for g in potential_gmms]#  fit all
                        fitnesses = []
                        if self.gmm_fitness_fun == 'bic':
                            fitnesses = [m.bic(tasks) for m in potential_gmms]
                        elif self.gmm_fitness_fun == 'aic':
                            fitnesses = [m.aic(tasks) for m in potential_gmms]
                        elif self.gmm_fitness_fun == 'aicc':
                            n = self.fit_rate
                            fitnesses = []
                            for l, m in enumerate(potential_gmms):
                                k = self.get_nb_gmm_params(m)
                                penalty = (2*k*(k+1)) / (n-k-1)
                                fitnesses.append(m.aic(cur_tasks_lps) + penalty)
                        else:
                            raise NotImplementedError
                            exit(1)
                        # plt.plot(self.potential_ks, fitnesses, label='AIC')
                        # plt.show(block=False)
                        # plt.pause(0.5)
                        self.clusters_gmms.append(potential_gmms[np.argmin(fitnesses)])

                # book-keeping
                self.bk['clusters_weights'].append([gmm.weights_.copy() for gmm in self.clusters_gmms])
                self.bk['clusters_covariances'].append([gmm.covariances_.copy() for gmm in self.clusters_gmms])
                self.bk['clusters_means'].append([gmm.means_.copy() for gmm in self.clusters_gmms])
                self.bk['clusters_lps'].append(self.clusters_alp)
                self.bk['clusters_tasks'].append([c_task for c_task in clusters])
                self.bk['tasks'] = self.tasks
                self.bk['episodes'].append(len(self.tasks))

    def sample_goal(self, kwargs=None):
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):  # random tasks until enough data is collected
            new_task = self.random_task_generator.sample()
        else:
            # STEP 1- Choose a tasks cluster proportionally to its associated Absolute Learning Progress
            idx = proportional_choice(self.clusters_alp, eps=0.0)

            # STEP 2- Generate new tasks from clusters' GMM (regular GMM way, or ALP-GMM way)
            new_task = self.clusters_gmms[idx].sample()[0][0]
            new_task = np.clip(new_task, self.mins, self.maxs)
            if len(new_task) != 2:
                print('sorcery')

        assert(len(new_task) == 2)
        if np.array_equal(self.previous_task, np.array(new_task)):
            print('whaaauiout')

        self.previous_task = np.array(new_task)
        return np.array(new_task)

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict


# #a = np.array([0,0,12,13,15,51,53,80,80]).reshape(-1, 1)
# a = np.zeros((250,1))

a = np.array([[0.  ],
       [0.01],
       [0.01],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.01],
       [0.  ],
       [0.  ],
       [0.06],
       [0.04],
       [0.01],
       [0.01],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.04],
       [0.  ],
       [0.  ],
       [0.01],
       [0.  ],
       [0.01],
       [0.08],
       [0.01],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.09],
       [0.  ],
       [0.  ],
       [0.01],
       [0.  ],
       [0.  ],
       [0.05],
       [0.  ],
       [0.01],
       [0.15],
       [0.02],
       [0.  ],
       [0.17],
       [0.12],
       [0.19],
       [0.02],
       [0.  ],
       [0.01],
       [0.12],
       [0.13],
       [0.19],
       [0.  ],
       [0.11],
       [0.16],
       [0.17],
       [0.14],
       [0.  ],
       [0.01],
       [0.2 ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.14],
       [0.  ],
       [0.01],
       [0.01],
       [0.  ],
       [0.29],
       [0.  ],
       [0.11],
       [0.12],
       [0.12],
       [0.  ],
       [0.  ],
       [0.38],
       [0.  ],
       [0.01],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.4 ],
       [0.12],
       [0.  ],
       [0.  ],
       [0.02],
       [0.02],
       [0.  ],
       [0.  ],
       [0.34],
       [0.  ],
       [0.  ],
       [0.25],
       [0.13],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.32],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.12],
       [0.  ],
       [0.  ],
       [0.33],
       [0.41],
       [0.  ],
       [0.  ],
       [0.34],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.13],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.49],
       [0.39],
       [0.12],
       [0.  ],
       [0.  ],
       [0.02],
       [0.24],
       [0.  ],
       [0.  ],
       [0.37],
       [0.  ],
       [0.  ],
       [0.31],
       [0.  ],
       [0.16],
       [0.56],
       [0.  ],
       [0.49],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.49],
       [0.  ],
       [0.47],
       [0.66],
       [0.  ],
       [0.  ],
       [0.07],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.55],
       [0.  ],
       [0.  ],
       [0.28],
       [0.18],
       [0.02],
       [0.  ],
       [0.38],
       [0.19],
       [0.01],
       [0.02],
       [0.01],
       [0.34],
       [0.  ],
       [0.03],
       [0.03],
       [0.  ],
       [0.08],
       [0.05],
       [0.  ],
       [0.  ],
       [0.06],
       [0.39],
       [0.07],
       [0.37],
       [0.51],
       [0.  ],
       [0.  ],
       [0.59],
       [0.53],
       [0.08],
       [0.09],
       [0.  ],
       [0.02],
       [0.42],
       [0.07],
       [0.08],
       [0.43],
       [0.3 ],
       [0.  ],
       [0.11],
       [0.1 ],
       [0.12],
       [0.13],
       [0.48],
       [0.84],
       [0.14],
       [0.  ],
       [0.  ],
       [0.  ],
       [0.52],
       [0.  ],
       [0.15],
       [0.43],
       [0.33],
       [0.16],
       [0.17],
       [0.23],
       [0.09],
       [0.1 ],
       [0.69],
       [0.18],
       [0.  ],
       [0.01],
       [0.19],
       [0.  ],
       [0.  ],
       [0.22],
       [0.01],
       [0.21],
       [0.22],
       [0.75],
       [0.2 ],
       [0.  ],
       [0.06],
       [0.21],
       [0.06],
       [0.  ],
       [0.21],
       [0.22],
       [0.23],
       [0.24],
       [0.  ],
       [0.01],
       [0.95],
       [0.02],
       [0.58],
       [0.01],
       [0.31],
       [0.14],
       [0.26],
       [0.27],
       [0.28],
       [0.  ],
       [0.29],
       [0.42],
       [0.3 ]])
# groups_idx, groups, groups_max = get_groups(a, plot=True)
# print(groups_idx)
# print(groups)
# print(groups_max)