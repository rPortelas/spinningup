import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import time
from param_env_utils.active_goal_sampling import SAGG_RIAC
from param_env_utils.imgep_utils.gmm import InterestGMM
from param_env_utils.imgep_utils.cma_es import InterestCMAES
from param_env_utils.imgep_utils.plot_utils import cmaes_plot_gif,region_plot_gif, plot_gmm,\
    gmm_plot_gif, plot_cmaes
import copy
import cProfile

class NDummyEnv(object):  # n-dimensional grid
    def __init__(self, nb_cells=5, ndims=2):
        self.nb_cells = nb_cells
        self.step_size = 1/nb_cells
        self.bnds = [np.arange(0,1+self.step_size,self.step_size) for i in range(ndims)]
        self.points = []
        self.cell_counts = np.zeros((nb_cells, ) * ndims)
        self.cell_competence = self.cell_counts.copy()
        self.noise = 0.0
        self.max_per_cell = 100
        self.nb_random_dims = 0
        self.ndims = ndims + self.nb_random_dims

    def get_score(self):
        score = np.where(self.cell_competence > (3*(self.max_per_cell/4)))
        return len(score[0])

    def episode(self, point):
        assert(len(point) == self.ndims)
        for v in point:
            if (v < 0.0) or (v > 1.0):
                print("OUT OF BOUNDS")
                self.points.append(point)
                return 0.
        pts = point
        self.points.append(pts)
        # find in which cell pts falls and add to total cell counts
        arr_pts = np.array([pts])
        cells = sp.binned_statistic_dd(arr_pts, np.ones(arr_pts.shape), 'count',
                                       bins=self.bnds).statistic
        self.cell_counts += cells[0]
        cell_idx = tuple([v[0] for v in cells[0].nonzero()])
        competence_added = False
        if all(v == 0 for v in cell_idx):  # if root cell, no need to have previous cells with high competence to learn
            self.cell_competence[cell_idx] = min(self.cell_competence[cell_idx] + 1, self.max_per_cell)
            competence_added = True

        # find index of "previous" adjacent cells
        prev_cell_idx = [[idx, max(0, idx - 1)] for idx in cell_idx]
        previous_neighbors_idx = np.array(np.meshgrid(*prev_cell_idx)).T.reshape(-1,len(prev_cell_idx))
        for pn_idx in previous_neighbors_idx:
            prev_idx = tuple(pn_idx)
            #print(prev_idx)
            if all(v == cell_idx[i] for i,v in enumerate(prev_idx)):  # original cell, not previous neighbor
                continue
            else:
                if self.cell_competence[prev_idx] >= (3*(self.max_per_cell/4)):  # one previous neighbor with high comp
                    self.cell_competence[cell_idx] = min(self.cell_competence[cell_idx] + 1, self.max_per_cell)
                    competence_added = True
                    break

        normalized_competence = np.interp(self.cell_competence[cell_idx], (0, self.max_per_cell), (0, 1))
        if self.noise >= 0.0:
            normalized_competence = np.clip(normalized_competence + np.random.normal(0,self.noise), 0, 1)
        return normalized_competence


class DummyEnv(object):
    def __init__(self, nb_cells=10, nb_random_dims=0):
        self.nb_cells = nb_cells
        self.step_size = 1/nb_cells
        self.x_bnds = np.arange(0,1+self.step_size,self.step_size)
        self.y_bnds = np.arange(0, 1 + self.step_size, self.step_size)
        self.points = []
        self.cell_counts = np.zeros((len(self.x_bnds)-1, len(self.y_bnds)-1))
        self.cell_competence = self.cell_counts.copy()
        self.noise = 0.2
        self.max_per_cell = 100
        self.nb_random_dims = nb_random_dims
        self.ndims = 2 + self.nb_random_dims

    def get_score(self):
        score = np.where(self.cell_competence > (3*(self.max_per_cell/4)))
        return len(score[0])

    def episode(self, point):
        assert(len(point) == self.ndims)
        pts = point[0:2]
        if (pts[0] < 0.0) or (pts[1] < 0.0) or (pts[1] > 1.0) or (pts[0] > 1.0):
            print("OUT OF BOUNDS")
            self.points.append(pts)
            return 0.

        self.points.append(pts)
        # find in which cell pts falls and add to total cell counts
        cells = sp.binned_statistic_2d([pts[0]], [pts[1]], None, 'count',
                                bins=[self.x_bnds, self.y_bnds]).statistic
        self.cell_counts += cells
        cell_x, cell_y = cells.nonzero()
        # find index of "previous" adjacent cells
        prev_xs = [cell_x, max(0,cell_x - 1)]
        prev_ys = [cell_y, max(0,cell_y - 1)]
        if prev_xs[1] == 0 and prev_ys[0] == 0:
            # root cell
            self.cell_competence[cell_x, cell_y] = min(self.cell_competence[cell_x, cell_y] + 1, self.max_per_cell)
        else:
            competence_added = False
            for x in prev_xs:
                for y in prev_ys:
                    if x == cell_x and y == cell_y: # current cell
                        continue
                    else:
                        if self.cell_competence[x, y] >= (3*(self.max_per_cell/4)):
                            self.cell_competence[cell_x, cell_y] = min(self.cell_competence[cell_x, cell_y] + 1, self.max_per_cell)
                            competence_added = True
                            break
                if competence_added:
                    break

        normalized_competence = np.interp(self.cell_competence[cell_x, cell_y], (0, self.max_per_cell), (0, 1))
        if self.noise >= 0.0:
            normalized_competence = np.clip(normalized_competence + np.random.normal(0,self.noise), 0, 1)
        return normalized_competence[0]
        #print(self.cell_counts[0, 0])
        #print(self.cell_counts)
        #print(np.sum(self.cell_counts))

        # compute competence return


    def render(self):
        points = np.array(self.points)
        plt.plot(points[:,0], points[:,1], 'r.')
        plt.xlim(left=0, right=1)
        plt.ylim(bottom=0,top=1)
        plt.show()

def test_sagg_iac(env, nb_episodes, gif=True, nb_dims=2):
    goal_generator = SAGG_RIAC(np.array([0.0]*nb_dims),
                                    np.array([1.0]*nb_dims), temperature=20)
    all_boxes = []
    iterations = []
    interests = []
    rewards = []
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            if nb_dims == 2:
                print(env.cell_competence)
            else:
                print(env.get_score())
        goal = goal_generator.sample_goal(None)
        comp = env.episode(goal)
        split, _ = goal_generator.update(np.array(goal), comp, None)

        # book keeping
        if split:
            boxes = goal_generator.region_bounds
            interest = goal_generator.interest
            interests.append(copy.copy(interest))
            iterations.append(i)
            all_boxes.append(copy.copy(boxes))
        rewards.append(comp)

    if gif:
        region_plot_gif(all_boxes, interests, iterations, goal_generator.sampled_goals,
                        gifname='dummysagg', ep_len=[1]*nb_episodes, rewards=rewards, gifdir='gifs/')
    return env.get_score()

def test_interest_gmm(env, nb_episodes, gif=True, nb_dims=2):
    goal_generator = InterestGMM([0]*nb_dims, [1]*nb_dims)
    rewards = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'goals_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            if ndims == 2:
                print(env.cell_competence)
                print(env.get_score())
            else:
                print(env.get_score())
        if i>100 and (i % goal_generator.fit_rate) == 0:
            bk['weights'].append(goal_generator.gmm.weights_.copy())
            bk['covariances'].append(goal_generator.gmm.covariances_.copy())
            bk['means'].append(goal_generator.gmm.means_.copy())
            bk['goals_lps'].append(np.array(goal_generator.goals_lps.copy()))
            bk['episodes'].append(i)
            if ndims == 2:
                bk['comp_grids'].append(env.cell_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())

        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update([np.array(goal)], [comp])
        rewards.append(comp)
    if gif:
        gmm_plot_gif(bk, gifname='gmm'+str(time.time()), gifdir='gifs/')
    return env.get_score()

def test_CMAES(env, nb_episodes, gif=True):
    print("cmaes run")
    pop_s = 250
    goal_generator = InterestCMAES(2, popsize=pop_s, sigma_init=0.5)
    bk = {'covariances': [], 'means': [], 'goals': [], 'episodes': [], 'interests':[], 'sigmas':[]}
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            if ndims == 2:
                print(env.cell_competence)
            else:
                print(env.get_score())

        #goal = np.clip(goal_generator.sample_goal(),0,1)
        goal = np.array(goal_generator.sample_goal())
        #print(goal)
        if goal_generator.counter == (goal_generator.popsize - 1):  # just generated new population, record stuff
            bk['covariances'].append(goal_generator.es.C.copy())
            bk['means'].append(goal_generator.es.mean.copy())
            bk['episodes'].append(i)
            bk['sigmas'].append(goal_generator.es.sigma)
            # plot_cmaes(bk['means'][-1],
            #          bk['covariances'][-1],
            #          bk['interests'],
            #          np.array(bk['goals']))
            # plt.show(block=False)
            # plt.pause(0.5)
            # plt.close()

        comp = env.episode(goal)
        #print("goal{}".format(goal))
        goal_generator.update(np.array(goal), comp)
        bk['interests'].append(goal_generator.current_fitnesses[-1])
        bk['goals'].append(goal.copy())
    if gif:
        cmaes_plot_gif(bk, gifname='cmaes' + str(time.time()), gifdir='gifs/')
    return env.get_score()


def test_random(env, nb_episodes, ndims=2, gif=False):
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            if ndims == 2:
                print(env.cell_competence)
            else:
                print(env.get_score())
        p = np.random.random(ndims)
        env.episode(p)
    return env.get_score()



if __name__=="__main__":
    cp = cProfile.Profile()
    cp.enable()
    nb_episodes = 20000
    nb_rand_dims = 0
    ndims = 3
    env = NDummyEnv(ndims=ndims)
    test_sagg_iac(env, nb_episodes, nb_dims=ndims, gif=False)
    #test_interest_gmm(env, nb_episodes, nb_dims=ndims, gif=False)
    #test_CMAES(env, nb_episodes, gif=True)
    # env = DummyEnv()
    #score = test_random(env, nb_episodes, ndims=ndims)
    # print(score)
    # cp.disable()
    # cp.dump_stats("test.cprof")
    #env.render()
    #

    # # Statistical analysis
    # nb_episodes = 20000
    # nb_rand_dims = 0
    # algo_fs = (test_sagg_iac, test_interest_gmm)
    # algo_results, algo_times = [[] for _ in range(len(algo_fs))], [[] for _ in range(len(algo_fs))]
    # for i in range(100):
    #     for j in range(len(algo_fs)):
    #         env = DummyEnv(nb_random_dims=nb_rand_dims)
    #         start = time.time()
    #         print(algo_fs[j])
    #         algo_results[j].append(algo_fs[j](env, nb_episodes, nb_dims=2+nb_rand_dims, gif=False))
    #         end = time.time()
    #         algo_times[j].append(round(end-start))
    #
    # # Plot results
    # for i, (scores, times) in enumerate(zip(algo_results, algo_times)):
    #     print("Algo:{} \n"
    #           " scores -> mu:{},sig{},all{}\n"
    #           " times -> mu:{},sig{}".format(i, np.mean(scores), np.std(scores), scores,
    #                                          np.mean(times), np.std(times)))
    #
