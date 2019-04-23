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


class DummyEnv(object):
    def __init__(self, nb_cells=10):
        self.nb_cells = nb_cells
        self.step_size = 1/nb_cells
        self.x_bnds = np.arange(0,1+self.step_size,self.step_size)
        self.y_bnds = np.arange(0, 1 + self.step_size, self.step_size)
        self.points = []
        self.cell_counts = np.zeros((len(self.x_bnds)-1, len(self.y_bnds)-1))
        self.cell_competence = self.cell_counts.copy()
        self.noise = 0.0
        self.max_per_cell = 100

    def episode(self, point):
        pts = point
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

def test_sagg_iac(env, nb_episodes, gif=True):
    goal_generator = SAGG_RIAC(np.array([0.0, 0.0]),
                                    np.array([1.0, 1.0]), temperature=20)
    all_boxes = []
    iterations = []
    interests = []
    rewards = []
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            print(env.cell_competence)
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

def test_interest_gmm(env, nb_episodes, gif=True):
    print("gmm run")
    goal_generator = InterestGMM([0, 0], [1, 1], n_components=4)
    interests = []
    rewards = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'goals_lps':[], 'episodes':[]}
    for i in range(nb_episodes):
        if (i % 500) == 0:
            print(env.cell_competence)
            if i>100:
                bk['weights'].append(goal_generator.gmm.weights_.copy())
                bk['covariances'].append(goal_generator.gmm.covariances_.copy())
                bk['means'].append(goal_generator.gmm.means_.copy())
                bk['goals_lps'].append(np.array(goal_generator.goals_lps.copy()))
                bk['episodes'].append(i)

                # plot_gmm(bk['weights'][-1],
                #          bk['means'][-1],
                #          bk['covariances'][-1],
                #          bk['goals_lps'][-1])
                # plt.show(block=False)
                # plt.pause(0.5)
                # plt.close()


        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update([np.array(goal)], [comp])
        rewards.append(comp)
    if gif:
        gmm_plot_gif(bk, gifname='gmm'+str(time.time()), gifdir='gifs/')

def test_CMAES(env, nb_episodes, gif=True):
    print("cmaes run")
    pop_s = 250
    goal_generator = InterestCMAES(2, popsize=pop_s, sigma_init=0.25)
    bk = {'covariances': [], 'means': [], 'goals': [], 'episodes': [], 'interests':[]}
    for i in range(nb_episodes):
        if (i % 500) == 0:
            print(env.cell_competence)

        #goal = np.clip(goal_generator.sample_goal(),0,1)
        goal = np.array(goal_generator.sample_goal())
        #print(goal)
        if goal_generator.counter == (goal_generator.popsize - 1):  # just generated new population, record stuff
            bk['covariances'].append(goal_generator.es.C.copy())
            bk['means'].append(goal_generator.es.mean.copy())
            bk['episodes'].append(i)
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


def test_random(env, nb_episodes):
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            print(env.cell_competence)
        p = np.random.random(2)
        env.episode(p)



if __name__=="__main__":
    nb_episodes = 30000
    env = DummyEnv()
    #test_sagg_iac(env, nb_episodes,gif=False)
    #test_interest_gmm(env, nb_episodes)
    test_CMAES(env, nb_episodes)
    # env = DummyEnv()
    #test_random(env, nb_episodes)

    #env.render()