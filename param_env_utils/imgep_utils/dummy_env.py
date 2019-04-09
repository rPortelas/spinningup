import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import time
from param_env_utils.active_goal_sampling import SAGG_RIAC
from param_env_utils.imgep_utils.plot_utils import region_plot_gif
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
        self.noise = 0.1
        self.max_per_cell = 50

    def episode(self, point):
        self.points.append(point)
        # find in which cell point falls and add to total cell counts
        cells = sp.binned_statistic_2d([point[0]], [point[1]], None, 'count',
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
            normalized_competence = np.clip(normalized_competence + np.random.normal(0,self.noise),0,1)
        return normalized_competence
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


        # compute competence

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


def test_random(env, nb_episodes):
    for i in range(nb_episodes):
        if (i % 1000) == 0:
            print(env.cell_competence)
        p = np.random.random(2)
        env.episode(p)



if __name__=="__main__":
    nb_episodes = 10000
    env = DummyEnv()
    test_sagg_iac(env, nb_episodes)
    # env = DummyEnv()
    #test_random(env, nb_episodes)

    #env.render()