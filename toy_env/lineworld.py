import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import scipy.stats as sp
import time
from teachers.sagg_iac import SAGG_IAC
from teachers.algos.riac import RIAC
from teachers.algos.florensa_riac import Florensa_RIAC
from teachers.algos.alp_gmm import ALPGMM
from teachers.algos.covar_gmm import CovarGMM
#from teachers.algos.cma_es import InterestCMAES
import pickle
import copy
import sys
from collections import OrderedDict
#import cProfile
import math
from teachers.algos.plot_utils import region_plot_gif, plot_gmm, gmm_plot_gif
from teachers.algos.riac import RIAC
from teachers.algos.covar_gmm import CovarGMM
from teachers.algos.egep import EGEP

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# class CosWorld(object):  # n-dimensional grid
#     def __init__(self, nb_funs=5, nb_dims=2, nb_steps=5, min_f=0, max_f=7, eps=3, render=True):
#         self.nb_funs = nb_funs
#         self.d = nb_dims
#         self.nb_steps = nb_steps
#         self.translations = np.random.uniform(-2,2,(self.nb_funs, self.d))
#         self.rotations = np.random.uniform(0,360,(self.nb_funs, 1))
#         self.min_f = min_f
#         self.max_f = max_f
#         self.range = self.max_f - self.min_f
#         self.steps = np.arange(self.min_f, self.max_f, self.range/self.nb_steps)
#         self.current_step_idx = np.ones((self.nb_funs,), dtype=np.int) * -1
#         self.eps = eps
#
#         self.do_render = render
#
#         self.current_state = np.zeros((self.nb_funs,self.d))
#         for i in range(nb_funs):
#             self.step_cosworld(i)
#
#         self.nn = NearestNeighbors(n_neighbors=self.nb_funs, algorithm='ball_tree')
#         # >> > X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#         # >> > nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
#         # >> > distances, indices = nbrs.kneighbors(X)
#
#         if self.do_render:
#             self.ax = plt.gca()
#             for j in range(nb_funs):
#                 d = []
#                 rot = self.rotations[j]
#                 transl = self.translations[j,:]
#                 X = self.steps
#                 Y = [np.cos(i) for i in X]
#                 for k,(x, y) in enumerate(zip(X, Y)):
#                     d_rot = rotate([0, 0], [x, y], rot)
#                     d_rot_offset = [v + offset for v, offset in zip(d_rot, transl)]
#                     d.append(d_rot_offset)
#                 d = np.array(d)
#                 self.ax.plot(d[:, 0], d[:, 1])
#             plt.ion()
#             plt.show()
#
#
#     def step_cosworld(self, f_idx):
#         if self.current_step_idx[f_idx] == self.nb_steps:
#             return
#
#         x = self.steps[self.current_step_idx[f_idx]]
#         rot_cos = rotate([0,0],[x,np.cos(x)], self.rotations[f_idx])
#         rot_translate_cos = [v + translate for v, translate in zip(rot_cos, self.translations[f_idx])]
#         self.current_state[f_idx] = rot_translate_cos
#         self.current_step_idx[f_idx] += 1
#
#     def get_score(self):
#         return np.mean(self.current_step_idx) / self.nb_steps
#
#     def render(self, point):
#         for i in range(self.nb_funs):
#             print(self.current_state[i])
#             self.ax.scatter(self.current_state[i][0], self.current_state[i][1], c='r', s=20, zorder=2)
#             self.ax.scatter(point[0,0], point[0,1], c='b', s=2, zorder=2)
#         plt.draw()
#         plt.pause(0.05)
#
#     def episode(self, point):
#         assert(len(point) == self.d)
#
#         # # fit nn
#         point = point.reshape(-1,self.d)
#         self.nn.fit(self.current_state)
#         distances, indices = self.nn.kneighbors(point)
#         reward = 0
#         # for dist, f_idx in zip(distances[0], indices[0]):
#         #     if dist < self.eps:
#         #         self.step_cosworld(f_idx)
#         #         reward += 1
#         #     else:
#         #         break  # no need to continue, distances is ordered
#         for i in range(self.nb_funs):
#             self.step_cosworld(i)
#
#         if self.do_render: self.render(np.zeros((1,2)))
#
#         return reward

class LineWorld(object):  # n-dimensional grid
    def __init__(self, nb_funs=10, nb_dims=2, nb_steps=5, min_f=0, max_f=7, eps=0.2, render=True):
        self.nb_funs = nb_funs
        self.d = nb_dims
        self.nb_steps = nb_steps

        # generate d-dimensional points defining all 'nb_funs' lines
        self.start_points = np.random.uniform(0, 1, (self.nb_funs, self.d))
        self.end_points = np.random.uniform(0, 1, (self.nb_funs, self.d))
        self.step_vectors = (self.end_points - self.start_points) / self.nb_steps
        self.step_counters = [0.] * self.nb_steps
        self.current_states = np.copy(self.start_points)
        self.eps = eps

        self.do_render = render


        # for i in range(nb_funs):
        #     self.step_cosworld(i)

        #self.nn = NearestNeighbors(n_neighbors=self.nb_funs, algorithm='ball_tree')
        # >> > X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        # >> > nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
        # >> > distances, indices = nbrs.kneighbors(X)

        if self.do_render:
            self.ax = plt.gca()
            for j in range(nb_funs):
                print('start {} stop {}'.format(self.start_points[j,:], self.end_points[j,:]))
                self.ax.plot([self.start_points[j,0], self.end_points[j,0]],
                             [self.start_points[j, 1], self.end_points[j, 1]])
            plt.ion()
            plt.show()

    def get_score(self):
        return np.sum(self.step_counters)

    def render(self, point):
        for i in range(self.nb_funs):
            print(self.current_states[i])
            self.ax.scatter(self.current_states[i][0], self.current_states[i][1], c='r', s=20, zorder=2)
            self.ax.scatter(point[0], point[1], c='b', s=2, zorder=2)
        plt.draw()
        plt.pause(0.5)

    def get_d_point_to_line(self,a,b,p):
        pa = p - a
        ba = b - a
        t = np.dot(pa, ba) / np.dot(ba, ba)
        return np.linalg.norm(pa - t * ba)

    # n_vector pa = P - A
    #     n_vector ba = B - A
    #     double t = dot(pa, ba)/dot(ba, ba)
    #        double d = length(pa - t * ba)


    def episode(self, point):
        assert(len(point) == self.d)

        # for i,(start_p, end_p, cur_p) in enumerate(zip(self.start_points, self.end_points, self.current_state)):
        #     assert(len(start_p) == 2)
        #     if self.step_counters[i] < self.nb_steps:
        #         self.current_state[i] += self.step_vectors[i]
        #         self.step_counters[i] += 1
        #     else:
        #         print('end {} cur {}'.format(end_p, cur_p))
        #
        # compute distance to each line
        reward = 0
        for i,(start_p, end_p, cur_p) in enumerate(zip(self.start_points, self.end_points, self.current_states)):
            #d_point_to_line = np.linalg.norm(np.cross(end_p-start_p, start_p-point))/np.linalg.norm(end_p-start_p)
            d_point_to_line = self.get_d_point_to_line(start_p, end_p, point)
            #print('regular: {}, my way: {}'.format(d_point_to_line, d_point_to_line2))

            d_cur_to_start = np.linalg.norm(start_p - cur_p)
            d_point_to_start = np.linalg.norm(start_p - point)
            #print('point: {}, fun {}, start: {} , dist_to_line: {}'.format(point, i, start_p, d_point_to_line))
            if d_point_to_line < self.eps:  # close enough to line, reward is distance to origin if before or near cur point
                d_point_to_cur = np.linalg.norm(point - cur_p)
                if d_point_to_cur < self.eps: # close to current state, reward + update current state !
                    if self.step_counters[i] < self.nb_steps:
                        self.current_states[i] += self.step_vectors[i]
                        self.step_counters[i] += 1
                    reward += 0.1 + d_point_to_start
                    #reward += self.step_counters[i]
                elif d_point_to_start < d_cur_to_start: # below current point, reward only
                    reward += 0.1 + d_point_to_start
                    #reward += self.step_counters[i]
                else:  # d_point_to_start >= d_cur_to_start, area not unlocked yet, no reward
                    pass

        if self.do_render: self.render(point)
        return reward

def test_random(env, nb_episodes, nb_dims, gif=False, score_step=1000, verbose=True, params={}):
    scores = []
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if verbose:
                print(scores[-1])
        p = np.random.uniform(0,1,nb_dims)
        env.episode(p)
    return scores

def test_riac(env, nb_episodes, gif=False, nb_dims=2, score_step=1000, verbose=True, params={}):
    if 'use_florensa' in params:
        print('florensuse')
        goal_generator = Florensa_RIAC(np.array([0.0] * nb_dims),
                              np.array([1.0] * nb_dims), params=params)
    else:
        goal_generator = RIAC(np.array([0.0] * nb_dims),
                              np.array([1.0]*nb_dims), params=params)
    all_boxes = []
    iterations = []
    interests = []
    rewards = []
    scores = []
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if verbose:
                print(scores[-1])
        goal = goal_generator.sample_goal(None)
        comp = env.episode(goal)
        split, _ = goal_generator.update(np.array(goal), comp, None)

        # book keeping
        if split and gif:
            boxes = goal_generator.regions_bounds
            interest = goal_generator.interest
            interests.append(copy.copy(interest))
            iterations.append(i)
            all_boxes.append(copy.copy(boxes))
        rewards.append(comp)

    if gif:
        region_plot_gif(all_boxes, interests, iterations, goal_generator.sampled_goals,
                        gifname='dummysaggriac', ep_len=[1]*nb_episodes, rewards=rewards, gifdir='gifs/')
    return scores

def test_interest_gmm(env, nb_episodes, gif=False, nb_dims=2, score_step=1000, verbose=True, params={}):
    goal_generator = ALPGMM([0] * nb_dims, [1] * nb_dims, params=params)
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'goals_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
    if nb_dims == 2:
        bk['current_states'] = []
        bk['start_points'] = env.start_points
        bk['end_points'] = env.end_points
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if verbose:
                print(scores[-1])
        if i>100 and (i % goal_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(goal_generator.gmm.weights_.copy())
            bk['covariances'].append(goal_generator.gmm.covariances_.copy())
            bk['means'].append(goal_generator.gmm.means_.copy())
            bk['goals_lps'] = goal_generator.goals_lps
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['current_states'].append(env.current_states.copy())


        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update([np.array(goal)], [comp])
        rewards.append(comp)
    if gif:
        gmm_plot_gif(bk, gifname='lineworld_gmm'+str(time.time()), gifdir='gifs/')
    return scores

def test_egep(env, nb_episodes, gif=False, nb_dims=2, score_step=1000, verbose=True, params={}):
    goal_generator = EGEP([0]*nb_dims, [1]*nb_dims, params=params)
    rewards = []
    scores = []
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if verbose:
                print(scores[-1])

        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update(np.array(goal), comp)
        rewards.append(comp)
    return scores

def test_covar_gmm(env, nb_episodes, gif=False, nb_dims=2, score_step=1000, verbose=True, params={}):
    goal_generator = CovarGMM([0]*nb_dims, [1]*nb_dims, params=params)
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'goals_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
    if nb_dims == 2:
        bk['current_states'] = []
        bk['start_points'] = env.start_points
        bk['end_points'] = env.end_points
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if verbose:
                print(scores[-1])
        if i>100 and (i % goal_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(goal_generator.gmm.weights_.copy())
            bk['covariances'].append(goal_generator.gmm.covariances_.copy())
            bk['means'].append(goal_generator.gmm.means_.copy())
            bk['goals_lps'] = goal_generator.goals_lps
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['current_states'].append(env.current_states.copy())


        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update(np.array(goal), comp)
        rewards.append(comp)
    if gif:
        gmm_plot_gif(bk, gifname='lineworld_gmm'+str(time.time()), gifdir='gifs/')
    return scores

def load_stats(id="test",fnum=0):
    from teachers.algos.plot_utils import region_plot_gif, plot_gmm, gmm_plot_gif
    per_model_colors = OrderedDict({})
    model_medians = OrderedDict({})
    try:
        scores, times, names, nb_episodes = pickle.load(open("lineworld_save_{}.pkl".format(id), "rb"))
    except FileNotFoundError:
        print('no data for lineworld_save_{}.pkl'.format(id))
        return 0
    plt.figure(fnum)
    ax = plt.gca()
    colors = ['red','blue','green','orange','purple']
    legend = True
    max_y = 0
    for i, algo_scores in enumerate(scores):
        if names[i] not in model_medians:
            model_medians[names[i]] = None
        ys = algo_scores
        model_medians[names[i]] = np.median(np.array(ys), axis=0)
        # print(median)
        episodes = np.arange(0,nb_episodes+1000,1000) / 1000
        #print(episodes)
        #print(model_medians)
        for k, y in enumerate(ys):
            if max(y) > max_y:
                max_y = max(y)
            # print("max:{} last:{}".format(max(y), y[-1]))
            if names[i] in per_model_colors:
                model_color = per_model_colors[names[i]]
            else:
                model_color = None
            ax.plot(episodes, y, color=model_color, linewidth=0.9, alpha=0.2)
    for algo_name, med in model_medians.items():
        ax.plot(episodes, med, color=model_color, linewidth=5, label=algo_name)
    ax.set_xlabel('Episodes (x1000)', fontsize=20)
    ax.set_ylabel('% Mastered cells', fontsize=20)
    ax.set_xlim(xmin=0, xmax=nb_episodes/1000)
    ax.set_ylim(ymin=0, ymax=max_y)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=15)
    if legend:
        leg = ax.legend(loc='bottom right', fontsize=14)

    if "long" in id:
        id = id[:-4]
    ax.set_title(id, fontsize=28)

    print("{}: Algo:{} times -> mu:{},sig{}".format(id, names[i], np.mean(times[i]), np.std(times[i])))
    plt.tight_layout()
    plt.savefig(id+'.png')

def run_stats(nb_episodes=100000, nb_steps=100, render=False, nb_dims=2, eps=0.01, nb_funs=5, algo_fs=None,
              nb_seeds=100, id="test", params={}, names=[], save=True):
    print("starting stats on {}".format(id))
    algo_results, algo_times = [[] for _ in range(len(algo_fs))], [[] for _ in range(len(algo_fs))]
    if len(names) == 0:
        names = [str(f).split(" ")[1] for f in algo_fs]
    for i in range(nb_seeds):
        print(i)
        for j in range(len(algo_fs)):
            env = LineWorld(nb_steps=nb_steps, render=False, nb_dims=nb_dims, eps=eps, nb_funs=nb_funs)
            start = time.time()
            algo_results[j].append(algo_fs[j](env, nb_episodes, nb_dims=nb_dims, gif=False, verbose=False, params=params[j]))
            end = time.time()
            algo_times[j].append(round(end-start))

    # Plot results
    for i, (scores, times) in enumerate(zip(algo_results, algo_times)):
        print("{}:{}: scores -> mu:{},sig{} | times -> mu:{},sig{}".format(id, names[i], np.mean(scores,axis=0)[-1],
                                                                           np.std(scores, axis=0)[-1],
                                                                           np.mean(times), np.std(times)))
    if save:
        data = [algo_results, algo_times, names, nb_episodes]
        pickle.dump(data, open("lineworld_save_{}.pkl".format(id), "wb"))

batch_exp = True
if batch_exp:
    nb_eps=100000
    nb_seeds = 30
    algos = (test_random, test_interest_gmm, test_interest_gmm, test_covar_gmm, test_riac)
    names = ["random",
             "alp-gmm",
             "alp-gmm_mlp",
             "covar_gmm",
             "riac"]
    params = [{},
              {"gmm_fitness_fun":'aic',"potential_ks":np.arange(2,11,1)},
              {"gmm_fitness_fun":'aic',"potential_ks":np.arange(2,11,1), "multiply_lp":True},
              {},
              {}]

    exp_args = [{"id":"2d001", "nb_episodes":nb_eps*2, "eps":0.01, "nb_dims":2, "algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
                {"id":"5d01", "nb_episodes":nb_eps*2, "eps":0.1, "nb_dims":5, "algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
                {"id":"10d035", "nb_episodes":nb_eps*3, "eps":0.35, "nb_dims":10,"algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
                {"id":"20d075", "nb_episodes":nb_eps*3, "eps":0.75, "nb_dims":20,"algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
                {"id":"50d191", "nb_episodes":nb_eps*3, "eps":1.91, "nb_dims":50,"algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
                {"id":"100d315", "nb_episodes":nb_eps*3, "eps":3.15, "nb_dims":100,"algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names}]

    if len(sys.argv) != 2:
        print('launching all experiences')
        exp_nbs = np.arange(0, len(exp_args))
    elif int(sys.argv[1]) >= len(exp_args):
        print(sys.argv[1] + ": not an expe")
        exit(0)
    else:
        exp_nbs = [int(sys.argv[1])]
        print("launching expe" + sys.argv[1] + " : " + exp_args[exp_nbs[0]]["id"])

    # for i in exp_nbs:
    #      run_stats(**exp_args[i])

    #Display all stats
    all_ids = []
    for i,exp in enumerate(exp_args):
        all_ids.append(exp["id"])
        load_stats(all_ids[-1], fnum=i)
    plt.show()
else:
    start = time.time()
    np.random.seed(43)
    nb_steps = 100
    nb_dims = 5
    nb_funs = 5
    eps = 0.1#0.75 #1.95
    env = LineWorld(nb_steps=nb_steps, render=False, nb_dims=nb_dims, eps=eps, nb_funs=nb_funs)
    #test_egep(env, 100000, nb_dims=nb_dims)
    test_riac(env, 100000, nb_dims=nb_dims)
    #test_covar_gmm(env, 100000, nb_dims=nb_dims, gif=False, params={"gmm_fitness_fun":'aic',"potential_ks":np.arange(2,11,1)})
    #test_random(env, 100000, nb_dims=nb_dims)
