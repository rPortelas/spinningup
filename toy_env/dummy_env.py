import numpy as np
import scipy.stats as sp
import time
#from teachers.active_goal_sampling import SAGG_IAC
from teachers.algos.riac import RIAC
from teachers.algos.florensa_riac import Florensa_RIAC
from teachers.algos.alp_gmm import InterestGMM
from teachers.algos.covar_gmm import CovarGMM
#from teachers.algos.egep import EGEP
#from teachers.algos.hgmm import HGMM
#from teachers.algos.cma_es import InterestCMAES
import pickle
import copy
import sys
from collections import OrderedDict
#import cProfile
#from teachers.algos.plot_utils import egep_plot_gif, hgmm_plot_gif

class NDummyEnv(object):  # n-dimensional grid
    def __init__(self, nb_cells=10, nb_dims=2, noise=0.0, nb_rand_dims=0, forget=False):
        self.nb_cells = nb_cells
        self.nb_total_cells = nb_cells ** nb_dims
        self.step_size = 1/nb_cells
        self.bnds = [np.arange(0,1+self.step_size,self.step_size) for i in range(nb_dims)]
        self.points = []
        self.cell_counts = np.zeros((nb_cells, ) * nb_dims)
        self.cell_competence = self.cell_counts.copy()
        self.forget = forget
        if self.forget:
            self.cell_forget = self.cell_counts.copy()
            self.forget_rate = 500
            self.forget_counter = 0
        self.noise = noise
        self.max_per_cell = 100
        self.nb_random_dims = nb_rand_dims
        self.nb_dims = nb_dims
        self.all_nb_dims = nb_dims + self.nb_random_dims


    def get_score(self):
        score = np.where(self.cell_competence > (3*(self.max_per_cell/4)))
        return (len(score[0]) / self.nb_total_cells)*100

    def episode(self, point):
        assert(len(point) == self.all_nb_dims)
        for v in point:
            if (v < 0.0) or (v > 1.0):
                #print("OUT OF BOUNDS")
                self.points.append(point)
                return 0.
        pts = point[0:self.nb_dims]  # discard random dimensions
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

        # handle forgetting is enabled
        if self.forget:
            self.cell_forget += cells[0]  # recently touched cells will have > 0 values
            self.forget_counter += 1
            if (self.forget_counter % self.forget_rate) == 0:  # time for a forget step
                self.cell_forget[self.cell_forget == 0] = -1  # tmp, just to remember which are forgotten cells
                self.cell_forget[self.cell_forget > 0] = 0  # no forget on recently visited cells
                self.cell_forget[self.cell_forget == -1] = 26  # forget 1 competence on recently unvisited cells
                # then substract only where there is already competence
                self.cell_competence = np.clip(self.cell_competence - self.cell_forget, 0, np.inf)
                # reset cell forget
                self.cell_forget = np.zeros((self.nb_cells, ) * self.nb_dims)



        normalized_competence = np.interp(self.cell_competence[cell_idx], (0, self.max_per_cell), (0, 1))
        # if self.noise >= 0.0:
        #     normalized_competence = np.clip(normalized_competence + np.random.normal(0,self.noise), 0, 1)
        return normalized_competence

def test_riac(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
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
            if nb_dims == 2:
                if verbose:
                    print(env.cell_competence)
            else:
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

def test_interest_gmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    goal_generator = InterestGMM([0]*nb_dims, [1]*nb_dims, params=params)
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'goals_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cell_competence)
            else:
                if verbose:
                    print(scores[-1])
        if i>100 and (i % goal_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(goal_generator.gmm.weights_.copy())
            bk['covariances'].append(goal_generator.gmm.covariances_.copy())
            bk['means'].append(goal_generator.gmm.means_.copy())
            bk['goals_lps'] = goal_generator.goals_lps
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['comp_grids'].append(env.cell_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())

        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update([np.array(goal)], [comp])
        rewards.append(comp)
    if gif:
        gmm_plot_gif(bk, gifname='gmm'+str(time.time()), gifdir='gifs/')
    return scores

def test_covar_gmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    goal_generator = CovarGMM([0] * nb_dims, [1] * nb_dims, params=params)
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'goals_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cell_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i,scores[-1]))
        if i>100 and (i % goal_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(goal_generator.gmm.weights_.copy())
            bk['covariances'].append(goal_generator.gmm.covariances_.copy())
            bk['means'].append(goal_generator.gmm.means_.copy())
            bk['goals_lps'] = goal_generator.goals_times_comps
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['comp_grids'].append(env.cell_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())

        goal = goal_generator.sample_goal()
        comp = env.episode(goal)
        goal_generator.update(np.array(goal), comp)
        rewards.append(comp)
    if gif:
        gmm_plot_gif(bk, gifname='baranesgmm'+str(time.time()), gifdir='gifs/')
    return scores

# def test_egep(env, nb_episodes, gif=False, nb_dims=2, score_step=1000, verbose=True, params={}):
#     goal_generator = EGEP([0]*nb_dims, [1]*nb_dims, params=params)
#     rewards = []
#     scores = []
#     bk = {"elites_idx":[], 'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
#     for i in range(nb_episodes+1):
#         if (i % score_step) == 0:
#             #print(i)
#             scores.append(env.get_score())
#             if verbose:
#                 print(scores[-1])
#             bk["elites_idx"].append(copy.copy(goal_generator.elite_tasks_idx))
#             if nb_dims == 2:
#                 bk['comp_grids'].append(env.cell_competence.copy())
#                 bk['comp_xs'].append(env.bnds[0].copy())
#                 bk['comp_ys'].append(env.bnds[1].copy())
#
#         goal = goal_generator.sample_goal()
#         comp = env.episode(goal)
#         goal_generator.update(np.array(goal), comp)
#         rewards.append(comp)
#     bk['tasks'] = goal_generator.tasks
#     bk['lps'] = goal_generator.tasks_lps
#     if gif:
#         egep_plot_gif(bk, gifname='egep'+str(time.time()), gifdir='gifs/')
#     return scores

# def test_hgmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
#     goal_generator = HGMM([0]*nb_dims, [1]*nb_dims, params=params)
#     rewards = []
#     scores = []
#     bk = {'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
#     for i in range(nb_episodes+1):
#         if (i % score_step) == 0:
#             scores.append(env.get_score())
#             if nb_dims == 2:
#                 if verbose:
#                     print(env.cell_competence)
#             else:
#                 if verbose:
#                     print(scores[-1])
#         if i>100 and (i % goal_generator.fit_rate) == 0 and (gif is True):
#             if nb_dims == 2:
#                 bk['comp_grids'].append(env.cell_competence.copy())
#                 bk['comp_xs'].append(env.bnds[0].copy())
#                 bk['comp_ys'].append(env.bnds[1].copy())
#
#         goal = goal_generator.sample_goal()
#         comp = env.episode(goal)
#         goal_generator.update(np.array(goal), comp)
#         rewards.append(comp)
#     if gif:
#         goal_generator.dump(bk)
#         hgmm_plot_gif(bk, gifname='Hgmm'+str(time.time()), gifdir='gifs/')
#     return scores

def test_random(env, nb_episodes, nb_dims=2, gif=False, score_step=1000, verbose=True, params={}):
    scores = []
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cell_competence)
            else:
                if verbose:
                    print(scores[-1])
        p = np.random.random(nb_dims)
        env.episode(p)
    return scores


def run_stats(nb_episodes=20000, nb_dims=2, nb_rand_dims = 0, algo_fs=None,
              nb_seeds=100, noise=0.0, nb_cells=10, id="test", params={}, names=[], save=True):
    print("starting stats on {}".format(id))
    algo_results, algo_times = [[] for _ in range(len(algo_fs))], [[] for _ in range(len(algo_fs))]
    if len(names) == 0:
        names = [str(f).split(" ")[1] for f in algo_fs]
    for i in range(nb_seeds):
        print(i)
        for j in range(len(algo_fs)):
            env = NDummyEnv(nb_dims=nb_dims, nb_rand_dims=nb_rand_dims, noise=noise, nb_cells=nb_cells)
            start = time.time()
            algo_results[j].append(algo_fs[j](env, nb_episodes, nb_dims=nb_dims+nb_rand_dims, gif=False, verbose=False, params=params[j]))
            end = time.time()
            algo_times[j].append(round(end-start))

    # Plot results
    for i, (scores, times) in enumerate(zip(algo_results, algo_times)):
        print("{}:{}: scores -> mu:{},sig{} | times -> mu:{},sig{}".format(id, names[i], np.mean(scores,axis=0)[-1],
                                                                           np.std(scores, axis=0)[-1],
                                                                           np.mean(times), np.std(times)))
    if save:
        data = [algo_results, algo_times, names, nb_episodes]
        pickle.dump(data, open("dummy_env_save_{}.pkl".format(id), "wb"))

def load_stats(id="test",fnum=0):
    from teachers.algos.plot_utils import region_plot_gif, plot_gmm, gmm_plot_gif
    per_model_colors = OrderedDict({})
    model_medians = OrderedDict({})
    try:
        scores, times, names, nb_episodes = pickle.load(open("dummy_env_save_{}.pkl".format(id), "rb"))
        if id == "6d4cells" or id == "5d4cells":
            scores2, times2, names2, nb_episodes2 = pickle.load(open("dummy_env_save_{}.pkl".format("bmm"+id), "rb"))
    except FileNotFoundError:
        print('no data for dummy_env_save_{}.pkl'.format(id))
        return 0
    if 'test_' in names[0]:
        names = [n[5:] for n in names]  # remove "test_" from names
    plt.figure(fnum)
    ax = plt.gca()
    colors = ['red','blue','green','orange','purple']
    legend = True
    max_y = 0
    for k in range(len(names)):
        if names[k] == 'our_iac':
            names[k] = "our_riac_leaves_only"
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

def load_stats_camera_ready(id="test",fnum=0):
    from teachers.algos.plot_utils import region_plot_gif, plot_gmm, gmm_plot_gif
    per_model_colors = OrderedDict({'Random': "grey",
                        'RIAC': u'#ff7f0e',
                        'ALP-GMM': u'#1f77b4',
                        'Covar_GMM': "green"})
    model_medians = OrderedDict({'ALP-GMM': None,
                                 'RIAC': None,
                                 'Covar_GMM': None,
                                 'Random': None})
    try:
        scores, times, names, nb_episodes = pickle.load(open("dummy_env_save_{}.pkl".format(id), "rb"))
        if id == "6d4cells" or id == "5d4cells":
            scores2, times2, names2, nb_episodes2 = pickle.load(open("dummy_env_save_{}.pkl".format("bmm"+id), "rb"))
    except FileNotFoundError:
        print('no data for dummy_env_save_{}.pkl'.format(id))
        return 0
    names = [n[5:] for n in names]  # remove "test_" from names
    plt.figure(fnum)
    ax = plt.gca()
    colors = ['red','blue','green','orange','purple']
    legend = True
    max_y = 0
    for i, algo_scores in enumerate(scores):
        if names[i] == "baranes_gmm":
            names[i] = "Covar_GMM"
            if id == "6d4cells" or id == "5d4cells":
                algo_scores = scores2[0]
        if 'riac' in names[i]:
            names[i] = "RIAC"
        if 'interest_gmm' in names[i]:
            names[i] = "ALP-GMM"
        if names[i] == 'random':
            names[i] = 'Random'
        ys = algo_scores
        model_medians[names[i]] = np.median(np.array(ys), axis=0)
        # print(median)
        episodes = np.arange(0,nb_episodes+1000,1000) / 1000
        #print(episodes)
        for k, y in enumerate(ys):
            if max(y) > max_y:
                max_y = max(y)
            # print("max:{} last:{}".format(max(y), y[-1]))
            ax.plot(episodes, y, color=per_model_colors[names[i]], linewidth=0.9, alpha=0.2)
    for algo_name, med in model_medians.items():
        ax.plot(episodes, med, color=per_model_colors[algo_name], linewidth=5, label=algo_name)
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




if __name__=="__main__":

    batch_exps = True
    if batch_exps:
        nb_eps = 10000
        nb_seeds = 1
        #algos = (test_interest_gmm, test_interest_gmm, test_interest_gmm, test_interest_gmm, test_interest_gmm, test_interest_gmm)
        #names = ["gmm_aic"        , "gmm"            , "gmm_ws"         , "gmm3"            , "gmm6", "gmm9"]
        #params = [{"gmm_fitness_fun":"aic"}, {}      , {"warm_start":True}, {"potential_ks":np.arange(3,11,1)}, {"potential_ks":np.arange(6,11,1)}, {"potential_ks":np.arange(9,11,1)}]

        # algos = (test_riac, test_riac, test_riac, test_riac, test_riac, test_riac)
        # names = ["our_iac",
        #          "vanilla_riac",
        #          "florensa_riac",
        #          "florensa_riac_our_params",
        #          "our_riac",
        #          "our_riac_their_params"]
        # params = [{"sampling_in_leaves_only":True},
        #           {"min_reg_size":1, "min_dims_range_ratio":1/np.inf},
        #           {"use_florensa":True},
        #           {"use_florensa":True, "max_region_size":200, "lp_window_size":200},
        #           {},
        #           {"max_region_size":500, "lp_window_size":100}]

        # algos = [test_egep]
        # names = ["egep"]
        # params = [{}]

        algos = (test_riac, test_interest_gmm, test_covar_gmm, test_random)
        names = ["RIAC", "ALP-GMM", "Covar_GMM", "Random"]
        params = [{}, {}, {}, {}]

        exp_args = [{"id":"2d10cells", "nb_episodes":nb_eps, "algo_fs":algos, "nb_seeds":nb_seeds, "params":params, "names":names},
                    {"id": "4d4cells", "nb_episodes": nb_eps*2, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cells": 4,"nb_dims": 4, "params":params, "names":names},
                     {"id": "5d4cells", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cells": 4, "nb_dims": 5, "params":params, "names":names},
                    {"id": "6d4cells", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cells": 4, "nb_dims": 6, "params":params, "names":names},
                    {"id": "2d20cellslong", "nb_episodes": nb_eps * 2, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cells": 20, "params":params, "names":names},
                    {"id": "2d50cellslong", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cells": 50, "params":params, "names":names},
                    {"id": "2d100cellslong", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cells": 100, "params":params, "names":names},
                    {"id": "2d10rd", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 10, "params":params, "names":names},
                    {"id": "2d20rd", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 20, "params":params, "names":names},
                    {"id": "2d50rd", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 50, "params":params, "names":names}]

        # exp_args = [{"id":"2d10cells_us_vs_Florensa", "nb_episodes":nb_eps, "algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
        #             {"id": "4d4cells_us_vs_Florensa", "nb_episodes": nb_eps*2, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cells": 4,"nb_dims": 4, "params": params, "names":names},
        #             {"id": "5d4cells_us_vs_Florensa", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cells": 4, "nb_dims": 5, "params": params, "names":names},
        #             {"id": "6d4cells_us_vs_Florensa", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cells": 4, "nb_dims": 6, "params": params, "names":names},
        #             {"id": "2d20cellslong_us_vs_Florensa", "nb_episodes": nb_eps * 2, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cells": 20, "params": params, "names":names},
        #             {"id": "2d50cellslong_us_vs_Florensa", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cells": 50, "params": params, "names":names},
        #             {"id": "2d100cellslong_us_vs_Florensa", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cells": 100, "params": params, "names":names},
        #             {"id": "2d10rd_us_vs_Florensa", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 10, "params": params, "names":names},
        #             {"id": "2d20rd_us_vs_Florensa", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 20, "params": params, "names":names},
        #             {"id": "2d50rd_us_vs_Florensa", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 50, "params": params, "names":names}]
        #exp_args = [{"id": "4d4cellsegep", "nb_episodes": 150000, "algo_fs": algos, "nb_seeds": 30, "nb_cells": 4,"nb_dims": 4, "params": params, "names":names}]

        if len(sys.argv) != 2:
            print('launching all experiences')
            exp_nbs = np.arange(0,len(exp_args))
        elif int(sys.argv[1]) >= len(exp_args):
            print(sys.argv[1]+": not an expe")
            exit(0)
        else:
            exp_nbs = [int(sys.argv[1])]
            print("launching expe" + sys.argv[1] + " : " + exp_args[exp_nbs[0]]["id"])


        for i in exp_nbs:
             run_stats(**exp_args[i])

        # #Display all stats
        # import matplotlib.pyplot as plt
        # all_ids = []
        # for i,exp in enumerate(exp_args):
        #     all_ids.append(exp["id"])
        #     load_stats(all_ids[-1], fnum=i)
        # plt.show()
    else:
        nb_episodes = 5000
        nb_rand_dims = 0
        nb_dims = 2
        env = NDummyEnv(nb_dims=nb_dims, nb_rand_dims=nb_rand_dims)
        test_hgmm(env, nb_episodes, nb_dims=nb_dims+nb_rand_dims, gif=True, verbose=True)
        #test_egep(env, nb_episodes, nb_dims=nb_dims+nb_rand_dims, gif=False, verbose=True)
        #test_CMAES(env, nb_episodes, gif=True)
        # env = DummyEnv()
        #score = test_random(env, nb_episodes, nb_dims=nb_dims)
        # print(score)
        #env.render()
        #


    ############## GRID SEARCH ##############
    # algo_name = "gmm"
    # param_dict = {"fit_rate":[50,100,150,200],
    #           "potential_ks":[np.arange(1,11,1), np.arange(2,11,1), np.arange(3,11,1)],
    #               "nb_em_init":[1,10]}
    # results = {}
    # import itertools
    #
    # def product_dict(**kwargs):
    #     keys = kwargs.keys()
    #     vals = kwargs.values()
    #     for instance in itertools.product(*vals):
    #         yield dict(zip(keys, instance))
    #
    # names = []
    # params = []
    # algos = []
    # for p in product_dict(**param_dict):
    #     params.append(p)
    #     names.append(algo_name+str(p['fit_rate'])+'_'+str(p['potential_ks'][0]))
    #     algos.append(test_interest_gmm)
    #
    # test_env = {"id": "4d4cells_gs_", "nb_cells": 4,"nb_dims": 4, "nb_episodes": 60000, "nb_seeds": 20, 'names': names, 'algo_fs': algos,
    #             'params': params}
    # run_stats(**test_env)
    #
    # test_env = {"id": "2d10cells_gs_", "nb_episodes": 20000, "nb_seeds": 20, 'names':names, 'algo_fs':algos, 'params':params}
    # run_stats(**test_env)
