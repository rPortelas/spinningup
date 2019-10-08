import numpy as np
import scipy.stats as sp
import time
from teachers.algos.riac import RIAC
from teachers.algos.alp_gmm import ALPGMM
from teachers.algos.covar_gmm import CovarGMM
from teachers.utils.plot_utils import region_plot_gif, gmm_plot_gif
import matplotlib.pyplot as plt
import pickle
import copy
import sys
from collections import OrderedDict
import seaborn as sns; sns.set()

#from teachers.algos.egep import EGEP
#from teachers.algos.hgmm import HGMM
#from teachers.algos.cma_es import InterestCMAES
#import cProfile
#from teachers.algos.plot_utils import egep_plot_gif, hgmm_plot_gif
#from teachers.active_task_sampling import SAGG_IAC
#from teachers.algos.florensa_riac import Florensa_RIAC


# A simple n-dimensional toy parameter space to test teacher algorithms
class ToyEnv(object):  # n-dimensional grid
    def __init__(self, nb_cubes=10, nb_dims=2, noise=0.0):
        self.nb_cubes = nb_cubes  # Number of hypercubes per dimensions
        self.nb_dims = nb_dims  # Number of dimensions

        self.nb_total_cubes = nb_cubes ** nb_dims
        self.step_size = 1/nb_cubes
        self.bnds = [np.arange(0,1+self.step_size,self.step_size) for i in range(nb_dims)]
        self.params = []
        self.cube_competence = np.zeros((nb_cubes, ) * nb_dims)
        self.noise = noise
        self.max_per_cube = 100

    def reset(self):
        self.cube_competence = np.zeros((nb_cubes,) * nb_dims)
        self.params = []

    def get_score(self):  # Returns the percentage of "mastered" hypercubes (A cube is "mastered" if its competence >75)
        score = np.where(self.cube_competence > (3*(self.max_per_cube/4)))  #
        return (len(score[0]) / self.nb_total_cubes)*100

    def episode(self, param):
        # Ensure param values fall in bounds
        for v in param:
            if (v < 0.0) or (v > 1.0):
                print('param is out of bounds')
                exit(1)
        p = param[0:self.nb_dims]  # discard potential useless dimensions
        self.params.append(p)

        # 1 - Find in which hypercube the parameter vector falls
        arr_p = np.array([p])
        cubes = sp.binned_statistic_dd(arr_p, np.ones(arr_p.shape), 'count',
                                       bins=self.bnds).statistic
        cube_idx = tuple([v[0] for v in cubes[0].nonzero()])

        # 2 - Check if hypercube is "unlocked" by checking if a previous adjacent neighbor is unlocked
        if all(v == 0 for v in cube_idx):  # If initial cube, no need to have unlocked neighbors to learn
            self.cube_competence[cube_idx] = min(self.cube_competence[cube_idx] + 1, self.max_per_cube)
        else: # Find index of previous adjacent neighboring hypercubes
            prev_cube_idx = [[idx, max(0, idx - 1)] for idx in cube_idx]
            previous_neighbors_idx = np.array(np.meshgrid(*prev_cube_idx)).T.reshape(-1,len(prev_cube_idx))
            for pn_idx in previous_neighbors_idx:
                prev_idx = tuple(pn_idx)
                if all(v == cube_idx[i] for i,v in enumerate(prev_idx)):  # Original hypercube, not previous neighbor
                    continue
                else:
                    if self.cube_competence[prev_idx] >= (3*(self.max_per_cube/4)):  # Previous neighbor with high comp
                        self.cube_competence[cube_idx] = min(self.cube_competence[cube_idx] + 1, self.max_per_cube)
                        break
        normalized_competence = np.interp(self.cube_competence[cube_idx], (0, self.max_per_cube), (0, 1))
        # if self.noise >= 0.0:
        #     normalized_competence = np.clip(normalized_competence + np.random.normal(0,self.noise), 0, 1)
        return normalized_competence


# Controller functions for various teacher algorithms
def test_riac(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    # Init teacher
    task_generator = RIAC(np.array([0.0] * nb_dims), np.array([1.0]*nb_dims), params=params)

    # Init book keeping
    all_boxes = []
    iterations = []
    alps = []
    rewards = []
    scores = []

    # Launch run
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i, scores[-1]))
        task = task_generator.sample_task()
        reward = env.episode(task)
        split, _ = task_generator.update(np.array(task), reward)

        # Book keeping if RIAC performed a new split
        if split and gif:
            boxes = task_generator.regions_bounds
            alp = task_generator.regions_alp
            alps.append(copy.copy(alp))
            iterations.append(i)
            all_boxes.append(copy.copy(boxes))
        rewards.append(reward)

    if gif and nb_dims==2:
        region_plot_gif(all_boxes, alps, iterations, task_generator.sampled_tasks,
                        gifname='riac_'+str(time.time()), ep_len=[1]*nb_episodes, rewards=rewards, gifdir='gifs/')
    return scores

def test_alpgmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    # Init teacher
    task_generator = ALPGMM([0] * nb_dims, [1] * nb_dims, params=params)

    # Init book keeping
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'tasks_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}

    # Launch run
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i, scores[-1]))

        # Book keeping if ALP-GMM updated its GMM
        if i>100 and (i % task_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(task_generator.gmm.weights_.copy())
            bk['covariances'].append(task_generator.gmm.covariances_.copy())
            bk['means'].append(task_generator.gmm.means_.copy())
            bk['tasks_lps'] = task_generator.tasks_alps
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['comp_grids'].append(env.cube_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())

        task = task_generator.sample_task()
        reward = env.episode(task)
        task_generator.update(np.array(task), reward)
        rewards.append(reward)

    if gif and nb_dims==2:
        gmm_plot_gif(bk, gifname='alpgmm_'+str(time.time()), gifdir='gifs/')
    return scores

def test_covar_gmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
    # Init teacher
    task_generator = CovarGMM([0] * nb_dims, [1] * nb_dims, params=params)

    # Init book keeping
    rewards = []
    scores = []
    bk = {'weights':[], 'covariances':[], 'means':[], 'tasks_lps':[], 'episodes':[],
          'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}

    # Launch run
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i,scores[-1]))

        # Book keeping if Covar-GMM updated its GMM
        if i>100 and (i % task_generator.fit_rate) == 0 and (gif is True):
            bk['weights'].append(task_generator.gmm.weights_.copy())
            bk['covariances'].append(task_generator.gmm.covariances_.copy())
            bk['means'].append(task_generator.gmm.means_.copy())
            bk['tasks_lps'] = task_generator.tasks_times_rewards
            bk['episodes'].append(i)
            if nb_dims == 2:
                bk['comp_grids'].append(env.cube_competence.copy())
                bk['comp_xs'].append(env.bnds[0].copy())
                bk['comp_ys'].append(env.bnds[1].copy())
        task = task_generator.sample_task()
        reward = env.episode(task)
        task_generator.update(np.array(task), reward)
        rewards.append(reward)

    if gif and nb_dims==2:
        gmm_plot_gif(bk, gifname='covargmm_'+str(time.time()), gifdir='gifs/')
    return scores

# def test_egep(env, nb_episodes, gif=False, nb_dims=2, score_step=1000, verbose=True, params={}):
#     task_generator = EGEP([0]*nb_dims, [1]*nb_dims, params=params)
#     rewards = []
#     scores = []
#     bk = {"elites_idx":[], 'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
#     for i in range(nb_episodes+1):
#         if (i % score_step) == 0:
#             #print(i)
#             scores.append(env.get_score())
#             if verbose:
#                 print(scores[-1])
#             bk["elites_idx"].append(copy.copy(task_generator.elite_tasks_idx))
#             if nb_dims == 2:
#                 bk['comp_grids'].append(env.cube_competence.copy())
#                 bk['comp_xs'].append(env.bnds[0].copy())
#                 bk['comp_ys'].append(env.bnds[1].copy())
#
#         task = task_generator.sample_task()
#         reward = env.episode(task)
#         task_generator.update(np.array(task), reward)
#         rewards.append(reward)
#     bk['tasks'] = task_generator.tasks
#     bk['lps'] = task_generator.tasks_lps
#     if gif and nb_dims==2 and nb_dims==2:
#         egep_plot_gif(bk, gifname='egep'+str(time.time()), gifdir='gifs/')
#     return scores

# def test_hgmm(env, nb_episodes, gif=True, nb_dims=2, score_step=1000, verbose=True, params={}):
#     task_generator = HGMM([0]*nb_dims, [1]*nb_dims, params=params)
#     rewards = []
#     scores = []
#     bk = {'comp_grids':[], 'comp_xs':[], 'comp_ys':[]}
#     for i in range(nb_episodes+1):
#         if (i % score_step) == 0:
#             scores.append(env.get_score())
#             if nb_dims == 2:
#                 if verbose:
#                     print(env.cube_competence)
#             else:
#                 if verbose:
#                     print(scores[-1])
#         if i>100 and (i % task_generator.fit_rate) == 0 and (gif is True):
#             if nb_dims == 2:
#                 bk['comp_grids'].append(env.cube_competence.copy())
#                 bk['comp_xs'].append(env.bnds[0].copy())
#                 bk['comp_ys'].append(env.bnds[1].copy())
#
#         task = task_generator.sample_task()
#         reward = env.episode(task)
#         task_generator.update(np.array(task), reward)
#         rewards.append(reward)
#     if gif and nb_dims==2:
#         task_generator.dump(bk)
#         hgmm_plot_gif(bk, gifname='Hgmm'+str(time.time()), gifdir='gifs/')
#     return scores

def test_random(env, nb_episodes, nb_dims=2, gif=False, score_step=1000, verbose=True, params={}):
    scores = []
    for i in range(nb_episodes+1):
        if (i % score_step) == 0:
            scores.append(env.get_score())
            if nb_dims == 2:
                if verbose:
                    print(env.cube_competence)
            else:
                if verbose:
                    print("it:{}, score:{}".format(i, scores[-1]))
        p = np.random.random(nb_dims)
        env.episode(p)
    return scores


# TODO REMOVE FOR RELEASE
def run_stats(nb_episodes=20000, nb_dims=2, nb_useless_dims = 0, algo_fs=None,
              nb_seeds=100, noise=0.0, nb_cubes=10, id="test", params={}, names=[], save=True):
    print("starting stats on {}".format(id))
    algo_results, algo_times = [[] for _ in range(len(algo_fs))], [[] for _ in range(len(algo_fs))]
    if len(names) == 0:
        names = [str(f).split(" ")[1] for f in algo_fs]
    for i in range(nb_seeds):
        print(i)
        for j in range(len(algo_fs)):
            env = ToyEnv(nb_dims=nb_dims, noise=noise, nb_cubes=nb_cubes)
            start = time.time()
            algo_results[j].append(algo_fs[j](env, nb_episodes, nb_dims=nb_dims+nb_useless_dims, gif=False,
                                              verbose=False, params=params[j]))
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

# TODO REMOVE FOR RELEASE
def load_stats(id="test",fnum=0, folder=''):
    from teachers.utils.plot_utils import region_plot_gif, plot_gmm, gmm_plot_gif
    per_model_colors = OrderedDict({})
    model_medians = OrderedDict({})
    try:
        scores, times, names, nb_episodes = pickle.load(open(folder + "dummy_env_save_{}.pkl".format(id), "rb"))
        # if id == "6d4cubes" or id == "5d4cubes":
        #     scores2, times2, names2, nb_episodes2 = pickle.load(open(folder + "dummy_env_save_{}.pkl".format("bmm"+id), "rb"))
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
    ax.set_ylabel('% Mastered cubes', fontsize=20)
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

# TODO REMOVE FOR RELEASE
def load_stats_camera_ready(id="test",fnum=0, folder='final_aic_min2gmm'):
    from teachers.utils.plot_utils import region_plot_gif, plot_gmm, gmm_plot_gif
    per_model_colors = OrderedDict({'Random': "grey",
                        'RIAC': u'#ff7f0e',
                        'ALP-GMM': u'#1f77b4',
                        'Covar-GMM': "green"})
    model_medians = OrderedDict({'ALP-GMM': None,
                                 'RIAC': None,
                                 'Covar-GMM': None,
                                 'Random': None})
    try:
        # legacy arm-bending
        id = id.replace('cubes', 'cells')
        scores, times, names, nb_episodes = pickle.load(open(folder + "dummy_env_save_{}.pkl".format(id), "rb")) #TODO

    except FileNotFoundError:
        print('no data for toy_env_save_{}.pkl'.format(id))
        return 0
    #names = [n[5:] for n in names]  # remove "test_" from names
    plt.figure(fnum)
    ax = plt.gca()
    colors = ['red','blue','green','orange','purple']
    legend = True
    max_y = 0
    for i, algo_scores in enumerate(scores):
        if names[i] == "Covar_GMM":
            names[i] = "Covar-GMM"
        if 'riac' in names[i]:
            names[i] = "RIAC"
        if 'alpgmm' in names[i]:
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
    ax.set_ylabel('% Mastered cubes', fontsize=20)
    ax.set_xlim(xmin=0, xmax=nb_episodes/1000)
    ax.set_ylim(ymin=0, ymax=max_y)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=15)
    if legend:
        leg = ax.legend(loc='bottom right', fontsize=14)

    if "long" in id:
        id = id[:-4]
    old_id = id
    id = id.replace("cells", "cubes")
    id = id.replace("cubes"," cubes/D")
    id = id.replace("d", "-D   ",1)
    id = id.replace("rd", "-uselessD", 1)
    if "uselessD" in id:
        id = id[0:3] + " + " + id[6:]
    ax.set_title(id, fontsize=28)

    print("{}: Algo:{} times -> mu:{},sig{}".format(id, names[i], np.mean(times[i]), np.std(times[i])))
    plt.tight_layout()
    plt.savefig(old_id+'.png')
    plt.savefig(old_id + '.svg')
    plt.savefig(old_id + '.pdf')




if __name__=="__main__":

    batch_exps = False
    if batch_exps: # TODO REMOVE FOR RELEASE
        nb_eps = 10000
        nb_seeds = 1
        #algos = (test_alpgmm, test_alpgmm, test_alpgmm, test_alpgmm, test_alpgmm, test_alpgmm)
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



        # EXPE PAPIER
        algos = (test_riac, test_alpgmm, test_covar_gmm, test_random)
        names = ["RIAC", "ALP-GMM", "Covar_GMM", "Random"]
        params = [{}, {}, {}, {}]

        exp_args = [{"id":"2d10cubes", "nb_episodes":nb_eps, "algo_fs":algos, "nb_seeds":nb_seeds, "params":params, "names":names},
                    {"id": "4d4cubes", "nb_episodes": nb_eps*2, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cubes": 4,"nb_dims": 4, "params":params, "names":names},
                     {"id": "5d4cubes", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cubes": 4, "nb_dims": 5, "params":params, "names":names},
                    {"id": "6d4cubes", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cubes": 4, "nb_dims": 6, "params":params, "names":names},
                    {"id": "2d20cubeslong", "nb_episodes": nb_eps * 2, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cubes": 20, "params":params, "names":names},
                    {"id": "2d50cubeslong", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cubes": 50, "params":params, "names":names},
                    {"id": "2d100cubeslong", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cubes": 100, "params":params, "names":names},
                    {"id": "2d10rd", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 10, "params":params, "names":names},
                    {"id": "2d20rd", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 20, "params":params, "names":names},
                    {"id": "2d50rd", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 50, "params":params, "names":names}]
        # -----------------------------

        # # EXPE ALP Window
        # algos = [test_alpgmm]
        # names = ["ALP-GMM-win01"]
        # params = [{"alp_max_size": 500}]
        # nb_eps = 50000
        # nb_seeds = 10
        # exp_args = [{"id":"2d10cubesALPWIN3", "nb_episodes":nb_eps, "algo_fs":algos, "nb_seeds":nb_seeds, "params":params, "names":names}]


        # exp_args = [{"id":"2d10cubes_us_vs_Florensa", "nb_episodes":nb_eps, "algo_fs":algos, "nb_seeds":nb_seeds, "params": params, "names":names},
        #             {"id": "4d4cubes_us_vs_Florensa", "nb_episodes": nb_eps*2, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cubes": 4,"nb_dims": 4, "params": params, "names":names},
        #             {"id": "5d4cubes_us_vs_Florensa", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cubes": 4, "nb_dims": 5, "params": params, "names":names},
        #             {"id": "6d4cubes_us_vs_Florensa", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_cubes": 4, "nb_dims": 6, "params": params, "names":names},
        #             {"id": "2d20cubeslong_us_vs_Florensa", "nb_episodes": nb_eps * 2, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cubes": 20, "params": params, "names":names},
        #             {"id": "2d50cubeslong_us_vs_Florensa", "nb_episodes": nb_eps * 5, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cubes": 50, "params": params, "names":names},
        #             {"id": "2d100cubeslong_us_vs_Florensa", "nb_episodes": nb_eps * 10, "algo_fs": algos, "nb_seeds": nb_seeds,"nb_cubes": 100, "params": params, "names":names},
        #             {"id": "2d10rd_us_vs_Florensa", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 10, "params": params, "names":names},
        #             {"id": "2d20rd_us_vs_Florensa", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 20, "params": params, "names":names},
        #             {"id": "2d50rd_us_vs_Florensa", "nb_episodes": nb_eps, "algo_fs": algos, "nb_seeds": nb_seeds, "nb_rand_dims": 50, "params": params, "names":names}]
        #exp_args = [{"id": "4d4cubesegep", "nb_episodes": 150000, "algo_fs": algos, "nb_seeds": 30, "nb_cubes": 4,"nb_dims": 4, "params": params, "names":names}]

        if len(sys.argv) != 2:
            print('launching all experiences')
            exp_nbs = np.arange(0,len(exp_args))
        elif int(sys.argv[1]) >= len(exp_args):
            print(sys.argv[1]+": not an expe")
            exit(0)
        else:
            exp_nbs = [int(sys.argv[1])]
            print("launching expe" + sys.argv[1] + " : " + exp_args[exp_nbs[0]]["id"])

        #
        # for i in exp_nbs:
        #      run_stats(**exp_args[i])

        #Display all stats
        import matplotlib.pyplot as plt
        all_ids = []
        for i,exp in enumerate(exp_args):
            all_ids.append(exp["id"])
            load_stats_camera_ready(all_ids[-1], fnum=i, folder="final_aic_min2gmm/")
            #load_stats(all_ids[-1], fnum=i, folder="")
        plt.show()
    else:
        nb_episodes = 50000
        nb_dims = 2
        nb_cubes = 10
        score_step = 1000
        env = ToyEnv(nb_dims=nb_dims, nb_cubes=nb_cubes)
        all_scores = []
        colors = ['r','g','blue','black']
        all_scores.append(test_random(env, nb_episodes, nb_dims, score_step=score_step, verbose=True))
        env.reset()
        all_scores.append(test_riac(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))
        env.reset()
        all_scores.append(test_alpgmm(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))
        env.reset()
        all_scores.append(test_covar_gmm(env, nb_episodes, gif=True, nb_dims=nb_dims, score_step=score_step, verbose=True))


        # Plot evolution of % of mastered hypercubes
        episodes = np.arange(0, nb_episodes + score_step, score_step) / score_step
        ax = plt.gca()
        for scores, color in zip(all_scores, colors):
            ax.plot(episodes, scores, color=color, linewidth=5)
        ax.set_xlabel('Episodes (x1000)', fontsize=20)
        ax.set_ylabel('% Mastered cubes', fontsize=20)
        ax.set_xlim(xmin=0, xmax=nb_episodes / score_step)
        ax.set_ylim(ymin=0, ymax=100)
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.show()






        #test_egep(env, nb_episodes, nb_dims=nb_dims+nb_useless_dims, gif=False, verbose=True)
        #test_CMAES(env, nb_episodes, gif=True)
        # env = DummyEnv()
        #score = test_random(env, nb_episodes, nb_dims=nb_dims)
        # print(score)
        #env.render()
        #

    # TODO REMOVE FOR RELEASE
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
    #     algos.append(test_alpgmm)
    #
    # test_env = {"id": "4d4cubes_gs_", "nb_cubes": 4,"nb_dims": 4, "nb_episodes": 60000, "nb_seeds": 20, 'names': names, 'algo_fs': algos,
    #             'params': params}
    # run_stats(**test_env)
    #
    # test_env = {"id": "2d10cubes_gs_", "nb_episodes": 20000, "nb_seeds": 20, 'names':names, 'algo_fs':algos, 'params':params}
    # run_stats(**test_env)
