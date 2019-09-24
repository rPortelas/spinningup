import numpy as np
from teachers.utils.dataset import BufferedDataset
import bisect
from gym.spaces import Box
from teachers.algos.gep_utils import proportional_choice


class EmpiricalLearningProgress():
    def __init__(self, task_size):
        self.interest_knn = BufferedDataset(1, task_size, buffer_size=500, lateness=0)
        #self.window_size = 1000

    def get_lp(self, task, competence):
        interest = 0
        if len(self.interest_knn) > 5:
            # compute learning progre   ss for new task
            dist, idx = self.interest_knn.nn_y(task)
            # closest_previous_task = previous_tasks[idx]
            closest_previous_task = self.interest_knn.get_y(idx[0])
            closest_previous_task_competence = self.interest_knn.get_x(idx[0])
            # print 'closest previous task is index:%s, val: %s' % (idx[0], closest_previous_task)

            # compute Progress as absolute difference in competence
            progress = closest_previous_task_competence - competence
            interest = np.abs(progress)

        # add to database
        self.interest_knn.add_xy(competence, task)
        return interest

class EGEP():
    def __init__(self, mins, maxs, seed=None, params={}):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        np.random.seed(self.seed)
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)
        self.random_task_ratio = 0.2
        self.mutation_noise = 1/20 * (self.maxs - self.mins)


        #self.nb_eli = False if "normalize reward" not in params else params["normalize reward"]
        self.nb_elites = 50
        self.nb_bootstrap = 1000

        self.tasks = []
        self.tasks_lps = []

        self.elite_tasks_idx = []
        self.elite_tasks_lps = []
        self.elite_tasks_nb_mutations = []

        self.nb_max_mutations = 10
        self.lp_computer = EmpiricalLearningProgress(len(mins))



    def update(self, task, reward,all_rewards=None):
        self.tasks.append(task)
        lp = self.lp_computer.get_lp(task, reward)
        self.tasks_lps.append(lp)



        insert_idx = bisect.bisect(self.elite_tasks_lps, lp)

        # insert only if elite list not full or if better than worst elite
        if len(self.elite_tasks_idx) <= self.nb_elites or insert_idx != 0:
            self.elite_tasks_idx.insert(insert_idx, len(self.tasks)-1)
            self.elite_tasks_lps.insert(insert_idx, lp)
            self.elite_tasks_nb_mutations.insert(insert_idx, 0)

            if self.nb_max_mutations in self.elite_tasks_nb_mutations:
                idx = self.elite_tasks_nb_mutations.index(self.elite_tasks_nb_mutations)
                self.elite_tasks_idx.pop(idx)
                self.elite_tasks_lps.pop(idx)
                self.elite_tasks_nb_mutations.pop(idx)
            if len(self.elite_tasks_idx) >= self.nb_elites:  # remove worst elite
                self.elite_tasks_idx = self.elite_tasks_idx[1:]
                self.elite_tasks_lps = self.elite_tasks_lps[1:]
                self.elite_tasks_nb_mutations = self.elite_tasks_nb_mutations[1:]

    def sample_task(self, kwargs=None, n_samples=1):
        if (len(self.tasks) < self.nb_bootstrap) or (np.random.random() < self.random_task_ratio):
            new_task = self.random_task_generator.sample()
        else:
            # if len(self.tasks) % 1000 == 0:
            #     #print(self.elite_tasks_lps[-15:])
            elite_idx = proportional_choice(self.elite_tasks_lps, eps=0.0)
            # if np.sum(self.elite_tasks_lps):
            #     print(self.elite_tasks_lps[-10:])
            task = self.tasks[self.elite_tasks_idx[elite_idx]]
            noise = np.random.normal(0,self.mutation_noise)
            new_task = np.clip(task + noise, self.mins, self.maxs)
        return new_task

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict
