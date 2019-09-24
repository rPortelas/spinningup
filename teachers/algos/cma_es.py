import cma
import numpy as np
import matplotlib.pyplot as plt
from teachers.algos.alp_gmm import EmpiricalLearningProgress
from gym.spaces import Box

def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

class InterestCMAES():
    '''CMA-ES wrapper.'''
    def __init__(self, num_params,      # number of model parameters
                 sigma_init=0.1,       # initial standard deviation
                 popsize=100, mins=[0.0,0.0], maxs=[1.0,1.0]):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.es = cma.CMAEvolutionStrategy( self.num_params * [0.5],
                                            self.sigma_init,
                                            {'popsize': self.popsize,
                                             'CMA_diagonal': 0,
                                             'CMA_active':True,
                                             'bounds':[0,1]})
        self.lp_computer = EmpiricalLearningProgress(num_params)
        self.counter = 0
        self.current_generation = None
        self.current_fitnesses = []
        self.random_task = 0.0
        self.random_task_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)

    def sample_task(self):
        if self.counter == 0:
            self.current_fitnesses = []
            self.current_generation = np.array(self.es.ask())
            self.counter = self.popsize

        self.counter -= 1
        if np.random.random() < self.random_task:
            #print('RANDOM GOAL')
            # sample random task
            self.current_generation[self.counter] = self.random_task_generator.sample()
        return np.array(self.current_generation[self.counter])

    def update(self, task, competence):  # WARNING GOALS ARENT CLIPPED
        self.current_fitnesses.append(self.lp_computer.get_lp(task, competence))
        if len(self.current_fitnesses) == self.popsize:
          # convert minimizer to maximizer.
          fitnesses = -np.array(self.current_fitnesses)
          # feed fitnesses to ES
          self.es.tell(self.current_generation.tolist(), fitnesses.tolist())
          # plotting and logging
          self.es.logger.add()
          assert self.counter == 0

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma*sigma))

    def current_param(self):
        return self.es.result[5] # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0] # best evaluated solution

    def result(self): # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])