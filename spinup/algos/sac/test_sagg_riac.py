from builtins import range
import os.path
import sys
import time
import json
import pickle
import time
import numpy as np
from gep import GEP
#from plot_utils import *
import collections
from imgep_utils.gep_utils import *
from imgep_utils.neural_network import PolicyNN
import gym
import gym_flowers
import imgep_utils.config as conf
import matplotlib.pyplot as plt

def get_o(obs):
    return obs[-2:]
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def run_episode(model, max_step=40):
    out = env.reset()
    state = get_o(out['observation'])
    nb_steps = 0
    done = False
    while not done:
        nb_steps += 1
        #print(nb_steps)
        actions = model.get_action(state.reshape(1, -1))
        out, _, done, _ = env.step(actions[0])
        state = get_o(out['observation'])
    return state

exploration_noise = 0.10
nb_bootstrap = 5
max_iterations = 5000
model_type = "random_flat"

# environment-related init
nb_arm_joints = 3
b = conf.get_env_bounds('armball_env')
# define variable's bounds for policy input and outcome
state_names = ['arm_x','arm_y']

nb_traj_steps = 1 #endpoint

# init neural network policy
input_names = state_names
input_bounds = b.get_bounds(input_names)
input_size = len(input_bounds)
print('input_bounds: %s' % input_bounds)
layers = [64]
action_set_size = 3
params = {'layers': layers, 'activation_function':'relu', 'max_a':1.,
          'dims':{'s':input_size,'a':action_set_size},'bias':True, 'size_sequential_nn':1}
nn = PolicyNN(params)
total_policy_params = get_n_params(nn)

# init IMGEP
full_outcome = input_names
objects, objects_idx = conf.get_objects('armball_env')
full_outcome = []
for obj in objects:
    full_outcome += obj * nb_traj_steps
full_outcome_bounds = b.get_bounds(full_outcome)

if (model_type == "random_flat") or (model_type == "random"):
    outcome1 = full_outcome
    config = {'policy_nb_dims': total_policy_params,
              'modules': {'mod1': {'outcome_range': np.arange(0,len(full_outcome),1),
                                   'focus_state_range': np.arange(0,len(full_outcome),1)//nb_traj_steps}}}
else:
    raise NotImplementedError
print(config)

seed = np.random.randint(1000)
np.random.seed(seed)
gep = GEP(layers, params, config, model_babbling_mode="random", explo_noise=exploration_noise)

starting_iteration = 0

env = gym.make('ArmBall-v0')
env.reset()
starting_iteration = 0
max_iterations = 5000
reached_Xs = []
reached_Ys = []
for i in range(starting_iteration, max_iterations):
    if ((i % 500) == 0): print("########### Iteration # %s ##########" % (i))
    # generate policy using gep
    policy_params, focus, add_noise = gep.produce(bootstrap=True) if i < nb_bootstrap else gep.produce()
    if add_noise:
        policy_params[0] += np.random.normal(0, exploration_noise, len(policy_params[0]))
        policy_params[0] = np.clip(policy_params[0], -1, 1)
    nn.set_parameters(policy_params[0])
    outcome = run_episode(nn)
    gep.perceive(outcome.astype('float32'), policy_params)

    reached_Xs.append(outcome[0])
    reached_Ys.append(outcome[1])

plt.figure()
plt.plot(reached_Xs, reached_Ys, marker='o.')

