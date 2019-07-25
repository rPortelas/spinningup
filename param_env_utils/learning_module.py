import numpy as np
from param_env_utils.imgep_utils.gep_utils import get_random_policy
from param_env_utils.imgep_utils.dataset import BufferedDataset
import copy
from active_goal_sampling import SAGG_RIAC


class LearningModule(object):
    # outcome_bounds must be a 2d array with column 1 = mins and column 2 = maxs
    def __init__(self, policy_nb_dims, layers, init_function_params, outcome_size, babbling_mode, explo_noise=0.1,
                 update_interest_step=5, mean_rate=100., goal_sampling='random'):
        self.policy_nb_dims = policy_nb_dims
        self.layers = layers
        self.init_function_params = init_function_params
        self.o_size = outcome_size
        self.explo_noise = explo_noise
        self.babbling_mode = babbling_mode
        self.goal_sampling = goal_sampling

        self.generated_goals = []
        self.observed_outcomes = []
        self.LOG = False

        if self.babbling_mode == "active":
            self.mean_rate = mean_rate # running mean window
            self.interest = 0
            self.progress = 0
            self.interest_knn = BufferedDataset(self.o_size, self.o_size, buffer_size=200, lateness=0)
            self.update_interest_step = update_interest_step # 4 exploration for 1 exploitation
            self.counter = 0


        self.knn = BufferedDataset(1, self.o_size, buffer_size=1000, lateness=0) #use index instead of policies

        self.current_goal = None
        if self.goal_sampling == 'active':
            self.act_goal_sampling = SAGG_RIAC(np.array([-1.]*self.o_size),
                                               np.array([1.]*self.o_size))

    # sample a goal in outcome space and find closest neighbor in (param,outcome) database
    # RETURN policy param with added gaussian noise
    def produce(self, policies, goal=None, logboy=False):
        if goal: # test time, no noise
            _, policy_idx = self.knn.nn_y(goal)
            policy = copy.deepcopy(policies[policy_idx[0]])
            return policy, False

        # draw random goal in bounded outcome space
        if self.goal_sampling != 'active':
            goal = np.random.random(self.o_size) * 2 - 1
            self.current_goal = goal
        else: #active sampling using RIAC
            goal = self.act_goal_sampling.sample_goal()
            self.current_goal = goal


        if self.LOG: print("goal is {} {}".format(goal[0:3], goal.shape))
        add_noise = True

        if self.babbling_mode == "active":
            #print self.counter
            self.counter += 1
            if self.update_interest_step == 1: #compute noisy interest at every step
                add_noise = True
            elif (self.counter % self.update_interest_step) == 0: #exploitation step
                add_noise = False
                self.generated_goals.append(goal)

        # get closest outcome in database and retreive corresponding policy
        _, policy_idx = self.knn.nn_y(goal)

        #if logboy: print("nb:{} val:{}".format(policy_idx, policies[policy_idx[0]][155]))

        policy_knn_idx = self.knn.get_x(policy_idx[0])
        if logboy: print(policy_knn_idx)
        assert(policy_idx[0] == policy_knn_idx)
        policy = copy.deepcopy(policies[policy_idx[0]])

        # add gaussian noise for exploration
        if add_noise:
            if policy_idx[0] == 0:  # the first ever seen is the best == we found nothing, revert to random motor
                if logboy: print("{} reveeeeert".format(policy_idx))
                policy = get_random_policy(self.layers, self.init_function_params)
                add_noise = False
            else:
                pass # noise will be added at run time
        if logboy: print("noise: {} {}: before: {}, after: {}, ({})".format(add_noise, self.counter, policies[policy_idx[0]][0][155], policy[0][155], self.explo_noise))
        return policy, add_noise, goal

    def perceive(self, policy_idx, outcome): # must be called for each episode
        # add to knn
        self.knn.add_xy(policy_idx, outcome)

        if self.goal_sampling == 'active':
            if self.current_goal is not None:
                #print("goal:{},out:{},comp:{}".format(self.current_goal, outcome,
                #                                      1 - (np.linalg.norm(self.current_goal - outcome) / self.o_size)))
                self.split, self.order = self.act_goal_sampling.update([self.current_goal], None,
                                              continuous_competence=[1 - (np.linalg.norm(self.current_goal - outcome) / self.o_size)])

    def update_interest(self, outcome): # must be called only if module is selected
        if self.babbling_mode == "active":
            # update interest, only if:
            # - not in bootstrap phase since no goal is generated during this phase
            # - not in an exploration phase (update progress when exploiting for better accuracy)
            if len(self.generated_goals) < 3 and ((self.counter % self.update_interest_step) == 0):
                self.interest_knn.add_xy(outcome, self.generated_goals[-1])
                if ((self.counter % self.update_interest_step) == 0):
                    self.counter = 0  # reset counter
                return
            elif ((self.counter % self.update_interest_step) == 0):
                self.counter = 0 # reset counter
                #print 'updating interest'
                #assert(len(self.generated_goals) == (len(self.observed_outcomes) + 1))
                #previous_goals = self.generated_goals[:-1]
                current_goal = self.generated_goals[-1]
                #print 'current_generated_goal: %s, with shape: %s' % (current_goal,current_goal.shape)
                #print 'previous_generated_goal: %s, with shape: %s' % (previous_goals,previous_goals.shape)
                # find closest previous goal to current goal
                dist, idx = self.interest_knn.nn_y(current_goal)
                #closest_previous_goal = previous_goals[idx]
                closest_previous_goal = self.interest_knn.get_y(idx[0])
                closest_previous_goal_outcome = self.interest_knn.get_x(idx[0])
                #print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)
                # retrieve old outcome corresponding to closest previous goal
                #closest_previous_goal_outcome = self.observed_outcomes[idx]

                # compute Progress as dist(s_g,s') - dist(s_g,s)
                # with s_g current goal and s observed outcome
                # s_g' closest previous goal and s' its observed outcome
                #print 'old interest: %s' % self.interest
                dist_goal_old_outcome = np.linalg.norm(current_goal - closest_previous_goal_outcome) / self.o_size
                dist_goal_cur_outcome = np.linalg.norm(current_goal - outcome) / self.o_size
                progress = dist_goal_old_outcome - dist_goal_cur_outcome
                self.progress = ((self.mean_rate-1)/self.mean_rate) * self.progress + (1/self.mean_rate) * progress
                self.interest = np.abs(self.progress)

                #update observed outcomes
                #self.observed_outcomes.append(outcome)
                #self.interest_knn.add(self.generated_goals[-1])
                self.interest_knn.add_xy(outcome, self.generated_goals[-1])
            else:
                pass
