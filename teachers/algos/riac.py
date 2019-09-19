import numpy as np
from gym.spaces import Box
from collections import deque
import copy
from treelib import Tree
from itertools import islice

def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]

class Region(object):
    def __init__(self, maxlen, cps_gs=None, bounds=None, interest=None):
        self.cps_gs = cps_gs
        self.bounds = bounds
        self.interest = interest
        self.maxlen = maxlen

    def add(self, goal, comp, is_leaf):
        self.cps_gs[1].append(goal.copy())
        self.cps_gs[0].append(comp)

        need_split = False
        if is_leaf and (len(self.cps_gs[0]) > self.maxlen):
            # leaf is full, need split
            need_split = True
        return need_split


class RIAC():
    def __init__(self, mins, maxs, seed=None, params=dict()):  # example --> mins: [-1,-1] maxs: [1,1]

        assert len(mins) == len(maxs)
        self.maxlen = 200 if "max_region_size" not in params else params['max_region_size']
        self.window_cp = self.maxlen if "lp_window_size" not in params else params['lp_window_size']

        # init regions' tree
        self.tree = Tree()
        self.regions_bounds = [Box(mins, maxs, dtype=np.float32)]
        self.interest = [0.]
        self.tree.create_node('root', 'root',
                              data=Region(maxlen=self.maxlen,
                                          cps_gs=[deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)],
                                          bounds=self.regions_bounds[-1], interest=self.interest[-1]))
        self.nb_dims = len(mins)
        self.nb_split_attempts = 50 if "nb_split_attempts" not in params else params['nb_split_attempts']
        self.ndims = len(mins)

        self.sampling_in_leaves_only = False if "sampling_in_leaves_only" not in params else params["sampling_in_leaves_only"]

        # additional tricks to original RIAC, enforcing splitting rules

        # 1 - minimum population required for both children when splitting
        self.minlen = self.maxlen / 20 if "min_reg_size" not in params else params['min_reg_size']

        # 2 - minimum children region size (compared to initial range of each dimension)
        self.dims_ranges = maxs - mins
        self.min_dims_range_ratio = 1/15 if "min_dims_range_ratio" not in params else params["min_dims_range_ratio"]

        # if after nb_split_attempts, no split is valid, flush oldest points of parent region
        self.discard_ratio = 1/4 if "discard_ratio" not in params else params["discard_ratio"]

        # book-keeping
        self.sampled_goals = []
        self.all_boxes = []
        self.all_interests = []
        self.update_nb = -1
        self.split_iterations = []

        if seed is None:
            seed = np.random.randint(42,424242)
        np.random.seed(seed)
        self.hyperparams = locals()

    def compute_interest(self, sub_region):
        if len(sub_region[0]) > 2:
            cp_window = min(len(sub_region[0]), self.window_cp)  # not completely window
            half = int(cp_window / 2)
            # print(str(cp_window) + 'and' + str(half))
            first_half = np.array(sub_region[0])[-cp_window:-half]
            snd_half = np.array(sub_region[0])[-half:]
            diff = first_half.mean() - snd_half.mean()
            cp = np.abs(diff)
        else:
            cp = 0
        interest = np.abs(cp)
        return interest

    def split(self, nid):
        # try nb_split_attempts splits
        reg = self.tree.get_node(nid).data
        best_split_score = 0
        best_bounds = None
        best_sub_regions = None
        is_split = False
        for i in range(self.nb_split_attempts):
            sub_reg1 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
            sub_reg2 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]

            # repeat until the two sub regions contain at least minlen of the mother region
            while len(sub_reg1[0]) < self.minlen or len(sub_reg2[0]) < self.minlen:
                # decide on dimension
                dim = np.random.choice(range(self.nb_dims))
                threshold = reg.bounds.sample()[dim]
                bounds1 = Box(reg.bounds.low, reg.bounds.high, dtype=np.float32)
                bounds1.high[dim] = threshold
                bounds2 = Box(reg.bounds.low, reg.bounds.high, dtype=np.float32)
                bounds2.low[dim] = threshold
                bounds = [bounds1, bounds2]
                valid_bounds = True

                if np.any(bounds1.high - bounds1.low < self.dims_ranges * self.min_dims_range_ratio):
                    valid_bounds = False
                if np.any(bounds2.high - bounds2.low < self.dims_ranges * self.min_dims_range_ratio):
                    valid_bounds = valid_bounds and False

                # perform split in sub regions
                sub_reg1 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
                sub_reg2 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
                for i, goal in enumerate(reg.cps_gs[1]):
                    if bounds1.contains(goal):
                        sub_reg1[1].append(goal)
                        sub_reg1[0].append(reg.cps_gs[0][i])
                    else:
                        sub_reg2[1].append(goal)
                        sub_reg2[0].append(reg.cps_gs[0][i])
                sub_regions = [sub_reg1, sub_reg2]

            # compute interest
            interest = [self.compute_interest(sub_reg1), self.compute_interest(sub_reg2)]

            # compute score
            split_score = len(sub_reg1) * len(sub_reg2) * np.abs(interest[0] - interest[1])
            if split_score >= best_split_score and valid_bounds:
                is_split = True
                best_split_score = split_score
                best_sub_regions = sub_regions
                best_bounds = bounds

        if is_split:
            # add new nodes to tree
            for i, (cps_gs, bounds) in enumerate(zip(best_sub_regions, best_bounds)):
                self.tree.create_node(identifier=self.tree.size(), parent=nid,
                                      data=Region(self.maxlen, cps_gs=cps_gs, bounds=bounds, interest=interest[i]))
        else:
            assert len(reg.cps_gs[0]) == (self.maxlen + 1)
            reg.cps_gs[0] = deque(islice(reg.cps_gs[0], int(self.maxlen * self.discard_ratio), self.maxlen + 1))
            reg.cps_gs[1] = deque(islice(reg.cps_gs[1], int(self.maxlen * self.discard_ratio), self.maxlen + 1))

        return is_split

    def add_goal_comp(self, node, goal, comp):
        reg = node.data
        nid = node.identifier
        if reg.bounds.contains(goal):  # goal falls within region
            self.nodes_to_recompute.append(nid)
            children = self.tree.children(nid)
            for n in children:  # if goal in region, goal is in one sub-region
                self.add_goal_comp(n, goal, comp)

            need_split = reg.add(goal, comp, children == []) # COPY ALL MODE
            if need_split:
                self.nodes_to_split.append(nid)


    def update(self, goal, continuous_competence, all_raw_rewards):
        self.update_nb += 1
        # add new (goal, competence) to regions nodes
        self.nodes_to_split = []
        self.nodes_to_recompute = []
        new_split = False
        root = self.tree.get_node('root')
        self.add_goal_comp(root, goal, continuous_competence)
        #print(self.nodes_to_split)
        assert len(self.nodes_to_split) <= 1

        # split a node if needed
        need_split = len(self.nodes_to_split) == 1
        if need_split:
            new_split = self.split(self.nodes_to_split[0])
            if new_split:
                #print(self.update_nb)
                # update list of regions_bounds
                if self.sampling_in_leaves_only:
                    self.regions_bounds = [n.data.bounds for n in self.tree.leaves()]
                else:
                    self.regions_bounds = [n.data.bounds for n in self.tree.all_nodes()]

        # recompute interests of touched nodes
        for nid in self.nodes_to_recompute:
            node = self.tree.get_node(nid)
            reg = node.data
            reg.interest = self.compute_interest(reg.cps_gs)

        # collect new interests and new [comp, goals] lists
        all_nodes = self.tree.all_nodes() if not self.sampling_in_leaves_only else self.tree.leaves()
        self.interest = []
        self.cps_gs = []
        for n in all_nodes:
            self.interest.append(n.data.interest)
            self.cps_gs.append(n.data.cps_gs)

        # bk-keeping
        if new_split:
            self.all_boxes.append(copy.copy(self.regions_bounds))
            self.all_interests.append(copy.copy(self.interest))
            self.split_iterations.append(self.update_nb)
        assert len(self.interest) == len(self.regions_bounds)

        return new_split, None

    def draw_random_goal(self):
        return self.regions_bounds[0].sample()  # first region is root region

    def sample_goal(self, args):
        mode = np.random.rand()
        if mode < 0.1:  # "mode 3" (10%) -> sample on regions and then mutate lowest-performing goal in region
            if len(self.sampled_goals) == 0:
                self.sampled_goals.append(self.draw_random_goal())
            else:
                region_id = proportional_choice(self.interest, eps=0.0)
                worst_goal_idx = np.argmin(self.cps_gs[region_id][0])
                # mutate goal by a small amount (i.e a gaussian scaled to the regions range)
                goal = np.random.normal(self.cps_gs[region_id][1][worst_goal_idx].copy(), 0.1)
                # clip to stay within region (add small epsilon to avoid falling in multiple regions)
                goal = np.clip(goal, self.regions_bounds[region_id].low + 1e-5, self.regions_bounds[region_id].high - 1e-5)
                self.sampled_goals.append(goal)

        elif mode < 0.3:  # "mode 2" (20%) -> random goal
            self.sampled_goals.append(self.draw_random_goal())

        else:  # "mode 1" (70%) -> sampling on regions and then random goal in selected region
            region_id = proportional_choice(self.interest, eps=0.0)
            self.sampled_goals.append(self.regions_bounds[region_id].sample())

        return self.sampled_goals[-1]

    def dump(self, dump_dict):
        dump_dict['all_boxes'] = self.all_boxes
        dump_dict['split_iterations'] = self.split_iterations
        dump_dict['all_interests'] = self.all_interests
        dump_dict['riac_params'] = self.hyperparams
        return dump_dict

    @property
    def nb_regions(self):
        return len(self.regions_bounds)

    @property
    def get_regions(self):
        return self.regions_bounds