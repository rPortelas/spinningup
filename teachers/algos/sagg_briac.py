import numpy as np
from gym.spaces import Box
from collections import deque
from teachers.algos.gep_utils import proportional_choice
import copy
from treelib import Tree
from itertools import islice
import time

class Region(object):
    def __init__(self, maxlen, cps_gs=None, bounds=None, interest=None):
        self.cps_gs = cps_gs
        self.bounds = bounds
        self.interest = interest
        self.maxlen = maxlen

    def add(self, task, comp, is_leaf):
        self.cps_gs[1].append(task.copy())
        self.cps_gs[0].append(comp)

        need_split = False
        if is_leaf and (len(self.cps_gs[0]) > self.maxlen):
            # leaf is full, lets split
            need_split = True
        return need_split





# Implementation of SAGG-RIAC
class SAGG_BRIAC():
    def __init__(self, min, max, temperature=20):  # example --> min: [-1,-1] max: [1,1]

        assert len(min) == len(max)
        self.maxlen = 200
        self.window_cp = 200
        self.minlen = self.maxlen / 20
        self.maxregions = 80

        # init regions' tree
        self.tree = Tree()
        self.regions_bounds = [Box(min, max, dtype=np.float32)]
        self.interest = [0.]
        self.tree.create_node('root','root',data=Region(maxlen=self.maxlen,
                                                        cps_gs=[deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)],
                                                        bounds=self.regions_bounds[-1], interest=self.interest[-1]))
        self.nb_dims = len(min)
        self.temperature = temperature
        self.nb_split_attempts = 50
        self.max_difference = 0.2
        self.init_size = max - min
        self.ndims = len(min)
        self.mode_3_noise = 0.1

        # book-keeping
        self.sampled_tasks = []
        self.all_boxes = []
        self.all_interests = []
        self.update_nb = 0
        self.split_iterations = []

    def compute_interest(self, sub_region):
        if len(sub_region[0]) > self.minlen:  # TRICK NB 4
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
        best_abs_interest_diff = 0
        best_bounds = None
        best_sub_regions = None
        is_split = False
        for i in range(self.nb_split_attempts):
            sub_reg1 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
            sub_reg2 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]

            # repeat until the two sub regions contain at least minlen of the mother region TRICK NB 1
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
                if np.any(bounds1.high - bounds1.low < self.init_size / 15):  # to enforce not too small boxes TRICK NB 2
                    valid_bounds = False
                if np.any(bounds2.high - bounds2.low < self.init_size / 15):
                    valid_bounds = valid_bounds and False

                # perform split in sub regions
                sub_reg1 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
                sub_reg2 = [deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]
                for i, task in enumerate(reg.cps_gs[1]):
                    if bounds1.contains(task):
                        sub_reg1[1].append(task)
                        sub_reg1[0].append(reg.cps_gs[0][i])
                    else:
                        sub_reg2[1].append(task)
                        sub_reg2[0].append(reg.cps_gs[0][i])
                sub_regions = [sub_reg1, sub_reg2]

            # compute interest
            interest = [self.compute_interest(sub_reg1), self.compute_interest(sub_reg2)]

            # compute score
            split_score = len(sub_reg1) * len(sub_reg2) * np.abs(interest[0] - interest[1])
            if split_score >= best_split_score and valid_bounds: # TRICK NB 3, max diff #and np.abs(interest[0] - interest[1]) >= self.max_difference / 8
                is_split = True
                best_abs_interest_diff = np.abs(interest[0] - interest[1])
                best_split_score = split_score
                best_sub_regions = sub_regions
                best_bounds = bounds

        if is_split:
            if best_abs_interest_diff > self.max_difference:
                self.max_difference = best_abs_interest_diff
            # add new nodes to tree
            for i, (cps_gs, bounds) in enumerate(zip(best_sub_regions, best_bounds)):
                self.tree.create_node(parent=nid, data=Region(self.maxlen, cps_gs=cps_gs, bounds=bounds, interest=interest[i]))
        else:
            #print("abort mission")
            # TRICK NB 6, remove old stuff if can't find split
            assert len(reg.cps_gs[0]) == (self.maxlen + 1)
            reg.cps_gs[0] = deque(islice(reg.cps_gs[0], int(self.maxlen / 4), self.maxlen + 1))
            reg.cps_gs[1] = deque(islice(reg.cps_gs[1], int(self.maxlen / 4), self.maxlen + 1))

        return is_split

    def merge(self, all_nodes):
        # get a list of children pairs
        parent_children = []
        for n in all_nodes:
            if not n.is_leaf():  # if node is a parent
                children = self.tree.children(n.identifier)
                if children[0].is_leaf() and children[1].is_leaf():  # both children must be leaves for an easy remove
                    parent_children.append([n, children])  # [parent, [child1, child2]]

        # sort each pair of children by their summed interest
        parent_children.sort(key=lambda x: np.abs(x[1][0].data.interest - x[1][1].data.interest), reverse=False)

        # remove useless pair
        child1 = parent_children[0][1][0]
        child2 = parent_children[0][1][1]
        # print("just removed {} and {}, daddy is: {}, childs: {}".format(child1.identifier, child2.identifier,
        #                                                                 parent_children[0][0].identifier,
        #                                                                 self.tree.children(
        #
        # print("bef")  #                                                               parent_children[0][0].identifier)))
        # print([n.identifier for n in self.tree.all_nodes()])
        self.tree.remove_node(child1.identifier)
        self.tree.remove_node(child2.identifier)
        # print("aff remove {} and {}".format(child1.identifier), child2.identifier)
        # print([n.identifier for n in self.tree.all_nodes()])

        # remove 1/4 of parent to avoid falling in a splitting-merging loop
        dadta = parent_children[0][0].data  # hahaha!
        dadta.cps_gs[0] = deque(islice(dadta.cps_gs[0], int(self.maxlen / 4), self.maxlen + 1))
        dadta.cps_gs[1] = deque(islice(dadta.cps_gs[1], int(self.maxlen / 4), self.maxlen + 1))
        self.nodes_to_recompute.append(parent_children[0][0].identifier)

        # remove child from recompute list if they where touched when adding the current task
        if child1.identifier in self.nodes_to_recompute:
            self.nodes_to_recompute.pop(self.nodes_to_recompute.index(child1.identifier))
        if child2.identifier in self.nodes_to_recompute:
            self.nodes_to_recompute.pop(self.nodes_to_recompute.index(child2.identifier))




    def add_task_comp(self, node, task, comp):
        reg = node.data
        nid = node.identifier
        if reg.bounds.contains(task): # task falls within region
            self.nodes_to_recompute.append(nid)
            children = self.tree.children(nid)
            for n in children: # if task in region, task is in one sub-region
                self.add_task_comp(n, task, comp)

            need_split = reg.add(task, comp, children == []) # COPY ALL MODE
            if need_split:
                self.nodes_to_split.append(nid)


    def update(self, task, continuous_competence, all_raw_rewards):
        # add new (task, competence) to regions nodes
        self.nodes_to_split = []
        self.nodes_to_recompute = []
        new_split = False
        root = self.tree.get_node('root')
        self.add_task_comp(root, task, continuous_competence)
        #print(self.nodes_to_split)
        assert len(self.nodes_to_split) <= 1

        # split a node if needed
        need_split = len(self.nodes_to_split) == 1
        if need_split:
            new_split = self.split(self.nodes_to_split[0])
            if new_split:
                self.update_nb += 1
                #print(self.update_nb)
                # update list of regions_bounds
                all_nodes = self.tree.all_nodes()
                if len(all_nodes) > self.maxregions:  # too many regions, lets merge one of them
                    self.merge(all_nodes)
                    all_nodes = self.tree.all_nodes()
                self.regions_bounds = [n.data.bounds for n in all_nodes]

        # recompute interests of touched nodes
        #print(self.nodes_to_recompute)
        for nid in self.nodes_to_recompute:
            #print(nid)
            node = self.tree.get_node(nid)
            reg = node.data
            reg.interest = self.compute_interest(reg.cps_gs)

        # collect new interests and new [comp, tasks] lists
        all_nodes = self.tree.all_nodes()
        self.interest = []
        self.cps_gs = []
        for n in all_nodes:
            self.interest.append(n.data.interest)
            self.cps_gs.append(n.data.cps_gs)

        # bk-keeping
        self.all_boxes.append(copy.copy(self.regions_bounds))
        self.all_interests.append(copy.copy(self.interest))
        self.split_iterations.append(self.update_nb)
        assert len(self.interest) == len(self.regions_bounds)

        return new_split, None

    def draw_random_task(self):
        return self.regions_bounds[0].sample()  # first region is root region

    def sample_task(self, args):
        mode = np.random.rand()
        if mode < 0.1:  # "mode 3" (10%) -> sample on regions and then mutate lowest-performing task in region
            if len(self.sampled_tasks) == 0:
                self.sampled_tasks.append(self.draw_random_task())
            else:
                region_id = proportional_choice(self.interest, eps=0.0)
                worst_task_idx = np.argmin(self.cps_gs[region_id][0])
                # mutate task by a small amount (i.e a gaussian scaled to the regions range)
                task = np.random.normal(self.cps_gs[region_id][1][worst_task_idx].copy(), 0.1)
                # clip to stay within region (add small epsilon to avoid falling in multiple regions)
                task = np.clip(task, self.regions_bounds[region_id].low + 1e-5, self.regions_bounds[region_id].high - 1e-5)
                self.sampled_tasks.append(task)

        elif mode < 0.3:  # "mode 2" (20%) -> random task
            self.sampled_tasks.append(self.draw_random_task())

        else:  # "mode 1" (70%) -> sampling on regions and then random task in selected region
            region_id = proportional_choice(self.interest, eps=0.0)
            self.sampled_tasks.append(self.regions_bounds[region_id].sample())


        # # sample region
        # if np.random.rand() < 0.2:
        #     region_id = np.random.choice(range(self.nb_regions))
        # else:
        #     region_id = np.random.choice(range(self.nb_regions), p=np.array(self.probas))

        # # sample task
        # self.sampled_tasks.append(self.regions_bounds[region_id].sample())
        #
        # return self.sampled_tasks[-1].tolist()
        # sample region
        # region_id = proportional_choice(self.interest, eps=0.2)
        # # sample task
        # self.sampled_tasks.append(self.regions_bounds[region_id].sample())

        return self.sampled_tasks[-1]

    def dump(self, dump_dict):
        dump_dict['all_boxes'] = self.all_boxes
        dump_dict['split_iterations'] = self.split_iterations
        dump_dict['all_interests'] = self.all_interests
        return dump_dict

    @property
    def nb_regions(self):
        return len(self.regions_bounds)

    @property
    def get_regions(self):
        return self.regions_bounds