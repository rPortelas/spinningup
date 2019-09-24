
import argparse
import sys
import math
import os
import os.path as osp
import tensorflow as tf
import numpy as np
from teachers.task_gan_utils.state.generator import StateGAN
from teachers.task_gan_utils.state.utils import StateCollection
#from teachers.task_gan_utils.utils import set_env_no_gpu, format_experiment_prefix
#set_env_no_gpu()

class GOAL_GAN(object):
    def __init__(self, mins, maxs, seed=None, param_dict=None):
        if param_dict is None:
            self.p = {}
            # # GeneratorEnv params
            self.p['task_size'] = len(mins)
            self.p['terminal_eps'] = lambda task_size: [math.sqrt(task_size) / math.sqrt(2) * 0.3]
            self.p['only_feasible'] = True
            self.p['task_range'] = 1  # this will be used also as bound of the state_space
            self.p['state_bounds'] = lambda task_range, task_size, terminal_eps: [
                (1, task_range) + (0.3,) * (task_size - 2) + (task_range,) * task_size]
            self.p['distance_metric'] = 'L2'
            self.p['task_weight'] = 1
            #############################################
            # task-algo params
            self.p['min_reward'] = lambda task_weight: [
                task_weight * 0.1]  # now running it with only the terminal reward of 1!
            self.p['max_reward'] = lambda task_weight: [task_weight * 0.9]
            self.p['smart_init'] = False#True
            # replay buffer
            self.p['replay_buffer'] = True
            self.p['coll_eps'] = 0.3  # lambda terminal_eps: [terminal_eps])
            self.p['num_new_tasks'] = 200
            self.p['num_old_tasks'] = 100
            # sampling params
            self.p['horizon'] = 200
            self.p['outer_iters'] = 200
            self.p['inner_iters'] = 5
            self.p['pg_batch_size'] = 20000
            # policy params
            self.p['output_gain'] = 1  # check here if it goes wrong! both were 0.1
            self.p['policy_init_std'] = 1
            self.p['learn_std'] = True
            self.p['adaptive_std'] = False
            # gan_configs
            self.p['num_labels'] = 1
            self.p['gan_generator_layers'] = [32,32]#[256, 256]
            self.p['gan_discriminator_layers'] = [16,16]#[128, 128]
            self.p['gan_noise_size'] = 4
            self.p['task_noise_level'] = 0.5
            self.p['gan_outer_iters'] = 100
            if seed is None:
                seed = np.random.randint(42,42424242)
            self.p['seed'] = seed
                
        np.random.seed(self.p['seed'])

        print("Instantiating the GAN...")
        tf_session = tf.Session()
        gan_configs = {key[4:]: value for key, value in self.p.items() if 'GAN_' in key}
    
        self.gan = StateGAN(
            state_size=self.p['task_size'],
            evaluater_size=self.p['num_labels'],
            state_range=self.p['task_range'],
            state_noise_level=self.p['task_noise_level'],
            generator_layers=self.p['gan_generator_layers'],
            discriminator_layers=self.p['gan_discriminator_layers'],
            noise_size=self.p['gan_noise_size'],
            tf_session=tf_session,
            configs=gan_configs,
        )

        final_gen_loss = 11
        k = -1
        while final_gen_loss > 10:
            k += 1
            self.gan.gan.initialize()
            # img = plot_gan_samples(gan, self.p['task_range'], '{}/start.png'.format(log_dir))
            print("pretraining the GAN...")
            if self.p['smart_init']:
                initial_tasks = generate_initial_tasks(env, policy, self.p['task_range'], horizon=self.p['horizon'])
                if np.size(initial_tasks[0]) == 2:
                    plt.figure()
                    plt.scatter(initial_tasks[:, 0], initial_tasks[:, 1], marker='x')
                    plt.xlim(-self.p['task_range'], self.p['task_range'])
                    plt.ylim(-self.p['task_range'], self.p['task_range'])
                    img = save_image()
                    report.add_image(img, 'tasks sampled to pretrain GAN: {}'.format(np.shape(initial_tasks)))
                dis_loss, gen_loss = gan.pretrain(
                    initial_tasks, outer_iters=30
                    # initial_tasks, outer_iters=30, generator_iters=10, discriminator_iters=200,
                )
                final_gen_loss = gen_loss
                print("Loss at the end of {}th trial: {}gen, {}disc".format(k, gen_loss, dis_loss))
            else:
                self.gan.pretrain_uniform()
                final_gen_loss = 0
            #print("Plotting GAN samples")
            #img = plot_gan_samples(gan, self.p['task_range'], '{}/start.png'.format(log_dir))
            # report.add_image(img, 'GAN pretrained %i: %i gen_itr, %i disc_itr' % (k, 10 + k, 200 - k * 10))

        self.all_tasks = StateCollection(self.p['coll_eps'])

        self.raw_tasks = []
        self.rewards = []

    def sample_task(self, kwargs=None, n_samples=1):
        print("Sampling tasks...")
        raw_task, _ = self.gan.sample_states_with_noise(1)
        self.raw_tasks.append(raw_task[0])
        return raw_task[0]

    def update(self, task, competence, all_rewards=None):
        self.rewards.append(competence)
        if len(self.raw_tasks) == self.p['num_new_tasks']:  # update time
            assert(len(self.rewards) == self.p['num_new_tasks'])

            if self.p['replay_buffer'] and self.all_tasks.size > 0:
                # add old tasks to new ones
                old_tasks = self.all_tasks.sample(self.p['num_old_tasks'])
                # print("old_tasks: {}, raw_tasks: {}".format(old_tasks, raw_tasks))
                tasks = np.vstack([self.raw_tasks, old_tasks])
            else:
                tasks = self.raw_tasks

            rewards_before = None
            if self.p['num_labels'] == 3:
                rewards_before = evaluate_states(tasks, env, policy, self.p['horizon'], n_traj=n_traj)

            # this re-evaluate the final policy in the collection of tasks
            print("Generating labels by re-evaluating policy on List of tasks...")
            labels = np.vstack([np.array(self.rewards) > 0.1, np.array(self.rewards) < 0.9]).astype(np.float32)
            if self.p['num_labels'] == 1:
                labels = np.logical_and(labels[0, :], labels[1, :]).astype(int).reshape((-1, 1))
            print("Training GAN...")
            self.gan.train(
                tasks, labels,
                self.p['gan_outer_iters'],
            )

            # append new tasks to list of all tasks (replay buffer): Not the low reward ones!!
            filtered_raw_tasks = [task for task, label in zip(tasks, labels) if label[0] == 1]
            self.all_tasks.append(filtered_raw_tasks)


if __name__ == '__main__':
    GGAN = GOAL_GAN([0.,0.],[1.,1.])
    g = GGAN.sample_task()
    GGAN.update(g,0.95)


