import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import numpy as np
from spinup.utils.normalization_utils import MaxMinFilter
import gym
import gym_flowers
import imageio
from matplotlib import pyplot as plt


def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True, make_gif=True):
    #env = gym.make('flowers-Walker-continuous-v0')
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    env_babbling = "random"
    norm_obs = False

    def get_mu_sigma(v_min, v_max):  # assumes sigma has same bounds as mu
        random_2dparams = np.random.uniform(v_min, v_max, 2)
        return random_2dparams.tolist()  # returning mu and sigma
    def set_test_env_params(**kwargs):
        random_tunnel_h = None
        random_stump_h = None
        if kwargs['stump_height'] is not None:
            random_stump_h = get_mu_sigma(kwargs['stump_height'][0], kwargs['stump_height'][1])
            random_stump_h[1] = 0.1
        if kwargs['tunnel_height'] is not None:
            random_tunnel_h = get_mu_sigma(kwargs['tunnel_height'][0], kwargs['tunnel_height'][1])
            random_tunnel_h[1] = 0.1
        env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                tunnel_height=random_tunnel_h, obstacle_spacing=kwargs['obstacle_spacing'],
                                gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                step_number=kwargs['step_number'])

    env_kwargs = {'roughness':None,
                  'stump_height':[1.0, 1.5],#stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
                  'tunnel_height':None,
                  'obstacle_spacing':2.5,
                  'gap_width':None,
                  'step_height':None,
                  'step_number':None}

    if norm_obs:
        norm = MaxMinFilter(env_params_dict=env_kwargs)

    images = []

    set_test_env_params(**env_kwargs)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    img = env.render(mode='rgb_array')
    o = norm(o) if norm_obs else o
    obss = [o]
    skip = 2
    cpt = 0

    while n < num_episodes:
        if render:
            cpt+=1
            if cpt == skip:
                if make_gif:
                    images.append(img)
                    img = env.render(mode='rgb_array')
                else:
                    env.render()
                cpt = 0
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        o = norm(o) if norm_obs else o
        obss.append(o)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            set_test_env_params(**env_kwargs)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            o = norm(o) if norm_obs else o
            n += 1
            #print("MAX:{}".format(np.max(obss, axis=0)))
            #print("MIN:{}".format(np.min(obss,axis=0)))


    # logger.log_tabular('EpRet', with_min_and_max=True)
    # logger.log_tabular('EpLen', average_only=True)
    # logger.dump_tabular()
    # print(len(images))
    # print(np.array(images[0]).shape)
    # imageio.mimsave('graphics/saggcontez.gif', [np.array(img)[200:315,:-320,:] for i, img in enumerate(images)], fps=29)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))