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
#from matplotlib import pyplot as plt
import pickle
import copy

empty_arg_ranges = {'roughness':None,
              'stump_height':None,#[0,4.0],#stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
              'stump_width':None,
              'tunnel_height':None,
              'obstacle_spacing':None,#[0,6.0],
              'poly_shape':None,
              'gap_width':None,
              'step_height':None,
              'step_number':None,
              'stump_seq':None}


def params_2_env_list(param_list,key):
    env_list = []
    for p in param_list:
        env_arg = copy.copy(empty_arg_ranges)
        if len(key) == 1:
            print(p)
            env_arg[key] = p
        else:
            for i, k in enumerate(key):
                # print(i)
                # print(p)
                # print(env_arg)
                env_arg[k] = p[i]
        env_list.append(env_arg)
    return env_list



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
        # if kwargs['stump_height'] is not None:
        #     random_stump_h = get_mu_sigma(kwargs['stump_height'][0], kwargs['stump_height'][1])
        #     random_stump_h[1] = 0.1
        if 'poly_shape' not in kwargs.keys():
            kwargs['poly_shape'] = None
        random_stump_h = None
        random_tunnel_h = None
        random_stump_r = None
        random_stump_w = None
        random_ob_spacing = None
        random_stump_seq = None
        if kwargs['stump_height'] is not None:
            random_stump_h = [kwargs['stump_height'], 0.1]
        if 'stump_rot' in kwargs.keys() and kwargs['stump_rot'] is not None:
            random_stump_r = [kwargs['stump_rot'], 0.1]
        if kwargs['stump_width'] is not None:
            random_stump_w = [kwargs['stump_width'], 0.1]
        if kwargs['tunnel_height'] is not None:
            random_tunnel_h = [kwargs['tunnel_height'], 0.1]
        if kwargs['obstacle_spacing'] is not None:
            random_ob_spacing = kwargs['obstacle_spacing']
        if kwargs['stump_seq'] is not None:
            random_stump_seq = kwargs['stump_seq']
        env.env.set_environment(roughness=kwargs['roughness'], stump_height=random_stump_h,
                                stump_width=random_stump_w, stump_rot=random_stump_r,
                                tunnel_height=None, obstacle_spacing=random_ob_spacing,
                                gap_width=kwargs['gap_width'], step_height=kwargs['step_height'],
                                step_number=kwargs['step_number'], poly_shape=kwargs['poly_shape'], stump_seq=random_stump_seq)

    def poly_2_width_height(params):
            scaling = 14 / 30.0
            obstacle_polygon = [(-0.5, 0), (-0.5, 0.25), (-0.25, 0.5), (0.25, 0.5), (0.5, 0.25), (0.5, 0)]
            paired_params = [[params[i], params[i + 1]] for i in range(0, len(params), 2)]
            # first recover polygon coordinate
            poly_coord = []
            for i, (b, d) in enumerate(zip(obstacle_polygon, paired_params)):
                # print(paired_params)
                if i != 0 and i != (len(obstacle_polygon) - 1):
                    poly_coord.append([(b[0] * scaling) + (d[0] * scaling),
                                       (b[1] * scaling) + (d[1] * scaling)])
                else:
                    poly_coord.append([(b[0] * scaling) + (d[0] * scaling),
                                       (b[1] * scaling)])
            # the find maximal width and height
            poly_coord = np.array(poly_coord)
            min_x = np.min(poly_coord[:, 0])
            max_x = np.max(poly_coord[:, 0])
            min_y = np.min(poly_coord[:, 1])
            max_y = np.max(poly_coord[:, 1])
            height_width_params = [(max_x - min_x) / scaling, (max_y - min_y) / scaling]
            return np.round(height_width_params,2)

    # simple exp: random short fails compared to gmm -> [0.84,5.39] run 11

    env_kwargs = {'roughness':None,
                  'stump_height':[0.50,0.50],#stump_levels = [[0., 0.66], [0.66, 1.33], [1.33, 2.]]
                  'tunnel_height':None,
                  'stump_rot':None,
                  'stump_width':None,
                  'obstacle_spacing':4,
                  'gap_width':None,
                  'step_height':None,
                  'step_number':None}

    #test_env_list = pickle.load(open("/home/remy/projects/spinningup/teachers/test_sets/poly_shape0_4.0.pkl", "rb"))
    test_env_list = pickle.load(open("/home/remy/projects/spinningup/teachers/test_sets/stump_height0_3.0obstacle_spacing0_6.0.pkl", "rb"))
    test_env_list = pickle.load(
        open("/home/remy/projects/spinningup/teachers/test_sets/stump_seq0_6.0.pkl",
             "rb"))
    #test_env_list = params_2_env_list([[0.4,0.8]],['stump_height','obstacle_spacing']) #short agent seed 7(or 11)
    #test_env_list = params_2_env_list([[0,0],[0.7,1.0],[1.6,5.5],[1.9,0.01]],['stump_height', 'obstacle_spacing']) # default agent seed 0
    #test_env_list = params_2_env_list([[0,0],[3.0,0.0],[3.0,5], [1.5,0.5]],['stump_height', 'obstacle_spacing']) # long agent seed 0

    #test_env_list = params_2_env_list([[5.0,1.0,5.0,1.0,5.0,1.0,5.0,1.0,5.0,1.0]],'stump_seq') # long agent seed 0

    # final_list = []
    # for i in [19]:
    #     final_list.append(test_env_list[i])
    # for i in range(5):
    #     prev_args = copy.copy(final_list[-1])
    #     last_poly = prev_args['poly_shape']
    #     prev_args['poly_shape'] = np.clip(np.random.normal(last_poly,0.5),0,10)
    #     final_list.append(prev_args)
    # test_env_list = final_list
    # #print(test_env_list)


    if norm_obs:
        norm = MaxMinFilter(env_params_dict=env_kwargs)

    # increments = np.array([-0.4, 0, -0.4, 0.2, -0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4, 0.0])
    # init_poly = np.zeros(12)
    # init_poly += 5
    for i,args in enumerate(test_env_list):

        #args = params_2_env_list([init_poly],'poly_shape')[0]
        # if i not in [0,1,3,6,4]:
        #     continue
        #if i not in [1,5,8,10,25,35]:
        #    continue
        #print("{}: {}".format(i, args['poly_shape']))
        set_test_env_params(**args)
        #init_poly += increments
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        img = env.render(mode='rgb_array')
        o = norm(o) if norm_obs else o
        obss = [o]
        skip = 2
        cpt = 0
        #wh = poly_2_width_height(args['poly_shape'])

        save_img = False
        images = []
        while n < num_episodes:
            if render:
                cpt+=1
                if (cpt%skip) == 0:
                    if make_gif:
                        img = env.render(mode='rgb_array')
                        images.append(img)

                        if save_img:
                            plt.imsave("graphics/walker_images/a_quadru_complex_walker_gmm_{}_{}_{}.png".format(wh, i, cpt), np.array(img)[150:315,:-320,:])
                    else:
                        env.render()
                time.sleep(1e-3)

            a = get_action(o)
            o, r, d, _ = env.step(a)
            o = norm(o) if norm_obs else o
            obss.append(o)
            ep_ret += r
            ep_len += 1

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                #print('Episode {}:{} \t EpRet {} \t EpLen {}'.format(i, wh, ep_ret, ep_len))
                print('Episode {}:{} \t EpRet {} \t EpLen {}'.format(i, args['stump_height'], ep_ret, ep_len))
                #set_test_env_params(**env_kwargs)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                o = norm(o) if norm_obs else o
                n += 1
                #print("MAX:{}".format(np.max(obss, axis=0)))
                #print("MIN:{}".format(np.min(obss,axis=0)))

        #
        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.dump_tabular()
        # print(len(images))
        # print(np.array(images[0]).shape)
    #[150:315,:-320,:] for long
    #[200:315,:-320,:] for default
        imageio.mimsave('graphics/demo_quadru_seq_env_{}.gif'.format(i), [np.array(img)[110:315,:-320,:] for i, img in enumerate(images)], fps=29)
        #imageio.mimsave('graphics/stump_gmm_demo_compact_{}.gif'.format(i), [np.array(img)[150:315,:-320,:] for i, img in enumerate(images)], fps=29)
        #imageio.mimsave('graphics/demo_short_stump_gmm_asquad_{}.gif'.format(i), [np.array(img)[150:315,:-320,:] for i, img in enumerate(images)], fps=29)
        #imageio.mimsave('graphics/demo_default_stump_gmm_asquad_{}.gif'.format(i), [np.array(img)[150:315,:-320,:] for i, img in enumerate(images)], fps=29)
        #imageio.mimsave('graphics/demo_quadru_stump_gmm_compact_{}.gif'.format(i), [np.array(img)[150:315,:-320,:] for i, img in enumerate(images)], fps=29)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--leg_size', default='none')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)

    env.env.my_init({'leg_size':args.leg_size})

    run_policy(env, get_action, args.len, args.episodes, not(args.norender))