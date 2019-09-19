import gym
import gym_flowers
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import importlib


def time_test(ntest=20):
    start = time.time()
    for i in range(ntest):
        env.reset()
        for j in range(2001):
            env.step(env.action_space.sample())
    end = time.time()
    #print("{} episodes --> {} secs".format(ntest,end-start))

# leg_size = "long"
# env = gym.make('flowers-Walker-continuous-v0') #4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
# env.env.my_init({'leg_size':leg_size})
#env.seed(564564)

env_kwargs = {'roughness': None,
              'stump_height': None,
              'gap_width': None,
              'step_height': None,
              'step_number': None}
#:[0.1341240555047989, 0.1], width:[0.07659460604190826, 0.1] spacing:0.48806294798851013




# env.env.set_environment(roughness=None, stump_height=[2.0, 0.0], stump_width=[1.5, 0.0], stump_rot=1.7*3.14, obstacle_spacing=0.5,
#                         gap_width=None, step_height=None, tunnel_height=None, step_number=None)
# #time_test()
# for i in range(20):
#     env.reset()
#     env.render()
#     time.sleep(0.1)

# GENERATE TERRAIN IMAGES:
imname="agent_default"
#time_test()
np.random.seed(np.random.randint(0,65454654))

def poly_params_to_image(p):
    env = gym.make('flowers-Walker-movement-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': 'long'})
    env.env.set_environment(poly_shape=p)
    env.reset()
    img = env.render(mode='rgb_array')
    return img


# [172:315,:-320,:] for demo env
def draw_env(v):
    env.env.set_environment(moving_stumps=[4.0])#np.random.uniform(0,2*np.pi,10))
    #env.env.set_environment(poly_shape=v)
    #env.env.set_environment(#poly_shape=np.random.uniform(0,4,20))
    #env.env.set_environment(stump_height=[np.random.uniform(1,1), 0.0])
        # env.env.set_environment(roughness=np.random.uniform(0,10), stump_height=[np.random.uniform(0,6), 0.1],
        #                         stump_width=[np.random.uniform(0,6), 0.1],
        #                         stump_rot= [np.random.uniform(0,2*np.pi),0.1],
        #                         obstacle_spacing=np.random.uniform(0,6),
        #                         gap_width=None, step_height=None, tunnel_height=None, step_number=None)


        # increments = np.array([-0.4,0,-0.4,0.2,-0.2,0.4,0.2,0.4,0.4,0.2,0.4,0.0])
        # init_poly = np.zeros(12)
        # init_poly += 5

def generate_track_examples():
    n_tracks = 10
    #first simple tracks
    env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': "default"})
    # for i in range(n_tracks):
    #     env.env.set_environment(stump_height=[np.random.uniform(0,3), 0.1], obstacle_spacing=np.random.uniform(0,6))
    #     env.reset()
    #     img = env.render(mode='rgb_array')
    #     plt.imsave("stump_track_{}.jpg".format(i), img)

    #complex track
    env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': "long"})
    for i in range(n_tracks):
        env.env.set_environment(poly_shape=np.random.uniform(0,4,20))
        env.reset()
        img = env.render(mode='rgb_array')
        plt.imsave("hexagon_track_{}.jpg".format(i), img)

def generate_walker_pictures():
    # short one
    # env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    # env.env.my_init({'leg_size': "short"})
    # env.env.set_environment(stump_height=[0.8, 0.1], obstacle_spacing=6)
    # env.reset()
    # img = env.render(mode='rgb_array')
    # plt.imsave("demo_small_agent.jpg", img[172:315,:-280,:])

    env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': "default"})
    env.env.set_environment(stump_height=[1.8, 0.1], obstacle_spacing=6)
    env.reset()
    img = env.render(mode='rgb_array')
    plt.imsave("demo_default_agent.jpg", img[172:315,:-280,:])

    # importlib.reload(gym_flowers)
    # env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    # env.env.my_init({'leg_size': "long"})
    # env.env.set_environment(poly_shape=np.random.uniform(0,4,20), obstacle_spacing=2)
    # env.reset()
    # for i in range(10):
    #     env.step(np.random.uniform(-1,1,8))
    #     img = env.render(mode='rgb_array')
    #     plt.imsave("demo_long_agent_{}.jpg".format(i), img[172:315,:-280,:])


def speed_test():
    for k in range(3):
        start = time.time()
        for i in range(10):
            draw_env('lol')
            env.reset()
            for j in range(1000):
                env.step(env.env.action_space.sample())
        end = time.time()
        print(end-start)


#generate_track_examples()
#generate_walker_pictures()

n_tracks = 20
#first simple tracks
env = gym.make('flowers-Walker-movement-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
env.env.my_init({'leg_size': "long"})
#speed_test()
for i in range(n_tracks):
    draw_env('lol')
    env.reset()
    for j in range(20000):
        #env.step(env.env.action_space.sample())
        img = env.render(mode='rgb_array')
    #plt.imsave("stump_seq_{}.jpg".format(i), img)

# v = np.zeros(12)
# inc = np.array([0.1]*12)
# for j in range(100):
#     draw_env(v)
#     v = v + inc
#     env.reset()
#     for i in range(1):
#         img = env.render(mode='rgb_array')
#         #plt.imsave(imname+str(j)+leg_size+".jpg", img)
#         env.step(env.env.action_space.sample())
#     time.sleep(0.2)
#

# for i in range(20):
#     draw_env()
#     env.reset()
#     img = env.render(mode='rgb_array')
#     plt.imsave(imname+str(i)+".jpg", img)
#     time.sleep(0.1)


# done = False
# norm = MaxMinFilter()
# for j in range(1):
#     start = time.time()
#     env.reset()
#     for i in range(2000):
#         o, r, d, _ = env.step(np.random.rand(4))
#         env.render()
#         n_o = norm(o)
#         print("bef {} af: {}".format(len(o),len(n_o)))
#         print("bef {} af: {}".format(o, n_o))
#     end = time.time()
#     #print(end-start)


