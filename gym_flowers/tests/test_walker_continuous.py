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
    env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': 'long'})
    env.env.set_environment(poly_shape=p)
    env.reset()
    img = env.render(mode='rgb_array')
    return img


# [172:315,:-320,:] for demo env
def draw_env(v):
    env.env.set_environment(stump_seq=np.random.uniform(0,6,10))
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



def generate_walker_pictures(agent="quadru_st"):
    if agent is "short":
        # short one
        env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
        env.env.my_init({'leg_size': "short"})
        env.env.set_environment(stump_height=[1.8, 0.0], obstacle_spacing=6)
        env.reset()
        img = env.render(mode='rgb_array')
        plt.imsave("../../graphics/BW_env_graphs/demo_small_agent.jpg", img[172:315,17:-320,:])#img[172:315,:-280,:])
    if agent is "default":
        env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
        env.env.my_init({'leg_size': "default"})
        env.env.set_environment(stump_height=[1.8, 0.0], obstacle_spacing=6)
        env.reset()
        img = env.render(mode='rgb_array')
        plt.imsave("../../graphics/BW_env_graphs/demo_default_agent.jpg", img[172:315,17:-320,:])
    if agent is "quadru_st":
        env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
        env.env.my_init({'leg_size': "long"})
        env.env.set_environment(stump_height=[1.8, 0.0], obstacle_spacing=6)
        env.reset()
        for i in range(10):
            env.step(np.random.uniform(-1,1,8))
            img = env.render(mode='rgb_array')
            plt.imsave("../../graphics/BW_env_graphs/demo_quadru_agent_stump_tracks{}.jpg".format(i), img[172:315,17:-320,:])
    if agent is "quadru_ht":
        env = gym.make(
            'flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
        env.env.my_init({'leg_size': "long"})
        env.env.set_environment(poly_shape=np.random.uniform(0, 4, 20), obstacle_spacing=2)
        env.reset()
        for i in range(10):
            env.step(np.random.uniform(-1, 1, 8))
            img = env.render(mode='rgb_array')
            plt.imsave("../../graphics/BW_env_graphs/demo_quadru_agent_hexagon_tracks{}.jpg".format(i), img[172:315, 47:-280, :])

def generate_hexagon_and_quadru():
    env = gym.make(
        'flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': "long"})
    p = np.round(np.random.uniform(0, 4, 12), 1)
    env.env.set_environment(poly_shape=p, obstacle_spacing=2)
    print(p)
    env.reset()
    for i in range(10):
        env.step(np.random.uniform(-1, 1, 8))
        img = env.render(mode='rgb_array')
        plt.imsave("../../graphics/BW_env_graphs/hexagon_example2_{}.jpg".format(i), img[172:315, 47:-280, :])

def generate_hexagons(nb_hexagons=100):
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

    env = gym.make(
        'flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    env.env.my_init({'leg_size': "long"})

    cpt = 0
    while cpt != 100:
        p = np.round(np.random.uniform(0, 4, 12), 1)
        height_width = poly_2_width_height(p)
        if np.round(height_width[0]) == 1 and np.round(height_width[0]) == 1:
            env.env.set_environment(poly_shape=p, obstacle_spacing=3)
            env.reset()
            env.step(np.random.uniform(0,0, 8))
            img = env.render(mode='rgb_array')
            plt.imsave("../../graphics/BW_env_graphs/hexagons/hexagon_{}.jpg".format(cpt), img[200:315, 350:450, :])
            cpt +=1
    # for i in range(nb_hexagons):
    #     p = np.round(np.random.uniform(0, 4, 12), 1)
    #     env.env.set_environment(poly_shape=p, obstacle_spacing=3)
    #     env.reset()
    #     env.step(np.random.uniform(0,0, 8))
    #     img = env.render(mode='rgb_array')
    #     plt.imsave("../../graphics/BW_env_graphs/hexagons/hexagon_{}.jpg".format(i), img[200:315, 350:450, :])




    # importlib.reload(gym_flowers)
    # env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
    # env.env.my_init({'leg_size': "long"})
    # env.env.set_environment(poly_shape=np.random.uniform(0,4,20), obstacle_spacing=2)
    # env.reset()
    # for i in range(10):
    #     env.step(np.random.uniform(-1,1,8))
    #     img = env.render(mode='rgb_array')
    #     plt.imsave("../../graphics/BW_env_graphs/demo_long_agent_{}.jpg".format(i), img[172:315,:-280,:])


#generate_track_examples()
#generate_walker_pictures()

generate_hexagons()
# n_tracks = 20
# #first simple tracks
# env = gym.make('flowers-Walker-continuous-v0')  # 4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
# env.env.my_init({'leg_size': "long"})
# for i in range(n_tracks):
#     draw_env('lol')
#     env.reset()
#     img = env.render(mode='rgb_array')
#     plt.imsave("../../graphics/BW_env_graphs/stump_seq_{}.jpg".format(i), img)

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