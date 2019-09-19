import argparse
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.sac.sac import sac
from spinup.algos.sac import core
import gym
from teachers.teacher_controller import EnvParamsSelector

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--seed', '-s', type=int, default=0)

# Deep RL student arguments, so far only works with SAC
parser.add_argument('--hid', type=int, default=-1)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--gpu_id', type=int, default=-1)  # default is no GPU
parser.add_argument('--ent_coef', type=float, default=0.005)
parser.add_argument('--max_ep_len', type=int, default=2000)
parser.add_argument('--steps_per_ep', type=int, default=200000)
parser.add_argument('--buf_size', type=int, default=1000000)
parser.add_argument('--nb_test_episodes', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--train_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1000)

# Parameterized bipedal walker related arguments
parser.add_argument('--env', type=str, default="flowers-Walker-continuous-v0")
parser.add_argument('--env_babbling', type=str, default="none")
parser.add_argument('--max_stump_h', type=float, default=None)
parser.add_argument('--max_stump_w', type=float, default=None)

parser.add_argument('--max_stump_r', type=float, default=None)
parser.add_argument('--roughness', type=float, default=None)

parser.add_argument('--max_obstacle_spacing', type=float, default=None)
parser.add_argument('--max_gap_w', type=float, default=None)
parser.add_argument('--step_h', type=float, default=None)
parser.add_argument('--step_nb', type=float, default=None)
parser.add_argument('--env_param_input', type=int, default=False)
parser.add_argument('--leg_size', type=str, default="default")
parser.add_argument('--poly_shape', '-poly', action='store_true')
parser.add_argument('--stump_seq', '-seq', action='store_true')

# Teacher-specific arguments:

# alp-gmm related arguments
parser.add_argument('--gmm_fitness_fun', '-fit', type=str, default=None)
parser.add_argument('--nb_em_init', type=int, default=None)
parser.add_argument('--min_k', type=int, default=None)
parser.add_argument('--max_k', type=int, default=None)
parser.add_argument('--fit_rate', type=int, default=None)
parser.add_argument('--weighted_gmm', '-wgmm', action='store_true')
parser.add_argument('--multiply_lp', '-lpm', action='store_true')

# covar-gmm related arguments
parser.add_argument('--absolute_lp', '-alp', action='store_true')

# riac related arguments
parser.add_argument('--max_region_size', type=int, default=None)
parser.add_argument('--lp_window_size', type=int, default=None)

args = parser.parse_args()

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


if args.gpu_id != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

ac_kwargs = dict()
if args.hid != -1:
    ac_kwargs['hidden_sizes'] = [args.hid] * args.l

params = {}
if args.env_babbling == 'gmm':
    if args.gmm_fitness_fun is not None:
        params['gmm_fitness_fun'] = args.gmm_fitness_fun
    if args.min_k is not None and args.max_k is not None:
        params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
    if args.weighted_gmm is True:
        params['weighted_gmm'] = args.weighted_gmm
    if args.nb_em_init is not None:
        params['nb_em_init'] = args.nb_em_init
    if args.fit_rate is not None:
        params['fit_rate'] = args.fit_rate
    if args.multiply_lp is True:
        params['multiply_lp'] = args.multiply_lp
elif args.env_babbling == 'bmm':
    if args.absolute_lp is True:
        params['absolute_lp'] = args.absolute_lp
elif args.env_babbling == "riac":
    if args.max_region_size is not None:
        params['max_region_size'] = args.max_region_size
    if args.lp_window_size is not None:
        params['lp_window_size'] = args.lp_window_size

env_kwargs = {'roughness': args.roughness,
              'stump_height': None if args.max_stump_h is None else [0, args.max_stump_h],
              'stump_width': None if args.max_stump_w is None else [0, args.max_stump_w],
              'stump_rot': None if args.max_stump_r is None else [0, args.max_stump_r],
              'obstacle_spacing': None if args.max_obstacle_spacing is None else [0, args.max_obstacle_spacing],
              'poly_shape': None if not args.poly_shape else [0, 4.0],
              'stump_seq': None if not args.stump_seq else [0, 6.0],
              'gap_width': args.max_gap_w,
              'step_height': args.step_h,
              'step_number': args.step_nb,
              'env_param_input': args.env_param_input}
env_f = lambda: gym.make(args.env)
env_init = {}
if args.env == "flowers-Walker-continuous-v0":
    env_init['leg_size'] = args.leg_size

# Initialize teacher
Teacher = EnvParamsSelector(args.env_babbling, args.nb_test_episodes, env_kwargs,
                            seed=args.seed, teacher_params=params)

# Launch Student training
sac(env_f, Teacher, actor_critic=core.mlp_actor_critic,
    ac_kwargs=ac_kwargs,
    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
    logger_kwargs=logger_kwargs, alpha=args.ent_coef, max_ep_len=args.max_ep_len,
    steps_per_epoch=args.steps_per_ep, replay_size=args.buf_size,
    env_babbling=args.env_babbling, env_kwargs=env_kwargs, env_init=env_init,
    env_name=args.env, nb_test_episodes=args.nb_test_episodes, lr=args.lr, train_freq=args.train_freq,
    batch_size=args.batch_size, teacher_params=params)