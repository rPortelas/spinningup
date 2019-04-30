import numpy as np
import tensorflow as tf
import gym
import gym_flowers
import time
from spinup.algos.sac import core
from param_env_utils.env_params_selection import EnvParamsSelector
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from spinup.utils.normalization_utils import MaxMinFilter
import os


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=100000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.005, batch_size=100, start_steps=10000,
        max_ep_len=2000, logger_kwargs=dict(), save_freq=1, env_babbling="none", env_kwargs=dict(),
        norm_obs=False, env_name='unknown', nb_test_episodes=15, train_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Parameterized env init
    env_params = EnvParamsSelector(env_babbling, nb_test_episodes, env_kwargs)

    env, test_env = env_fn(), env_fn()


    if env_babbling is not "none": env_params.set_env_params(env,env_kwargs)
    env.reset()

    obs_dim = env.env.observation_space.shape[0]
    print(obs_dim)
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Obs normalization
    if norm_obs:
        if env_name == 'BipedalWalker-v2' or env_name == 'BipedalWalkerHardcore-v2':
            norm = MaxMinFilter()
        elif env_name == 'flowers-Walker-v2':
            assert env_babbling == 'random'
            norm = MaxMinFilter(env_params_dict=env_kwargs)



    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)


    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            if env_babbling is not "none": env_params.set_test_env_params(test_env, env_kwargs)
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            o = norm(o) if norm_obs else o
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                o = norm(o) if norm_obs else o
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            env_params.record_test_episode(ep_ret, ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o = norm(o) if norm_obs else o
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env

        o2, r, d, _ = env.step(a)
        o2 = norm(o2) if norm_obs else o2
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(np.ceil(ep_len/train_freq).astype('int')):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                             VVals=outs[6], LogPi=outs[7])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            env_params.record_train_episode(ep_ret, ep_len)
            if env_babbling is not "none": env_params.set_env_params(env, env_kwargs)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            #print('not norm {}'.format(o))
            o = norm(o) if norm_obs else o
            #print('norm {}'.format(o))

        # End of epoch wrap-up
        if t > 0 and (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)#itr=epoch)

            # Test the performance of the deterministic version of the agent.
            test_agent(n=nb_test_episodes)
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t+1)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            logger.log_tabular('VVals', with_min_and_max=True) 
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            # Pickle parameterized env data
            #print(logger.output_dir+'/env_params_save.pkl')
            env_params.dump(logger.output_dir+'/env_params_save.pkl')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=-1)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--ent_coef', type=float, default=0.005)
    parser.add_argument('--max_ep_len', type=int, default=2000)
    parser.add_argument('--steps_per_ep', type=int, default=100000)
    parser.add_argument('--buf_size', type=int, default=1000000)
    # Parameterized bipedal walker related arguments
    parser.add_argument('--env_babbling', type=str, default="none")
    #parser.add_argument('--max_stump_h', type=float, default=None)
    parser.add_argument('--max_stump_h', type=float, default=None)
    parser.add_argument('--max_stump_w', type=float, default=None)
    parser.add_argument('--max_tunnel_h', type=float, default=None)
    parser.add_argument('--roughness', type=float, default=None)
    parser.add_argument('--max_obstacle_spacing', type=float, default=None)
    parser.add_argument('--max_gap_w', type=float, default=None)
    parser.add_argument('--step_h', type=float, default=None)
    parser.add_argument('--step_nb', type=float, default=None)
    parser.add_argument('--norm_obs', type=int, default=False)
    parser.add_argument('--env_param_input', type=int, default=False)
    parser.add_argument('--nb_test_episodes', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if args.gpu_id != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    ac_kwargs = dict()
    if args.hid != -1:
        ac_kwargs['hidden_sizes'] = [args.hid]*args.l

    env_kwargs = {'roughness':args.roughness,
                  'stump_height': None if args.max_stump_h is None else [0, args.max_stump_h],
                  'stump_width': None if args.max_stump_w is None else [0, args.max_stump_w],
                  'tunnel_height': None if args.max_tunnel_h is None else [0, args.max_tunnel_h],
                  'obstacle_spacing': None if args.max_obstacle_spacing is None else [0, args.max_obstacle_spacing],
                  'gap_width':args.max_gap_w,
                  'step_height':args.step_h,
                  'step_number':args.step_nb,
                  'env_param_input':args.env_param_input}
    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=ac_kwargs,
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, alpha=args.ent_coef, max_ep_len=args.max_ep_len,
        steps_per_epoch=args.steps_per_ep, replay_size=args.buf_size,
        env_babbling=args.env_babbling, env_kwargs=env_kwargs, norm_obs=args.norm_obs,
        env_name=args.env, nb_test_episodes=args.nb_test_episodes, lr=args.lr, train_freq=args.train_freq,
        batch_size=args.batch_size)