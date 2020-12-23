"""
overview:
    Search solutions to traveling salesman problem using Actor-Critic method

args:
    Following elements are specified in this code
        - log_dir: result output
        - n_episodes: num of episodes


output:
    Output following elements to above log_dir
        - checkpoint: checkpoint file
        - model.xxx-xxx.ckpt: Model file for each checkpoint
        - reward_log.csv: Log of monitoring indicators such as reward and loss

usage-example:
    python train.py --log_dir=./result --n_episodes=60000
"""

import os
import time
import math

import numpy as np
import tensorflow as tf

from agent.actor_critic_agent import ActorCriticAgent
from env.mus import MUSEnv
from utils import get_args, HistoryLogger

def train():

    # -------- PRE-PROCESS -------

    # Setup
    train_flg = True  # Whether to perform reinforcement learning

    fc = 28 * 1e9 # center frequency

    user = 4        # num of user
    BS = 1           # num of BS
    user_antenna = 1 # num of anntena per user
    BS_antenna = 2   # num of anntena per BS

    sel_user = math.floor(BS_antenna/user_antenna) # num of selected users

    SNRdB = 30 # SNR(dB)

    batch_size = 4               # batch size
    feature_dim = BS_antenna * 2  # dimensions of feature value

    # Get args
    args = get_args()
    log_dir = args.log_dir
    n_episodes = args.n_episodes

    # start tf.session
    sess = tf.Session()

    # create instance
    env = MUSEnv(train_flg, batch_size, fc, user, user_antenna, BS, BS_antenna, sel_user, SNRdB)
    agent = ActorCriticAgent(batch_size=batch_size, user=user, sel_user=sel_user, feature_dim=feature_dim, user_antenna=user_antenna, BS_antenna=BS_antenna, SNRdB=SNRdB)
    logger = HistoryLogger(log_dir)

    # initiallize network variables
    _init_g = tf.global_variables_initializer()
    sess.run(_init_g)

    # define history logging header
    _header = ['episode', 'avg_reward', 'avg_loss', 'avg_vloss', 'avg_aloss']
    logger.set_history_header(_header)


    # -------- TRAIN MAIN ---------

    # monotoring metrics
    min_metric = 0.0
    l, c_l, a_l, pred_l, cdus_l, rand_l= [], [], [], [], [], []
    start_time = time.time()

    # start train
    for i_episode in range(n_episodes):

        state, cdus_capacity, rand_capacity = env.reset()

        # update model（= agent.predict、env.step、agent.update）
        # First, generate sample sequence and record likelihood of each sample each time to the end,
        # Second, evaluate and update entire network using the recorded likelihood
        losses, reward, model_prds = agent.update_model(sess, state)
        loss, critic_loss, actor_loss = losses
        log_prob, combi, state_value = model_prds

        l.append(np.mean(loss))
        c_l.append(np.mean(critic_loss))
        a_l.append(np.mean(actor_loss))
        pred_l.append(np.mean(reward))
        cdus_l.append(np.mean(cdus_capacity))
        rand_l.append(np.mean(rand_capacity))
        # all_l.append(np.mean(all_capacity))

        # record results every 50 episodes
        if not i_episode % 100:

            # calculate monitoring metrics
            duration = time.time() - start_time
            avg_loss = np.mean(l)
            avg_crtic_loss = np.mean(c_l)
            avg_actor_loss = np.mean(a_l)
            avg_reward = np.mean(pred_l)
            avg_cdus = np.mean(cdus_l)
            avg_rand = np.mean(rand_l)
            # avg_all = np.mean(all_l)

            # reset monitoring
            l, c_l, a_l, pred_l, cdus_l, rand_l, all_l= [], [], [], [], [], [], []
            start_time = time.time()

            # print
            log_str = 'Episode: {0:6d}/{1:6d}'.format(i_episode, n_episodes)
            log_str += ' - Time: {0:3.2f}'.format(duration)
            log_str += ' - Avg_Reward: {0:3.3f}'.format(avg_reward)
            log_str += ' - Avg_CDUS: {0:3.3f}'.format(avg_cdus)
            log_str += ' - Avg_RAND: {0:3.3f}'.format(avg_rand)
            # log_str += ' - Avg_ALL: {0:3.3f}'.format(avg_all)
            log_str += ' - Avg_Loss: {0:3.5f}'.format(avg_loss)
            log_str += ' - Avg_Critic_Loss: {0:3.5f}'.format(avg_crtic_loss)
            log_str += ' - Avg_Actor_Loss: {0:3.5f}'.format(avg_actor_loss)
            print(log_str)

            # log model
            if not min_metric:
                min_metric = avg_reward
            min_metric = max(min_metric, avg_reward)

            if min_metric is avg_reward:
                args = [i_episode, avg_reward, avg_loss]
                agent.save_graph(sess, log_dir, args)

            # log monitoring metrics
            # log_list = [i_episode, avg_reward, avg_cdus, avg_rand, avg_all, avg_loss, avg_crtic_loss, avg_actor_loss]
            log_list = [i_episode, avg_reward, avg_cdus, avg_rand, avg_loss, avg_crtic_loss, avg_actor_loss]
            logger.history_save(log_list)


if __name__ == '__main__':
    train()
