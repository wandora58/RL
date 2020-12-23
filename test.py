"""
overview:
    Predict solutions to traveling salesman problem using trained model

args:
    Following elements are specified in this code
        - log_dir: output of following elements
        - model_path: trained model file
        - n_episodes: number of predicted episodes

output:
    Following elements to the above log_dir
        - list_results.pkl: list of predict results

usage-example:
    python3 train.py --log_dir=./result \
    --model_path=./results/model.099800-0.394-2.65370.ckpt \
    --n_episodes=5000
"""
import os
import argparse
import time
import pickle
import csv
import math

import numpy as np
import tensorflow as tf
import pandas as pd

from agent.actor_critic_agent import ActorCriticAgent
from env.mus import MUSEnv


class ObjectLogger(object):

    def __init__(self, log_dir):
        self.object_path = os.path.join(log_dir, 'test_results.csv')

    def object_save(self, avg_result):
        acg_result.to_csv(self.pbject_path)


def get_args():

    # Setting arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./result',
                        help='log directory')
    parser.add_argument('--model_path', type=str, default='./result/model.019400-28.867--0.33320.ckpt',
                        help='path to model ckpt file')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='# of episodes for test')

    return parser.parse_args()



def test():

    # ------- PRE-PROCESS --------

    # Setup
    train_flg = False  # Whether to perform reinforcement learning

    user = 20        # num of user
    BS = 1           # num of BS
    user_antenna = 1 # num of anntena per user
    BS_antenna = 10   # num of anntena per BS

    sel_user = math.floor(BS_antenna/user_antenna) # num of selected users

    SNRdB = 10 # SNR(dB)

    batch_size = 10               # batch size
    feature_dim = BS_antenna * 2  # dimensions of feature value

    # Get args
    args = get_args()
    log_dir = args.log_dir
    model_path = args.model_path
    n_episodes = args.n_episodes
    os.makedirs(args.log_dir, exist_ok=True)

    # start tf session
    sess = tf.Session()

    # create instance
    env = MUSEnv(train_flg, batch_size, user, user_antenna, BS, BS_antenna, sel_user, SNRdB)
    agent = ActorCriticAgent(batch_size=batch_size, user=user, sel_user=sel_user, feature_dim=feature_dim, user_antenna=user_antenna, BS_antenna=BS_antenna, SNRdB=SNRdB)
    logger = ObjectLogger(log_dir)

    # initiallize network variables
    _init_g = tf.global_variables_initializer()
    sess.run(_init_g)

    # Restore network variables from trained model
    agent.restore_graph(sess, model_path)


    # -------- TEST MAIN ----------

    # monitoring metrics
    # l, c_l, a_l, pred_l, cdus_l, rand_l, pred_t, cdus_t = [], [], [], [], [], [], [], []
    # avg_result = pd.DataFrame( columns=['avg_loss', 'avg_crtic_loss', 'avg_actor_loss', 'avg_reward', 'avg_cdus', 'avg_rand', 'avg_pred_time', 'avg_cdus_time'] )

    pred_t, cdus_t = [], []

    # start test
    for i_episode in range(n_episodes):

        state, cdus_capacity, rand_capacity, cdus_time = env.reset()

        # predict loss (= agent.predict、env.step）
        start = time.time()
        combi = agent.predict_loss(sess, state)
        pred_time = time.time() - start

        # loss, critic_loss, actor_loss = losses
        # log_prob, combi, state_value = model_prds

        # l.append(np.mean(loss))
        # c_l.append(np.mean(critic_loss))
        # a_l.append(np.mean(actor_loss))
        # pred_l.append(np.mean(reward))
        # cdus_l.append(np.mean(cdus_capacity))
        # rand_l.append(np.mean(rand_capacity))
        pred_t.append(pred_time)
        cdus_t.append(cdus_time)


    # avg_loss = np.mean(l)
    # avg_crtic_loss = np.mean(c_l)
    # avg_actor_loss = np.mean(a_l)
    # avg_reward = np.mean(pred_l)
    # avg_cdus = np.mean(cdus_l)
    # avg_rand = np.mean(rand_l)
    avg_pred_time = np.mean(pred_t)
    avg_cdus_time = np.mean(cdus_t)


    # print
    log_str = 'Episode: {0:6d}/{1:6d}'.format(i_episode, n_episodes)
    # log_str += ' - Avg_Reward: {0:3.3f}'.format(avg_reward)
    # log_str += ' - Avg_CDUS: {0:3.3f}'.format(avg_cdus)
    # log_str += ' - Avg_RAND: {0:3.3f}'.format(avg_rand)
    # log_str += ' - Avg_Loss: {0:3.5f}'.format(avg_loss)
    # log_str += ' - Avg_Critic_Loss: {0:3.5f}'.format(avg_crtic_loss)
    # log_str += ' - Avg_Actor_Loss: {0:3.5f}'.format(avg_actor_loss)
    log_str += ' - Avg_Pred_Time: {0:3.5f}'.format(avg_pred_time)
    log_str += ' - Avg_CDUS_Time: {0:3.5f}'.format(avg_cdus_time)
    print(log_str)

    # save
    # tmp_avg = pd.Series( [ avg_loss, avg_crtic_loss, avg_actor_loss, avg_reward, avg_cdus, avg_rand ], index=avg_result.columns)
    # avg_result = avg_result.append( tmp_avg, ignore_index=True )


    # ------ POST-PROCESS (TESTING) ------

    # Output of prediction result list object
    # logger.object_save(avg_result)

def capacity_test():
    fc = 28 * 1e9
    user = 10
    user_antenna = 1
    BS = 1
    BS_antenna = 5
    sel_user = 5

    SNRdB = 30
    SNR = 10 * np.log10(SNRdB)

    batch_size = 4

    c = Rappaport_channel(fc, user, user_antenna, BS, BS_antenna)
    batch = []
    combi = []
    for bch in range(batch_size):

        channel = c.create_channel()
        H_sel = np.zeros((BS_antenna, sel_user), dtype=np.complex)
        U = random.sample(list(range(user)), sel_user)

        i = 0
        for s in U:
            for r in range(BS_antenna):
                H_sel[r, i] = channel[r, s]

            i += 1

        H_sel = np.conjugate(H_sel).T @ H_sel
        C = np.real(cmath.log(np.linalg.det(np.eye(BS_antenna) + SNR/sel_user * H_sel), 2))
        print(C)

        H = np.zeros((BS_antenna*2, user_antenna), dtype=np.float)
        H_tmp = np.zeros((BS_antenna*2, user_antenna), dtype=np.float)

        for s in range(user):
            if s == 0:
                for r in range(BS_antenna):
                    H[2*r, 0] = np.real(channel[r,0])
                    H[2*r+1, 0] = np.imag(channel[r,0])

            else:
                for r in range(BS_antenna):
                    H_tmp[2*r, 0] = np.real(channel[r,s])
                    H_tmp[2*r+1, 0] = np.imag(channel[r,s])

                H = np.concatenate([H, H_tmp], axis=1)

        H = H.T
        batch.append(H)
        combi.append(U)
    batch = np.array(batch)
    combi = np.array(combi)
    session = tf.Session()

    print(combi.shape)
    print(session.run(get_channel_capacity(batch, combi, batch_size, sel_user, user, user_antenna, BS_antenna, SNR)))



if __name__ == '__main__':
    test()
    capacity_test()

