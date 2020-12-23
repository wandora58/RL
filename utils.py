
import os
import csv
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K


class HistoryLogger(object):

    def __init__(self, log_dir):
        self.history_path = os.path.join(log_dir, 'reward_log.csv')
        os.makedirs(log_dir, exist_ok=True)

    def set_history_header(self, log_header):
        with open(self.history_path, mode='w') as ofs:
            writer = csv.writer(ofs)
            writer.writerow(log_header)

    def history_save(self, log_list):
        with open(self.history_path, mode='a') as ofs:
            writer = csv.writer(ofs)
            writer.writerow(log_list)


# Setting arg parser
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./result', help='log directory')
    parser.add_argument('--n_episodes', type=int, default=100000, help='# of episodes for train')

    return parser.parse_args()


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))

    return numerator / denominator


# Calculate the correct distance for the selectged combination
def get_channel_capacity(channel, combinations, batch_size, sel_user, user, user_antenna, BS_antenna, SNR):
    """
    sel_channel = [batch, sel_user, BS_anntena*2]
    sel_channel = tf.transpose(complex_channel, [0, 2, 1]) [batch, BS_anntena*2, sel_user]
         us1  us2  sel_user
    a1_r[   ,    ,   ]
    a1_i[   ,    ,   ]
    a2_r[   ,    ,   ]
    a2_i[   ,    ,   ]

    re_index [batch, BS_antenna, 2]
    [[0,0],  [batch, 0] = batch 目の 0行めを取ってくる
     [0,2],
     [0,4],
     [0,6]]

    re_channel [batch, BS_antenna, sel_user]
         us1  us2  sel_user
    a1_r[   ,    ,   ]
    a2_r[   ,    ,   ]

    im_channel [batch, BS_antenna, sel_user]
         us1  us2  sel_user
    a1_i[   ,    ,   ]
    a2_i[   ,    ,   ]

    complex_channel [batch, BS_antenna, sel_user]
        us1  us2  sel_user
    a1[ r+i, r+i, r+i]
    a2[ r+i, r+i, r+i]



    complex_channel = [batch, BS_anntena, sel_user]
    """

    sel_channel = tf.batch_gather(channel, combinations)
    sel_channel = tf.transpose(sel_channel, [0, 2, 1])

    re_index = np.zeros((batch_size, BS_antenna, 2), dtype=np.int)
    im_index = np.zeros((batch_size, BS_antenna, 2), dtype=np.int)

    for bch in range(batch_size):
        for r in range(BS_antenna):
            re_index[bch][r][0] = bch
            re_index[bch][r][1] = 2*r

            im_index[bch][r][0] = bch
            im_index[bch][r][1] = 2*r+1

    re_channel = tf.gather_nd(sel_channel, re_index)
    im_channel = tf.gather_nd(sel_channel, im_index)

    complex_channel = tf.complex(re_channel, im_channel)
    conj_complex_channel = tf.transpose(tf.conj(complex_channel), [0, 2, 1])
    complex_channel = tf.matmul(complex_channel, conj_complex_channel)

    one = tf.ones([batch_size, BS_antenna])
    zero = tf.zeros([batch_size, BS_antenna, sel_user])
    # I = tf.cast(tf.complex(tf.matrix_diag(one), zero), dtype=tf.complex128)
    I = tf.complex(tf.matrix_diag(one), zero)

    sn_complex_channel = tf.add(I, SNR/sel_user * complex_channel)
    determinant = tf.matrix_determinant(sn_complex_channel)

    return tf.math.real(log2(determinant))


def set_train_plot(ax, df):
    if 'episode' not in df.keys():
        return
    if 'avg_reward' not in df.keys():
        return
    if 'avg_loss' not in df.keys():
        return

    loss_scale_1 = 10
    # loss_scale_2 = 1
    loss_scale_3 = 1
    loss_scale_4 = 10

    x = df['episode']
    y1 = df['avg_reward'] / loss_scale_1
    # y2 = df['avg_loss'] / loss_scale_2
    y3 = df['avg_vloss'] / loss_scale_3
    y4 = df['avg_aloss'] / loss_scale_4

    kwargs = {'alpha': 0.5}
    ax.plot(x, y1, 'r-',
            label='avg_reward/episode (x1/{})'.format(loss_scale_1), **kwargs)
    # ax.plot(x, y2, 'b:',
    #         label='avg_loss/episode (x1/{})'.format(loss_scale_2), **kwargs)
    ax.plot(x, y3, 'g-',
            label='avg_vloss/episode (x1/{})'.format(loss_scale_3), **kwargs)
    ax.plot(x, y4, 'b-',
            label='avg_ploss/episode (x1/{})'.format(loss_scale_4), **kwargs)

    ax.set_ylim(-0.6, 0.5)

    ax.legend(loc='best')

    ax.set_title('training history')
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('amplitude')


def set_route_plot(ax, batch_input, batch_tour, batch_reward, best_flg):

    # plot cities
    ax.scatter(batch_input[0, :, 0], batch_input[0, :, 1], c='r', s=30)

    # plot prd tours
    if not best_flg:

        for i_batch, (input, tour) in enumerate(zip(batch_input, batch_tour)):

            if i_batch % 1:  # 1, 5, 10
                continue

            # plot tours
            X = input[tour, 0]
            Y = input[tour, 1]

            _label = '' if i_batch else 'tours'
            ax.plot(X, Y, 'k-', alpha=0.1, label=_label)

    # plot best route
    if best_flg:

        # best tour
        best_batch = np.argmax(batch_reward)
        best_length = -1.0 * 100. * np.max(batch_reward)

        input = batch_input[best_batch]
        tour = batch_tour[best_batch]

        # plot best tour
        X = input[tour, 0]
        Y = input[tour, 1]

        ax.plot(X, Y, 'r-', alpha=0.8, label='best tour')
        print('best tour length: {:3.3f}'.format(best_length))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.legend(loc='best')

    ax.set_title('predicted route')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')


