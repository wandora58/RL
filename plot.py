

import csv
import os
import pathlib

import itertools
import numpy as np
import pandas as pd
import math
import cmath
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

file = "result/reward_log.csv"
df = pd.read_csv(file, index_col=0)
channel_df = df[['avg_reward', 'avg_cdus', 'avg_rand']]
loss_df = df[['avg_vloss','avg_aloss']]
loss_df['avg_aloss'] = loss_df['avg_aloss'] /10
loss_df = loss_df.rename(columns={'avg_vloss': 'avg_critic_loss', 'avg_aloss': 'avg_actor_loss'})

#figure　
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#plot
channel_df.plot()

#y軸の範囲設定
plt.ylim([27,29])
plt.yticks([27,27.5,28,28.5,29])

#ひげ消す
plt.gca().xaxis.set_tick_params(direction='in')
plt.gca().yaxis.set_tick_params(direction='in')

#x軸,y軸のラベル付け
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Channel capacity[bps/Hz]', fontsize=12)

#グリッド表示
plt.grid(which="both")

#凡例とタイトル
plt.legend(loc=(0.65, 0.55), prop={'size':12})

#保存　
plt.savefig('result/channel_log.pdf')


#figure　
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#plot
loss_df.plot()

#y軸の範囲設定
plt.ylim([-2,2])
plt.yticks([-2,-1,0,1,2])

#ひげ消す
plt.gca().xaxis.set_tick_params(direction='in')
plt.gca().yaxis.set_tick_params(direction='in')

#x軸,y軸のラベル付け
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)

#グリッド表示
plt.grid(which="both")

#凡例とタイトル
plt.legend(loc='best',prop={'size':12})

#保存　
plt.savefig('result/loss_log.pdf')


