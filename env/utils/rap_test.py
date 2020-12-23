import math, cmath
import random
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

fc = 28 * 1e9
c = 3.0 * 1e8
lamda = c/fc

user = 40
user_antenna = 1

BS = 1
BS_antenna = 10
space = np.array([lamda*5 * i for i in range(BS_antenna)])

select_user = math.floor(BS_antenna/user_antenna)

SNRdB = 30
SNR = 10 ** (SNRdB / 10)

column_name = []
for i in range(1,user+1):
    column_name.append('user{}'.format(i))


def CDUS(channel, user, user_antenna, BS_antenna, select_user):
    channel = channel.T

    def chordal_distance(H_cand, H_sel):

        g_H_cand, _ = np.linalg.qr(np.conjugate(H_cand).T)
        g_H_sel, _ = np.linalg.qr(np.conjugate(H_sel).T)

        return user_antenna - np.trace(np.conjugate(g_H_sel).T @ g_H_cand @ np.conjugate(g_H_cand).T @ g_H_sel)


    U = list(range(user))
    S = []
    H_cand = np.zeros((user_antenna, BS_antenna), dtype=np.complex)

    for k in range(select_user):
        if k == 0:

            s1 = 0
            tmp_s = 0

            for s in range(user):

                for i in range(user_antenna):
                    for j in range(BS_antenna):
                        H_cand[i, j] = channel[user_antenna * s + i, j]

                tmp = np.linalg.norm(H_cand,'fro')

                if tmp > tmp_s:
                    tmp_s = tmp
                    s1 = s
                    S1 = H_cand.copy()

            U.remove(s1)
            S.append(s1)
            H_sel = S1.copy()

        else:
            s1 = 0
            tmp_s = 0
            for s in U:
                for i in range(user_antenna):
                    for j in range(BS_antenna):
                        H_cand[i, j] = channel[user_antenna * s + i, j]

                tmp = np.abs(chordal_distance(H_cand, H_sel)) ** 2

                if tmp > tmp_s:
                    tmp_s = tmp
                    s1 = s
                    S1 = H_cand.copy()

            U.remove(s1)
            S.append(s1)
            H_sel = np.concatenate([H_sel, S1], axis=0)

    # C = get_channel_capacity(channel, S, user, user_antenna, BS_antenna, SNR)
    H_sel = np.conjugate(H_sel).T @ H_sel
    C = np.real(cmath.log(np.linalg.det(np.eye(BS_antenna) + SNR/select_user * H_sel), 2))

    return C


def RAND(channel, user, user_antenna, BS_antenna, select_user):

    H_sel = np.zeros((BS_antenna, select_user), dtype=np.complex)
    U = random.sample(list(range(user)), select_user)

    i = 0
    for s in U:
        for r in range(BS_antenna):
            H_sel[r, i] = channel[r, s]

        i += 1

    # C = get_channel_capacity(channel, U, select_user, user, user_antenna, BS_antenna, SNR)
    H_sel = np.conjugate(H_sel).T @ H_sel
    C = np.real(cmath.log(np.linalg.det(np.eye(BS_antenna) + SNR/select_user * H_sel), 2))

    return C


def ALL(channel, user, user_antenna, BS_antenna, select_user):

    capacity = 0
    for U in itertools.combinations(list(range(user)), select_user):

        H_tmp = np.zeros((BS_antenna, select_user), dtype=np.complex)

        i = 0
        for s in U:
            for r in range(BS_antenna):
                H_tmp[r, i] = channel[r, s]

            i += 1

        # C = get_channel_capacity(channel, U, user, user_antenna, BS_antenna, SNR)
        H_tmp = H_tmp @ np.conjugate(H_tmp).T
        C = np.real(cmath.log(np.linalg.det(np.eye(BS_antenna) + SNR/select_user * H_tmp), 2))

        if capacity < C:
            capacity = C
            A = list(U).copy()

    # print('all ', A)
    return capacity


H = np.zeros((BS_antenna, user), dtype=np.complex)
cor = []
ch = []

for _ in tqdm(range(100)):
    us_r = []
    us_t = []
    for s in range(user):

        # step0) ユーザの位置を設定
        radius_user = np.random.uniform(0, 20)
        theta_user = np.random.uniform(0, np.pi*2)
        us_r.append(radius_user)
        us_t.append(theta_user)

        # step1) 送受信アンテナ間の距離 d の決定
        d = np.random.uniform(0, 200)

        # step2) クラスターの数 N の決定
        N = 3

        # step3) AODとAOA の空間ローブの数 L の決定
        L = 3

        # step4) 各クラスタのサブパス数 Mn の決定
        Mn = 10

        # step5) 各クラスタのサブパスの相対遅延時間 ρm,n を決定
        Bb = 400 * 1e6
        rho_mn = np.zeros((Mn, N), dtype=np.float)
        for m in range(Mn):
            for n in range(N):
                rho_mn[m][n] = (1/Bb * m)

        # step6) 各クラスタの遅延時間 τn を決定
        tau_vec = np.sort(np.random.exponential(83*1e-9, N))
        tau = np.zeros(N)

        for n in range(1, N):
            delta_tau = tau_vec[n] - min(tau_vec)
            tau[n] = tau[n-1] + rho_mn[Mn-1][n-1] + delta_tau + 25 * 1e-9

        # step7) 各クラスタの電力 Pn を決定
        Pn_ = np.exp(-1 * tau / (49.4*1e-9))
        Pn = np.zeros(N)

        for n in range(N):
            tmp = 0
            for k in range(N):
                tmp += Pn_[k]

            Pn[n] = Pn_[n] / tmp


        # step8) 各クラスタのサブパスの電力 am,n を決定
        a_mn_ = np.zeros((Mn, N), dtype=np.float)
        a_mn = np.zeros((Mn, N), dtype=np.float)

        for m in range(Mn):
            for n in range(N):
                a_mn_[m][n] = np.exp(-1 * rho_mn[m][n] / (16.9*1e-9))

        for n in range(N):
            tmp = 0
            for m in range(Mn):
                tmp += a_mn_[m][n]

            for m in range(Mn):
                a_mn[m][n] = a_mn_[m][n] / tmp * Pn[n]

        # step9) 各クラスタのサブパスの位相 ψm,n を決定
        psi_mn = np.random.uniform(0, np.pi*2, (Mn, N))

        # step10) 各クラスタのサブパスの絶対遅延 tm,n を決定
        t0 = d/c
        t_mn = np.zeros((Mn, N), dtype=np.float)

        for m in range(Mn):
            for n in range(N):
                t_mn[m][n] = t0 + tau[n] + rho_mn[m][n]

        # step11) 空間ローブの方位角 θi を決定
        theta_AOD = np.zeros(L, dtype=np.float)
        theta_AOA = np.zeros(L, dtype=np.float)

        for i in range(L):
            theta_min = 360 * i / L
            theta_max = 360 * (i+1) / L

            theta_AOD[i] = np.random.uniform(theta_min, theta_max)
            theta_AOA[i] = np.random.uniform(theta_min, theta_max)

        # step12) 空間ローブの仰角 φi を決定
        phi_AOD = np.random.normal(-4.9, 4.5, L)
        phi_AOA = np.random.normal(3.6, 4.8, L)

        # step13) 各クラスタへのサブパスの AOD: (θm,n,AOD,φm,n,AOD) と AOA : (θm,n,AOA,φm,n,AOA) を決定
        theta_mn_AOD = np.zeros((Mn, L), dtype=np.float)
        phi_mn_AOD = np.zeros((Mn, L), dtype=np.float)

        theta_mn_AOA = np.zeros((Mn, L), dtype=np.float)
        phi_mn_AOA = np.zeros((Mn, L), dtype=np.float)

        for i in range(L):
            for m in range(Mn):

                delta_theta_mn_AOD = np.random.normal(0, 9.0)
                delta_phi_mn_AOD = np.random.normal(0, 2.5)

                delta_theta_mn_AOA = np.random.normal(0, 10.1)
                delta_phi_mn_AOA = np.random.laplace(0, 10.5)

                theta_mn_AOD[m][i] = theta_AOD[i] + delta_theta_mn_AOD
                phi_mn_AOD[m][i] = phi_AOD[i] + delta_phi_mn_AOD

                theta_mn_AOA[m][i] = theta_AOA[i] + delta_theta_mn_AOA
                phi_mn_AOA[m][i] = phi_AOA[i] + delta_phi_mn_AOA

        theta_mn_AOD = np.deg2rad(theta_mn_AOD)
        phi_mn_AOD = np.deg2rad(phi_mn_AOD)

        theta_mn_AOA = np.deg2rad(theta_mn_AOA)
        phi_mn_AOD = np.deg2rad(theta_mn_AOA)

        # step14) チャネル生成
        H_ = np.zeros((BS_antenna, 1), dtype=np.complex)

        for n in range(N):
            for m in range(Mn):
                at = np.exp(1j * 2 * np.pi / lamda * np.cos(theta_mn_AOD[m][n]) * space).T
                ar = np.exp(1j * 2 * np.pi / lamda * (radius_user * np.cos(phi_mn_AOA[m][n]) * np.cos(theta_user) * np.cos(theta_mn_AOA[m][n]) - radius_user * np.cos(phi_mn_AOA[m][n]) * np.sin(theta_user) * np.sin(theta_mn_AOA[m][n])))
                H_[:,0] = a_mn[m][n] * np.exp(1j*psi_mn[m][n]) * at * np.conjugate(ar) + H_[:,0]

        H[:,s] = H_[:,0]

    df_H = pd.DataFrame(H, columns=column_name)


    # print(CDUS(H, user, user_antenna, BS_antenna, select_user))
    # print(RAND(H, user, user_antenna, BS_antenna, select_user))

    U, D, Vh = np.linalg.svd(H, full_matrices=True)
    cor.append(min(D)/max(D))
    ch.append(CDUS(H, user, user_antenna, BS_antenna, select_user) - RAND(H, user, user_antenna, BS_antenna, select_user))

print(np.mean(cor))
print(np.mean(ch))