import math, cmath
import numpy as np

class Rappaport_channel:
    def __init__(self, fc, user, user_antenna, BS, BS_antenna):
        self.c = 3.0 * 1e8
        self.lamda = self.c/fc
        self.user = user
        self.user_antenna = user_antenna
        self.BS = BS
        self.BS_antenna = BS_antenna
        self.space = np.array([self.lamda/2 * i for i in range(self.BS_antenna)])


    def create_channel(self):

        H = np.zeros((self.BS_antenna, self.user), dtype=np.complex)
        for s in range(self.user):

            # step0) ユーザの位置を設定
            radius_user = np.random.uniform(0, 20)
            theta_user = np.random.uniform(0, np.pi*2)

            # step1) 送受信アンテナ間の距離 d の決定
            d = np.random.uniform(60, 80)

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
            t0 = d/self.c
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
            H_ = np.zeros((self.BS_antenna, 1), dtype=np.complex)
            for n in range(N):
                for m in range(Mn):
                    at = np.exp(1j * 2 * np.pi / self.lamda * np.cos(theta_mn_AOD[m][n]) * self.space).T
                    ar = np.exp(1j * 2 * np.pi / self.lamda * (radius_user * np.cos(phi_mn_AOA[m][n]) * np.cos(theta_user) * np.cos(theta_mn_AOA[m][n]) - radius_user * np.cos(phi_mn_AOA[m][n]) * np.sin(theta_user) * np.sin(theta_mn_AOA[m][n])))
        
                    H_[:,0] = a_mn[m][n] * np.exp(1j*psi_mn[m][n]) * at * np.conjugate(ar) + H_[:,0]


            H[:,s] = H_[:,0]

        return H