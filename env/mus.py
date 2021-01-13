
import gym
import random
import itertools
import numpy as np
import cmath, math
import time

from env.utils.channel import Channel
from env.utils.rap_channel import Rappaport_channel


class MUSEnv(gym.Env):

    def __init__(self, train_flg, batch_size=4, fc=28*1e9, user=6, user_antenna=1, BS=1, BS_antenna=4, sel_user=4, SNRdB=0):
        self.train_flg = train_flg

        # setup consts
        self.batch_size = batch_size
        self.fc = fc

        self.user = user
        self.user_antenna = user_antenna

        self.BS = BS
        self.BS_antenna = BS_antenna

        self.sel_user = sel_user

        self.SNR = 10 ** (SNRdB / 10)

        # create instance
        # self.channel = Channel(user=user, user_antenna=user_antenna, BS_antenna=BS_antenna, Nc=72, path=1, CP=1, Ps=1.0, code_symbol=72)
        self.channel = Rappaport_channel(fc=fc, user=user, user_antenna=user_antenna, BS=BS, BS_antenna=BS_antenna)


    def reset(self):
        """
        reset function
            NOTE: different format depending on train_flg

        Outputs:
            self.state: 3D ndarray [batch, user, BS_anntena*2]
                channel matrix

        """

        if self.train_flg:

            channel_data = []
            batch_data = []
            cdus_capacity = []
            rand_capacity = []
            for _ in range(self.batch_size):
                channel, data, cdus, rand = self._generate_data()

                channel_data.append(channel)
                batch_data.append(data)
                cdus_capacity.append(cdus)
                rand_capacity.append(rand)

            self.H = np.array(channel_data)
            self.state = np.array(batch_data)

            return self.H, self.state, cdus_capacity, rand_capacity


        else:
            batch_data = []

            channel, data, cdus_capacity, rand_capacity, duration = self._generate_data()
            channel_data = np.tile(channel, (self.batch_size, 1, 1))
            batch_data = np.tile(data, (self.batch_size, 1, 1))

            self.H = np.array(channel_data)
            self.state = np.array(batch_data)

            return self.H, self.state, cdus_capacity, rand_capacity, duration


    def _generate_data(self):

        """
        generate channel function

        Outputs:
            data: 2D ndarray [user, BS_anntena*2]
                channel matrix

        """

        channel = self.channel.create_channel()

        if self.train_flg:
            cdus = self.CDUS(channel)
        else:
            cdus, duration = self.CDUS(channel)

        rand = self.RAND(channel)

        H = np.zeros((self.BS_antenna*2, self.user_antenna), dtype=np.float)
        H_tmp = np.zeros((self.BS_antenna*2, self.user_antenna), dtype=np.float)

        for s in range(self.user):
            if s == 0:
                for r in range(self.BS_antenna):
                    H[2*r, 0] = np.real(channel[r,0])
                    H[2*r+1, 0] = np.imag(channel[r,0])

            else:
                for r in range(self.BS_antenna):
                    H_tmp[2*r, 0] = np.real(channel[r,s])
                    H_tmp[2*r+1, 0] = np.imag(channel[r,s])

                H = np.concatenate([H, H_tmp], axis=1)

        H = H.T

        channel = np.conjugate(channel.T) @ channel

        gH = np.zeros((self.user*2, self.user_antenna), dtype=np.float)
        gH_tmp = np.zeros((self.user*2, self.user_antenna), dtype=np.float)

        for s in range(self.user):
            if s == 0:
                for r in range(self.user):
                    gH[2*r, 0] = np.real(channel[r,0])
                    gH[2*r+1, 0] = np.imag(channel[r,0])

            else:
                for r in range(self.user):
                    gH_tmp[2*r, 0] = np.real(channel[r,s])
                    gH_tmp[2*r+1, 0] = np.imag(channel[r,s])

                gH = np.concatenate([gH, gH_tmp], axis=1)

        gH = gH.T

        if self.train_flg:
            return H, gH, cdus, rand
        else:
            return H, gH, cdus, rand, duration


    def CDUS(self, channel):

        def chordal_distance(H_cand, H_sel):

            g_H_cand, _ = np.linalg.qr(np.conjugate(H_cand).T)
            g_H_sel, _ = np.linalg.qr(np.conjugate(H_sel).T)

            return self.user_antenna - np.trace(np.conjugate(g_H_sel).T @ g_H_cand @ np.conjugate(g_H_cand).T @ g_H_sel)

        start = time.time()
        channel = channel.T

        U = list(range(self.user))
        S = []
        H_cand = np.zeros((self.user_antenna, self.BS_antenna), dtype=np.complex)

        for k in range(self.sel_user):
            if k == 0:

                s1 = 0
                tmp_s = 0

                for s in range(self.user):

                    for i in range(self.user_antenna):
                        for j in range(self.BS_antenna):
                            H_cand[i, j] = channel[self.user_antenna * s + i, j]

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
                    for i in range(self.user_antenna):
                        for j in range(self.BS_antenna):
                            H_cand[i, j] = channel[self.user_antenna * s + i, j]

                    tmp = np.abs(chordal_distance(H_cand, H_sel)) ** 2

                    if tmp > tmp_s:
                        tmp_s = tmp
                        s1 = s
                        S1 = H_cand.copy()

                U.remove(s1)
                S.append(s1)
                H_sel = np.concatenate([H_sel, S1], axis=0)

        # C = get_channel_capacity(channel, S, user, user_antenna, BS_antenna, SNR)
        duration = time.time() - start
        H_sel = np.conjugate(H_sel).T @ H_sel
        C = np.real(cmath.log(np.linalg.det(np.eye(self.BS_antenna) + self.SNR/self.sel_user * H_sel), 2))

        if self.train_flg:
            return C
        else:
            return C, duration


    def RAND(self, channel):

        H_sel = np.zeros((self.BS_antenna, self.sel_user), dtype=np.complex)
        U = random.sample(list(range(self.user)), self.sel_user)

        i = 0
        for s in U:
            for r in range(self.BS_antenna):
                H_sel[r, i] = channel[r, s]

            i += 1

        # C = get_channel_capacity(channel, U, sel_user, user, user_antenna, BS_antenna, SNR)
        H_sel = np.conjugate(H_sel).T @ H_sel
        C = np.real(cmath.log(np.linalg.det(np.eye(self.BS_antenna) + self.SNR/self.sel_user * H_sel), 2))

        return C


    def ALL(self, channel):

        capacity = 0
        for U in itertools.combinations(list(range(self.user)), self.sel_user):

            H_cand = np.zeros((self.BS_antenna, self.user_antenna), dtype=np.complex)
            H_tmp = np.zeros((self.BS_antenna, self.user_antenna), dtype=np.complex)

            cnt = 0
            for s in U:
                if cnt == 0:
                    for i in range(self.BS_antenna):
                        for j in range(self.user_antenna):
                            H_tmp[i, j] = channel[0, i*self.user*self.user_antenna + s*self.user_antenna + j]

                else:
                    for i in range(self.BS_antenna):
                        for j in range(self.user_antenna):
                            H_cand[i, j] = channel[0, i*self.user*self.user_antenna + s*self.user_antenna + j]

                    H_tmp = np.concatenate([H_tmp, H_cand], axis=1)

                cnt += 1

            H_tmp = H_tmp @ np.conjugate(H_tmp).T
            C = np.real(cmath.log(np.linalg.det(np.eye(self.BS_antenna) + self.SNR/self.sel_user * H_tmp), 2))

            if capacity < C:
                capacity = C

        return capacity


    def step(self, action, sel_user):
        """

        Advance state

        Args:
            action: 2D list [batch, sel_user]
                combination prediction as action
        """

        self.sel_user = sel_user

        self.state = self._advance_next_state(self.state, action)
        reward = self._get_channel_capacity(self.state)

        return self.state, reward, True, {}


    def _advance_next_state(self, channel, combination):
        return np.array([channel[i, combination[i], :] for i in range(self.sel_user)])


    def _get_channel_capacity(self, channel):

        C = np.real(cmath.log(np.linalg.det(np.eye(N) + SNR/M * channel), 2))

        capacity = []
        for bch in range(self.batch_size):

            H = np.zeros((self.BS_antenna, self.sel_user), dtype=np.complex)

            for i in range(self.BS_antenna):
                for j in range(self.sel_user):
                    H[j, i] = channel[bch, i, 2*j] + 1j*channel[bch, i, 2*j+1]

            Nt = self.sel_user * self.user_antenna
            Nr = self.BS_antenna
            INr = (Nt / self.SNR) * np.eye(Nr)
            W_MMSE = np.linalg.pinv(np.conjugate(H) @ H.T + INr)

            C_MMSE = 0

            for i in range(Nt):
                h = H[:,i]
                w = W_MMSE @ np.conjugate(h)

                C_MMSE += np.log2(1 + np.abs(w.T @ h) ** 2 / (np.conjugate(w.T) @ np.conjugate(h) - np.abs(w.T @ h) ** 2))

            capacity.append(np.real(C_MMSE))

        return capacity


def main():


    sel_user = 4
    env = MUSEnv(train_flg=True, sel_user=sel_user)

    state = env.reset()

    _size = state.shape[0]
    _user = state.shape[1]

    combi = np.array([random.sample(list(range(_user)), sel_user) for i in range(_size)])

    next_state, reward, _, _ = env.step(combi, sel_user)
    print(reward)


if __name__ == "__main__":
    main()

