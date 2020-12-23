
import random
import itertools

import numpy as np
from scipy import linalg

from Model.utils import MMSE_channel_capacity


class Selection:
    def __init__(self, user, user_antenna, BS, BS_antenna, select_user):
        self.user = user
        self.user_antenna = user_antenna
        self.BS = BS
        self.BS_antenna = BS_antenna
        self.select_user = select_user


    def CDUS(self, channel):
        """

        CDUS user selection

        Args:
            channel: 2D ndarray [Nc, user * user_antenna * BS_antenna]

                                          BS_a1                          /                         BS_a2
                     US1_a1   US1_a2    ...  /   US2_a1   US2_a2    ...  /   US1_a1   US1_a2    ...  /   US2_a1   US2_a2   ...
            path1 [[    h11      h12    h13         h21      h22    h23         h11      h12    h13         h21      h22   h23 ],
            path2  [    h11      h12    h13         h21      h22    h23         h11      h12    h13         h21      h22   h23 ],
            path3  [    h11      h12    h13         h21      h22    h23         h11      h12    h13         h21      h22   h23 ],


        Returns:
            H_sel: 2D ndarray [BS_antenna, select_user * user_antenna]

                     seU1_a1   seU1_a2    ...   /  seU2_a1   seU2_a2    ...
            BS_a1 [[     h11       h12    h13          h21       h22    h23 ],
            BS_a2  [     h11       h12    h13          h21       h22    h23 ],
            BS_a3  [     h11       h12    h13          h21       h22    h23 ],
                                               ...                          ]]

        """

        def chordal_distance(H_cand, H_sel):

            H_cand, _ = np.linalg.qr(H_cand)
            H_sel, _ = np.linalg.qr(H_sel)

            return self.user_antenna - np.trace(np.conjugate(H_cand.T) @ H_sel @ np.conjugate(H_sel.T) @ H_cand)


        U = list(range(self.user))
        S = []
        H_cand = np.zeros((self.BS_antenna, self.user_antenna), dtype=np.complex)
        H_sel = np.zeros((self.BS_antenna, self.user_antenna), dtype=np.complex)

        for k in range(self.select_user):
            if k == 0:
                s1 = 0
                tmp_s = 0

                for s in range(self.user):

                    for i in range(self.BS_antenna):
                        for j in range(self.user_antenna):
                            H_cand[i, j] = channel[0, i*self.user*self.user_antenna + s*self.user_antenna + j]

                    tmp = np.linalg.norm(H_cand,'fro')

                    if tmp > tmp_s:
                        tmp_s = tmp
                        s1 = s
                        S1 = H_cand

                U.remove(s1)
                S.append(s1)
                for i in range(self.BS_antenna):
                    for j in range(self.user_antenna):
                        H_sel[i,j] = S1[i,j]

            else:
                s1 = 0
                tmp_s = 0

                for s in U:
                    for i in range(self.BS_antenna):
                        for j in range(self.user_antenna):
                            H_cand[i, j] = channel[0, i*self.user*self.user_antenna + s*self.user_antenna + j]

                    tmp = np.real(chordal_distance(H_cand, H_sel))
                    if tmp > tmp_s:
                        tmp_s = tmp
                        s1 = s
                        S1 = H_cand.copy()

                U.remove(s1)
                S.append(s1)
                H_sel = np.concatenate([H_sel, S1], axis=1)

        return H_sel


    def RAND(self, channel):

        H_cand = np.zeros((self.BS_antenna, self.user_antenna), dtype=np.complex)
        H_sel = np.zeros((self.BS_antenna, self.user_antenna), dtype=np.complex)

        U = random.sample(list(range(self.user)), self.select_user)

        cnt = 0
        for s in U:
            if cnt == 0:
                for i in range(self.BS_antenna):
                    for j in range(self.user_antenna):
                        H_sel[i, j] = channel[0, i*self.user*self.user_antenna + s*self.user_antenna + j]

            else:
                for i in range(self.BS_antenna):
                    for j in range(self.user_antenna):
                        H_cand[i, j] = channel[0, i*self.user*self.user_antenna + s*self.user_antenna + j]

                H_sel = np.concatenate([H_sel, H_cand], axis=1)

            cnt += 1

        return H_sel


    def ALL(self, channel, SNR):

        capacity = 0
        for U in itertools.combinations(list(range(self.user)), self.select_user):

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

            C = MMSE_channel_capacity(H_tmp, self.select_user, self.user_antenna, self.BS_antenna, SNR)

            if capacity < C:
                capacity = C
                H_sel = H_tmp.copy()

        return H_sel, capacity






