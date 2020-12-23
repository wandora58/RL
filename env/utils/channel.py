
import numpy as np
import math

from env.utils import util


class Channel:
    def __init__(self, user, user_antenna, BS_antenna, Nc, path, CP, Ps, code_symbol):
        """

        Mimo channel class.

        """
        self.user = user
        self.user_antenna = user_antenna
        self.BS = BS_antenna
        self.Nc = Nc
        self.path = path
        self.CP = CP
        self.Ps = Ps
        self.code_symbol = code_symbol
        self.weight = util.Weight().create_weight(path)
        self.channel = np.zeros((Nc, user * user_antenna * BS_antenna), dtype=np.complex)


    def create_rayleigh_channel(self):
        """

        Create Rayleigh fading channel

        Returns:
            channel : 2D ndarray [Nc, user * user_antenna * BS_antenna]

                                   BS_a1                          /                         BS_a2
              US1_a1   US1_a2    ...  /   US2_a1   US2_a2    ...  /   US1_a1   US1_a2    ...  /   US2_a1   US2_a2   ...
     path1 [[    h11      h12    h13         h21      h22    h23         h11      h12    h13         h21      h22   h23 ],
     path2  [    h11      h12    h13         h21      h22    h23         h11      h12    h13         h21      h22   h23 ],
     path3  [    h11      h12    h13         h21      h22    h23         h11      h12    h13         h21      h22   h23 ],
                                                                                                                        ]]

        """
        for i in range(self.user * self.user_antenna * self.BS):

            Ps = self.Ps
            for path in range(self.path):
                self.channel[path, i] = util.Boxmuller().create_normalized_random(Ps / self.weight * math.sqrt(0.5))
                # Ps *= math.sqrt(math.pow(10, -0.1))  # 1dB減衰

        return self.channel


    def channel_multiplication(self, channel, send_signal):
        """

        Multiplies send signal by mimo channel

        Args:
            channel: 2D ndarray [BS_antenna, select_user * user_antenna]

                     seU1_a1   seU1_a2    ...   /  seU2_a1   seU2_a2    ...
            BS_a1 [[     h11       h12    h13          h21       h22    h23 ],
            BS_a2  [     h11       h12    h13          h21       h22    h23 ],
            BS_a3  [     h11       h12    h13          h21       h22    h23 ],
                                               ...                          ]]


            send_signal: 2D ndarray [select_user * user_antenna, symbol * 1/code_rate]

                                                  symbol * 1/code_rate

                                    seU1_a1  [[symbol11, symbol12, symbol13]
                                    seU1_a2   [symbol21, symbol22, symbol23]
                                       ..                   ..
                                    seU2_a1   [symbol31, symbol32, symbol33]
                                    seU2_a2   [  ...  ,    ...   ,   ...   ]]


        Returns:
            receive_signal: 2D ndarray [BS_antenna, symbol * 1/code_rate]

                                                  symbol * 1/code_rate

                                      BS_a1  [[symbol11, symbol12, symbol13]
                                      BS_a2   [symbol21, symbol22, symbol23]
                                      BS_a3   [symbol31, symbol32, symbol33]
                                              [  ...  ,    ...   ,   ...   ]]

                                       If path == 1:
                                          channel is converted to a diagonal matrix and multiplied by send signal

                                       If path >= 1:
                                          channel is processed by Cyclic Prefix and then send signal is multiplied

        """
        self.receive_signal = np.dot(channel, send_signal)


        # elif self.path >= 1:
        #     h = np.zeros(self.path, dtype=np.complex)
        #     Hb = np.zeros((self.symbol + self.CP, self.symbol + self.CP), dtype=np.complex)
        #     H = np.zeros((self.symbol, self.symbol), dtype=np.complex)
        #     CP = utils.CP(self.CP, self.symbol)
        #
        #     for r in range(self.BS):
        #
        #         send_tmp = np.zeros((self.symbol, 1), dtype=np.complex)
        #         receive_tmp = np.zeros((self.symbol, 1), dtype=np.complex)
        #
        #         for s in range(self.user):
        #
        #             for p in range(self.path):
        #                 h[p] = self.channel[p][r * BS + s]
        #
        #             k = 0
        #             for count in range(self.path):
        #                 for i in range(self.symbol + self.CP):
        #                     for j in range(self.symbol + self.CP):
        #
        #                         if (j + k == i):
        #                             Hb[i, j] = h[k]  # Add CP
        #
        #                 k += 1
        #
        #             H = np.dot(np.dot(CP.remove_CP(), Hb), CP.add_CP())  # Remove CP
        #
        #             for i in range(self.symbol):
        #                 send_tmp[i, 0] = self.send_signal[i, s]
        #
        #             receive_tmp += np.dot(H, send_tmp)
        #
        #         for i in range(self.symbol):
        #             self.receive_signal[i, r] = receive_tmp[i, 0]

        return self.receive_signal



