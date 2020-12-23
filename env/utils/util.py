
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

from collections import defaultdict

COMMON_COLUMN = 'SNR'
BER_COLUMN = 'BER'
NMSE_COLUMN = 'NMSE'
TRP_COLUMN = 'throughput'

class Weight:
    def __init__(self):
        pass

    def create_weight(self, path):
        w = 1
        ww = 1

        for i in range(path-1):
            w *= pow(10,-0.1)
            ww += w

        weight = math.sqrt(ww)

        return weight


class Boxmuller:
    def __init__(self):
        pass

    def create_normalized_random(self, k):
        a = random.random()
        b = random.random()

        x = k * math.sqrt(-2 * math.log(a)) * math.sin(2 * math.pi * b)
        y = k * math.sqrt(-2 * math.log(a)) * math.cos(2 * math.pi * b)

        z = x + y * 1j

        return z


class Noise(Boxmuller):
    def __init__(self):
        pass

    def create_noise(self, symbol, sigma, BS):
        """

        Create White Gaussian noise

        Returns:
            noise : 2D ndarray [BS, symbol * 1/code_rate]

        """
        noise = np.zeros((BS, symbol), dtype=np.complex)

        for r in range(BS):
            for i in range(symbol):
                noise[r, i] = (super().create_normalized_random(sigma))

        # noise /= np.sqrt(BS)

        return noise


class CP:
    def __init__(self, CP, symbol):
        self.CP = CP
        self.symbol = symbol

    def add_CP(self):
        O = np.zeros((self.CP, self.symbol - self.CP))
        Icp = np.eye(self.CP, k=0)
        I = np.eye(self.symbol, k=0)

        return np.vstack([np.hstack([O, Icp]), I])


    def remove_CP(self):
        O = np.zeros((self.symbol, self.CP))
        I = np.eye(self.symbol, k=0)

        return np.hstack([O, I])


class Result:
    def __init__(self, flame, user, BS, BW, data_symbol, Nc, conv_type, channel_type, csv_column, last_snr):
        self.flame = flame
        self.user = user
        self.BS = BS
        self.BW = BW
        self.symbol = data_symbol
        self.Nc = Nc
        self.conv_type = conv_type
        self.channel_type = channel_type
        self.csv_column = csv_column
        self.last_snr = last_snr

        self.tmp_true = np.zeros((Nc*user*BS),dtype=np.complex)
        self.tmp_estimated = np.zeros((Nc*user*BS),dtype=np.complex)

        self.h_true = np.zeros((flame*Nc*user*BS),dtype=np.complex)
        self.h_estimated = np.zeros((flame*Nc*user*BS),dtype=np.complex)

        self.SNR = 0
        self.BER = defaultdict(lambda: 0)
        self.NMSE = defaultdict(lambda: 0)
        self.TRP = defaultdict(lambda: 0)


    def calculate(self, count, send_data, receive_data, mod_signal, noise, SNR, M, bit_rate, code_rate, true_channel=None, estimated_channel=None):
        self.count = count
        self.send_data = send_data
        self.receive_data = receive_data
        self.send_signal = mod_signal
        self.noise = noise
        self.SNRs = SNR
        self.M = M
        self.bit_rate = bit_rate
        self.code_rate = code_rate

        self.true_channel = true_channel
        self.estimated_channel = estimated_channel

        if self.estimated_channel is None:
            self.type = 'true'
        else:
            self.type = 'zadoff'

        if self.count == 0:
            self.rrs = 0
            self.rrn = 0
            self.ber = 0
            self.bler = 0
            self.nmse = 0

        # SNR
        self.rrs += np.sum(np.abs(self.send_signal) * (np.abs(self.send_signal)))
        self.rrn += np.sum(np.abs(self.noise) * np.abs(self.noise))

        # BER
        tmp = 0
        for i in range(self.user * self.symbol * self.bit_rate):
            if self.send_data[i] != self.receive_data[i]:
                self.ber += 1
                tmp = 1

        if tmp == 1:
            self.bler += 1

        # NMSE
        if self.estimated_channel is None:
            pass
        else:
            for j in range(self.user*self.BS):
                for k in range(self.Nc):

                    self.tmp_true[self.Nc*j+k] = self.true_channel[k][j]
                    self.tmp_estimated[self.Nc*j+k] = self.estimated_channel[k][j]

            for i in range(self.Nc*self.user*self.BS):

                self.h_true[self.count*self.Nc*self.user*self.BS+i] = self.tmp_true[i]
                self.h_estimated[self.count*self.Nc*self.user*self.BS+i] = self.tmp_estimated[i]


        if self.count+1 == self.flame:
            self.snr = 10 * math.log10(self.rrs/self.rrn)
            self.ber /= self.flame * self.symbol * self.user * self.bit_rate
            self.bler /= self.flame * self.user
            self.trp = self.BW * self.user * self.symbol * self.bit_rate * self.code_rate * (1-self.bler)

            self.SNR += self.snr
            self.BER['{}({})'.format(self.mod_name(self.M), self.code_rate_name(self.code_rate))] = self.ber
            self.TRP['{}({})'.format(self.mod_name(self.M), self.code_rate_name(self.code_rate))] = self.trp

            if self.estimated_channel is None:
                pass
            else:
                self.nmse = 10 * math.log10(np.sum(np.abs(self.h_true-self.h_estimated)*np.abs(self.h_true-self.h_estimated)) / np.sum(np.abs(self.h_estimated)*np.abs(self.h_estimated)))
                self.NMSE['{}({})'.format(self.mod_name(self.M), self.code_rate_name(self.code_rate))] = self.nmse

            if self.M == self.csv_column[0][-1] and self.code_rate == self.csv_column[1][-1]:
                self.SNR /= len(self.csv_column[0]) * len(self.csv_column[1])
                self.BER['SNR'] = self.SNR
                self.TRP['SNR'] = self.SNR
                if self.estimated_channel is None:
                    pass
                else:
                    self.NMSE['SNR'] = self.SNR
                self.file_writing()

    def mod_name(self, M):
        if M == 4:
            return 'QPSK'

        elif M == 16:
            return '16QAM'

        elif M == 64:
            return '64QAM'

        elif M == 256:
            return '256QAM'


    def code_rate_name(self, code_rate):
        if code_rate == 1/2:
            return '1/2'

        elif code_rate == 2/3:
            return '2/3'

        elif code_rate == 3/4:
            return '3/4'


    def column_name(self):
        column = ['SNR']
        for M in self.csv_column[0]:
            for rate in self.csv_column[1]:
                column.append('{}({})'.format(self.mod_name(M), self.code_rate_name(rate)))

        return column


    def file_writing(self):

        columns = self.column_name()
        BER_file = "BER/{}/{}/user{}_BS{}.csv".format(self.type, self.channel_type, self.user, self.BS)

        if not os.path.exists(BER_file):
            pathlib.Path(BER_file).touch()

        TRP_file = "TRP/{}/{}/user{}_BS{}.csv".format(self.type, self.channel_type, self.user, self.BS)

        if not os.path.exists(TRP_file):
            pathlib.Path(TRP_file).touch()

        if self.estimated_channel is None:
            pass
        else:
            NMSE_file = "NMSE/{}/{}/user{}_BS{}.csv".format(self.type, self.channel_type, self.user, self.BS)

            if not os.path.exists(NMSE_file):
                pathlib.Path(NMSE_file).touch()

        if self.SNRs == 0:
            with open(BER_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

            with open(TRP_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

            if self.estimated_channel is None:
                pass
            else:
                with open(NMSE_file, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()


        with open(BER_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow(self.BER)

        with open(TRP_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow(self.TRP)

        if self.estimated_channel is None:
            pass
        else:
            with open(NMSE_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(self.NMSE)

        for column in columns:
            if column != 'SNR':
                print('--------------------------------')
                print('TYPE: {}'.format(column))
                print('SNR: {}'.format(self.SNR))
                print('BER: {}'.format(self.BER[column]))
                print('TRP: {}'.format(self.TRP[column]))
                if self.estimated_channel is None:
                    pass
                else:
                    print('NMSE: {}'.format(self.NMSE[column]))
                print('--------------------------------')

        if self.SNRs == self.last_snr-1:
            self.illustrate()


    def illustrate(self):

        BER_file = "BER/{}/{}/user{}_BS{}.csv".format(self.type, self.channel_type, self.user, self.BS)
        TRP_file = "TRP/{}/{}/user{}_BS{}.csv".format(self.type, self.channel_type, self.user, self.BS)

        BER_df = pd.read_csv(BER_file, index_col=0)
        TRP_df = pd.read_csv(TRP_file, index_col=0)

        #--------------BER---------------------------------------

        #figure　
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        #plot
        BER_df.plot()

        #y軸の範囲設定
        plt.ylim([0.0001,0.5])

        #y軸を片対数
        plt.yscale('log')

        #ひげ消す
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.gca().yaxis.set_tick_params(direction='in')

        #x軸間隔
        plt.xticks([0,5,10,15,20,25,30])

        #x軸,y軸のラベル付け
        plt.xlabel('Average SNR [dB]', fontsize=12)
        plt.ylabel('Average BER', fontsize=12)

        #グリッド表示
        plt.grid(which="both")

        #凡例とタイトル
        ax.legend(loc='best',prop={'size':12})

        #保存　
        plt.savefig('Image/{}/BER/user{}_BS{}.pdf'.format(self.type, self.user, self.BS))


        #--------------TRP---------------------------------------

        #figure　
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        #plot
        TRP_df.plot()

        #y軸の範囲設定
        # ax.set_xlim([0,25])
        # ax.set_ylim([0.0001,0.5])

        #ひげ消す
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.gca().yaxis.set_tick_params(direction='in')

        #x軸間隔
        plt.xticks([0,5,10,15,20,25,30])
        plt.yticks([0,10,20,30,40,50,60,70,80])

        #x軸,y軸のラベル付け
        plt.xlabel('Average SNR [dB]', fontsize=12)
        plt.ylabel('Throughput [Mbps]', fontsize=12)

        #グリッド表示
        plt.grid(which="both")

        #凡例とタイトル
        ax.legend(loc='best',prop={'size':12})

        #保存　
        plt.savefig('Image/{}/TRP/user{}_BS{}.pdf'.format(self.type, self.user, self.BS))


        #--------------NMSE---------------------------------------

        if self.estimated_channel is None:
            pass
        else:
            NMSE_file = "NMSE/{}/{}/user{}_BS{}.csv".format(self.type, self.channel_type, self.user, self.BS)
            NMSE_df = pd.read_csv(NMSE_file, index_col=0)

            #figure　
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            #plot
            NMSE_df.plot()

            #ひげ消す
            plt.gca().xaxis.set_tick_params(direction='in')
            plt.gca().yaxis.set_tick_params(direction='in')

            #x軸間隔
            plt.xticks([0,5,10,15,20,25,30])

            #x軸,y軸のラベル付け
            plt.xlabel('Average SNR [dB]', fontsize=12)
            plt.ylabel('Normalized MSE [dB]', fontsize=12)

            #グリッド表示
            plt.grid(which="both")

            #凡例とタイトル
            ax.legend(loc='best',prop={'size':12})

            #保存　
            plt.savefig('Image/{}/NMSE/user{}_BS{}.pdf'.format(self.type, self.user, self.BS))


class Load:
    def __init__(self, sample, test, user, total_bit, SNR, input_type):
        self.sample = sample
        self.test = test
        self.user = user
        self.total_bit = total_bit
        self.SNR = SNR

        self.input = "Data/train_data/{}/user{}_bit{}_SNR={}.csv".format(input_type, user, total_bit, SNR)
        self.ans = "Data/train_data/bit/user{}_bit{}_SNR={}.csv".format(user, total_bit, SNR)
        self.ans_dict = "Data/train_data/bit_dict/user{}_bit{}.csv".format(user, total_bit)


    def load_data(self):

        input = []
        ans_dict = []
        answer = []

        with open(self.input, 'r') as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                input.append(row)

        with open(self.ans_dict, 'r') as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                row = [int(i) for i in row]
                ans_dict.append(row)

        unit = np.eye(len(ans_dict), dtype=np.int)
        count = np.zeros(len(ans_dict), dtype=np.int)
        with open(self.ans, 'r') as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                row = [int(i) for i in row]
                answer.append(unit[ans_dict.index(row)])
                count[ans_dict.index(row)] += 1

        print('--------------------------------')
        for i in range(len(count)):
            print('DATA[{}] : {}'.format(i, count[i]))
        print('--------------------------------')


        input_data = input[:self.sample][:]
        input_test = input[self.sample:][:]

        answer_data = answer[:self.sample][:]
        answer_test = answer[self.sample:][:]

        return np.array(input_data), np.array(input_test), np.array(answer_data), np.array(answer_test)


class Serial2Parallel:
    def __init__(self, data_stream, select_user, user_antenna, bit_rate, data_symbol):
        self.data_stream = data_stream
        self.user = select_user
        self.user_antenna = user_antenna
        self.bit_rate = bit_rate
        self.symbol = data_symbol

    def create_parallel(self):
        """
        Serial to Parallel

        Args:
            data_stream : 1D ndarray [select_user * user_antenna * bit_rate * symbol],

        Returns:
            send_bit : 2D ndarray [select_user * user_antenna, bit_rate * symbol]

        """
        send_bit = []
        tmp = 0
        for i in range(self.user * self.user_antenna):
            if i == 0:
                send_bit.append(self.data_stream[: self.symbol*self.bit_rate])
            else:
                send_bit.append(self.data_stream[tmp : tmp + self.symbol*self.bit_rate])

            tmp += self.symbol*self.bit_rate

        return np.array(send_bit)


class Parallel2Serial:
    def __init__(self, receive_symbol):
        self.receive_symbol = receive_symbol

    def create_serial(self):
        return list(itertools.chain.from_iterable(self.receive_symbol))


class Bitrate:
    def __init__(self, M):
        self.M = M

    def count_bit_rate(self):
        """

        Calculates the number of bits per symbol from the M-ary modulation number M

        Returns:
            bit_rate (int): the number of bits per symbol

        """

        bit_rate = 0
        while(True):
            if 2 ** bit_rate == self.M:
                break
            bit_rate += 1

        return bit_rate


def create_cor_matirix(pilot_signal):

    pilot_len = np.shape(pilot_signal)[0]
    user = np.shape(pilot_signal)[1]

    cor_matrix = np.zeros((user, user), dtype=np.complex)
    cor_tmp1 = np.zeros((1, pilot_len), dtype=np.complex)
    cor_tmp2 = np.zeros((pilot_len, 1), dtype=np.complex)

    for i in range(user):

        for j in range(pilot_len):
            cor_tmp1[0,j] = pilot_signal[j,i]

        for k in range(user):
            for l in range(pilot_len):
                cor_tmp2[l,0] = pilot_signal[l,k]

            cor_matrix[i,k] = np.dot(cor_tmp1,np.conjugate(cor_tmp2)) / (np.linalg.norm(cor_tmp1, ord=2) * np.linalg.norm(cor_tmp2, ord=2))

    print(cor_matrix)



def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to NumPy array of the specified size containing
    bits (0 and 1).
    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.
    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width)
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width)
    return result


def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only work for one integer.
    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.
    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """
    result = np.zeros(bit_width, np.int8)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.
    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.
    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).
    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.
    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.
    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()

    return distance


def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays
    Parameters
    ----------
    in_array1 : 1D ndarray of floats
        NumPy array of real values.
    in_array2 : 1D ndarray of floats
        NumPy array of real values.
    Returns
    -------
    distance : float
        Squared Euclidean distance between two input arrays.
    """
    distance = ((in_array1 - in_array2) * (in_array1 - in_array2)).sum()

    return distance


def signal_power(signal):
    """
    Compute the power of a discrete time signal.
    Parameters
    ----------
    signal : 1D ndarray
             Input signal.
    Returns
    -------
    P : float
        Power of the input signal.
    """

    @np.vectorize
    def square_abs(s):
        return abs(s) ** 2

    P = np.mean(square_abs(signal))
    return P



def cumulative_probability(channel, user, BS):
    count = 100000
    matrix = np.zeros((count,user), dtype=np.float)

    for cnt in range(count):
        h = channel.create_rayleigh_channel()
        H = np.zeros((BS, user), dtype=np.complex)

        for r in range(BS):
            for s in range(user):
                H[r][s] = h[0][BS*r + s]

        gram_H = np.conjugate(H.T) @ H
        U, A ,Uh = svd(gram_H)

        for s in range(user):
            matrix[cnt][s] = 10 * math.log10(A[s])

    matrix = np.sort(matrix,axis=0)

    x = range(-40,20)
    cumulative_matrix = np.zeros((user, len(x)),dtype=np.double)

    for i in range(len(x)) :
        for s in range(user):
            tmp = np.where(matrix[:,s] <= x[i], 1, matrix[:,s])
            cumulative_matrix[s,i] = sum(tmp == 1) / count

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(user):
        ax.plot(x, cumulative_matrix[i], linestyle='-', color='k')

    plt.gca().xaxis.set_tick_params(direction='in')
    plt.gca().yaxis.set_tick_params(direction='in')

    ax.set_xlabel('eigenvalue [dB]', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)

    plt.grid(which="both")

    plt.savefig('Image/cdf/user{}_BS{}.pdf'.format(user, BS))


def MMSE_channel_capacity(channel, user, user_antenna, BS_anntena, SNR):

    H = channel
    Nt = user * user_antenna
    Nr = BS_anntena
    INr = (Nt / SNR) * np.eye(Nr)
    W_MMSE = np.linalg.pinv(np.conjugate(H) @ H.T + INr)

    C_MMSE = 0

    for i in range(Nt):
        h = H[:,i]
        w = W_MMSE @ np.conjugate(h)

        C_MMSE += np.log2(1 + np.abs(w.T @ h) ** 2 / (np.conjugate(w.T) @ np.conjugate(h) - np.abs(w.T @ h) ** 2))

    return np.real(C_MMSE)


