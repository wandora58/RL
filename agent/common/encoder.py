import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.distributions import Categorical


class Encoder(object):

    def __init__(self, n_neurons=128, batch_size=4, user=6):

        """

        Actor Encoder class

        Args:
            n_neurons: int
                Hidden layer of LSTM

            user: int
                Num of user

        Outputs:
            enc_outputs: 3D tensor [batch, user, n_neurons]
                Whole sequence outputs


            enc_state: 1D list [tensor, tensor]

                enc_state[0]: 2D tensor [batch, n_neurons]
                    Final memory state

                enc_state[1]: 2D tensor [batch, n_neurons]
                    Final carry state
        """

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.user = user

        # Define Recursive cell
        self.enc_rec_cell = LSTMCell(self.n_neurons)


    def build_model(self, inputs):

        # Insert Bi-directional LSTM layer
        inputs = Bidirectional(LSTM(self.n_neurons, return_sequences=True), merge_mode='concat')(inputs)

        # Reshape [user, batch_size, n_neurons*2]
        input_list = tf.transpose(inputs, [1, 0, 2])

        # Setting inputs and batch_size of LSTMCell
        state = self._get_initial_state()
        enc_outputs, enc_states = [], []

        for input in tf.unstack(input_list, axis=0):
            # input is time step of sequence with shape of [batch, n_neurons*2]
            output, state = self.enc_rec_cell(input, state)

            enc_outputs.append(output)
            enc_states.append(state)

        # Concat & transpose
        enc_outputs = tf.stack(enc_outputs, axis=0)
        enc_outputs = tf.transpose(enc_outputs, [1, 0, 2])
        enc_state = enc_states[-1]

        return enc_outputs, enc_state


    def _get_initial_state(self):
        state = self.enc_rec_cell.get_initial_state(inputs=None,
                                                    batch_size=self.batch_size,
                                                    dtype=tf.float32)
        return state



