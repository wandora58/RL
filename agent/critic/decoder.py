
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.distributions import Categorical


class CriticDecoder(object):

    def __init__(self, n_neurons=128, batch_size=4, user=6):
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.user = user

        # Glimpsing mechanism parameter variable
        self.W_ref_g = tf.get_variable('W_ref_g', [1, self.n_neurons, self.n_neurons])
        self.W_q_g = tf.get_variable('W_q_g', [self.n_neurons, self.n_neurons])
        self.v_g = tf.get_variable('v_g', [self.n_neurons])


    def build_model(self, enc_outputs, enc_state):

        """

        Define network
        Expected reward value = length of tour

        Args:
            enc_outputs: 3D tensor [batch, user, n_neurons]
                Whole sequence outputs


            enc_state: 1D list [tensor, tensor]

                enc_state[0]: 2D tensor [batch, n_neurons]
                    Final memory state

                enc_state[1]: 2D tensor [batch, n_neurons]
                    Final carry state

        Outputs:
            baseline: double
                Expected reward value
        """

        # ------ Attention -------

        # Encoder info: 3D tensor [batch, user, n_neuron]
        enc_ref_g = tf.nn.conv1d(enc_outputs, self.W_ref_g, 1, 'VALID', name='encoded_ref_g')

        # State info: 3D tensor [batch, 1, n_neuron]
        enc_q_g = tf.expand_dims(tf.matmul(enc_state[0], self.W_q_g, name='encoded_q_g'), 1)

        # Logit score: 2D tensor [batch, user]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(enc_ref_g + enc_q_g), [-1], name='scores_g')

        # Attention: 2D tensor [batch, user]
        attention_g = tf.nn.softmax(scores_g, name='attention_g')


        # ------- Glimpsing -------

        # Glimpsing: [batch, user, n_neuron] to [batch, n_neuron]
        glimpse = tf.multiply(enc_outputs, tf.expand_dims(attention_g, axis=2))
        glimpse = tf.reduce_sum(glimpse, axis=1)
        # glimpse = tf.reduce_mean(enc_outputs, axis=1)


        # -------- main FC --------
        hidden = Dense(self.n_neurons, activation='relu')(glimpse)
        baseline = Dense(1, activation='linear')(hidden)

        return baseline


