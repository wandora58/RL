import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.distributions import Categorical


class ActorDecoder(object):

    def __init__(self, n_neurons=128, batch_size=4, user=6):
        """

        Actor Decoder class

        Args:
            n_neurons: int
                Hidden layer of LSTM

            user: int
                Num of user

            self.infty: 1.0E+08
                Penalties for point mask

            self.mask: int
                point mask bit

            self.dec_first_input: 2D tensor [batch_size, n_neuron]
                Initial input parameter variable

        """
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.user = user

        self.infty = 1.0E+08
        self.mask = 0
        self.seed = None

        # Initial input parameter variable [batch_size, n_neuron]
        first_input = tf.get_variable('GO', [1, self.n_neurons])
        self.dec_first_input = tf.tile(first_input, [self.batch_size, 1])

        # Pointing mechanism parameter variable
        self.W_ref = tf.get_variable('W_ref', [1, self.n_neurons, self.n_neurons])
        self.W_out = tf.get_variable('W_out', [self.n_neurons, self.n_neurons])
        self.v = tf.get_variable('v', [self.n_neurons])

        # Define Recursive cell
        self.dec_rec_cell = LSTMCell(self.n_neurons)


    def set_seed(self, seed):
        self.seed = seed


    def build_model(self, enc_outputs, enc_state, sel_user):
        """

        Define network
        Network configuration by Pointing mechanism and calculation of corresponding log-likelihood

        Args:
            enc_outputs: 3D tensor [batch, user, n_neurons]
                Whole sequence outputs


            enc_state: 1D list [tensor, tensor]

                enc_state[0]: 2D tensor [batch, n_neurons]
                    Final memory state

                enc_state[1]: 2D tensor [batch, n_neurons]
                    Final carry state

            sel_user: int
                Num of select user

        Outputs:
            tour: 2D tensor [batch, sel_user]
                Sequence generated by Encoder + decoder network

            sequence likelihood: 1D tensor
                Likelihood of sequence

        """

        # Reshape [user, batch, n_neurons]
        output_list = tf.transpose(enc_outputs, [1, 0, 2])

        # input: 2D tensor [batch, n_neuron]
        # state: 2D tensor [batch, n_neuron]
        input, state = self.dec_first_input, enc_state

        locations, log_probs = [], []

        for step in range(sel_user):

            # output: 2D tensor [batch, n_neuron]
            #  state: 2D tensor [batch, n_neuron]
            output, state = self.dec_rec_cell(input, state)

            # next logit score at each point
            masked_scores = self._pointing(enc_outputs, output)

            # Definie Multinomial distribution with logit score at each point
            prob = Categorical(logits=masked_scores)

            # Selection of next point according to Multinomial distribution
            location = prob.sample(seed=self.seed)

            # Register of selected points
            locations.append(location)

            # Calculation of log-likelihood (tensor) of selected point
            logp = prob.log_prob(location)

            # Register Log-likelihood
            log_probs.append(logp)

            # Mask visited point
            self.mask = self.mask + tf.one_hot(location, self.user)

            # Update next input
            input = tf.gather(output_list, location)[0]


        # Sequence generated by Encoder + decoder network and likelihood of sequence
        combinations = tf.stack(locations, axis=1)
        log_prob = tf.add_n(log_probs)

        return log_prob, combinations


    def _pointing(self, enc_outputs, dec_output):

        """
        Calculate next logit score at each points using encoder info and decoder info

        Args:
            enc_outputs: 3D tensor [batch, user, n_neurons]
                Encoder outputs whole sequence

            dec_output: 2D tensor [batch, n_neuron]
                Decoder output for each time step

        Outputs:
            masked_scores: 2D tensor [batch, seq_len]
                Next logit scores at each points (visited points are masked)

        """

        # Encoder info: 3D tensor [batch, user, n_neuron]
        enc_term = tf.nn.conv1d(enc_outputs, self.W_ref, 1, 'VALID')

        # Decoder info: 3D tensor [batch, 1, n_neuron]
        dec_term = tf.expand_dims(tf.matmul(dec_output, self.W_out), 1)

        # Calculation of score by reference [batch, user]
        scores = tf.reduce_sum(self.v * tf.tanh(enc_term + dec_term), [-1])

        # Add -infty to score of masked points (different for each batch)
        masked_scores = scores - self.infty * self.mask

        return masked_scores