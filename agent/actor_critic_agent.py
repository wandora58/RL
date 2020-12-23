import os

import tensorflow as tf

from agent.common.encoder import Encoder
from agent.actor.decoder import ActorDecoder
from agent.critic.decoder import CriticDecoder

from agent.loss import calculate_loss
from agent.optimizer import build_critic_optimizer, build_actor_optimizer

from utils import get_channel_capacity


class ActorCriticAgent(object):

    def __init__(self, n_neurons=128, batch_size=4, user=6, sel_user=4, feature_dim=8, user_antenna=1, BS_antenna=4, SNRdB=10,
                 critic_learning_rate=1.0E-03, actor_learning_rate=1.0E-03):

        # Parameter
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.user = user
        self.sel_user = sel_user
        self.feature_dim = feature_dim
        self.user_antenna = user_antenna
        self.BS_antenna = BS_antenna
        self.SNR = 10 ** (SNRdB / 10)
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate

        # Define input data placeholder
        data_dim = (self.user, self.feature_dim)
        input = tf.placeholder(shape=(None, *data_dim), dtype=tf.float32, name='input_data')
        self.p_holders = input

        c_data_dim = (self.user, self.BS_antenna*2)
        channel = tf.placeholder(shape=(None, *c_data_dim), dtype=tf.float32, name='channel_data')
        self.c_holders = channel

        # Create common Encoder network
        self.encoder = Encoder(batch_size=self.batch_size, user=self.user)
        enc_outputs, enc_state = self.encoder.build_model(input)

        # Create network of Policy function (Actor) based on Encoder output
        self.actor_decoder = ActorDecoder(batch_size=self.batch_size, user=self.user)
        log_prob, self.combinations = self.actor_decoder.build_model(enc_outputs, enc_state, sel_user)

        # Create network of value function (Critic) based on Encoder output
        self.critic_decoder = CriticDecoder(batch_size=self.batch_size, user=self.user)
        self.state_value = self.critic_decoder.build_model(enc_outputs, enc_state)

        # Calculate distance (reward) for tour
        self.reward = get_channel_capacity(channel, self.combinations, batch_size, sel_user, user, user_antenna, BS_antenna, self.SNR)

        # Calculate loss
        self.model_prds = [log_prob, self.combinations, self.state_value]
        loss, loss_critic, loss_actor = calculate_loss(self.model_prds, self.reward)
        self.losses = [loss, loss_critic, loss_actor]

        # Create optimizer
        opt_critic, grad_critic = build_critic_optimizer(loss_critic, critic_learning_rate)
        opt_actor, grad_actor = build_actor_optimizer(loss_actor, actor_learning_rate)
        self.opts = [opt_critic, opt_actor]

        # Define saver for graph variables
        self.saver = self._build_graph_saver()


    # Update model (= agent.predict, env.step, agent.update)
    def update_model(self, sess, state, channel):

        input_data = self.p_holders
        channel_data = self.c_holders
        v_optim, p_optim = self.opts

        feed_dict = {input_data: state, channel_data: channel}

        # Update critic
        tensors = [v_optim, self.losses, self.reward, self.model_prds]
        _, losses, reward, mode_prds = sess.run(tensors, feed_dict)

        # Updata actor
        tensors = [p_optim, self.losses, self.reward, self.model_prds]
        _, losses, reward, mode_prds = sess.run(tensors, feed_dict)

        return losses, reward, mode_prds


    # Predict loss（= agent.predict、env.step）
    def predict_loss(self, sess, state, channel):

        input_data = self.p_holders
        channel_data = self.c_holders

        feed_dict = {input_data: state, channel_data: channel}
        tensors = self.combinations
        combi = sess.run(tensors, feed_dict)

        return combi
        # tensors = [self.losses, self.reward, self.model_prds]
        # losses, reward, model_prds = sess.run(tensors, feed_dict)
        #
        # return losses, reward, model_prds


    # save model
    def _build_graph_saver(self):
        variables_to_save = tf.global_variables()
        return tf.train.Saver(var_list=variables_to_save)

    def save_graph(self, sess, log_dir, args):
        fname = 'model.{0:06d}-{1:3.3f}-{2:3.5f}.ckpt'.format(*args)
        self.saver.save(sess, os.path.join(log_dir, fname))

    def restore_graph(self, sess, model_path):
        self.saver.restore(sess, model_path)






