import tensorflow as tf
from tensorflow.keras import backend as K


# huber loss for value
def rms_loss(y_true, y_pred, huber_loss_delta=1.0):
    err = y_true - y_pred

    cond = K.abs(err) < huber_loss_delta
    l2 = 0.5 * K.square(err)
    l1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)

    loss = tf.where(cond, l2, l1)

    return K.mean(loss)


# policy gradient loss for policy
def policy_gradient_loss(reward, log_probs):
    loss = -1 * reward * log_probs

    return K.mean(loss)


# calculate loss
def calculate_loss(model_prds, reward):
    """
    expected earnings = [13.297053, 13.401614, 13.325188, 13.591579]  expect G(t)
    true_earnings = [15.455151 15.467147 12.626222 13.090746]         true G(t)

    -1 (V - R) * logp x

    advantage = -1 * [-2.16, -2.06, 0.7, 0.5]
              = [2.16, 2.06, -0.7, -0.5] (- 失敗 / + 成功)

    log_prob = [-10.543707  -10.167185  -10.3442545  -9.802457 ]      log-likelifood of sequence

    loss = mean([-22.76, -20.92, 7.238, 4.9])

    advantage = true G(t) - expect G(t)
    actor_loss = mean(-1 * (true G(t) - expect G(t)) * log_prob)

    critic_loss = root mean square(true G(t) - expect G(t))
    """

    log_prob, tour, expect_earnings = model_prds
    true_earnings = reward

    # advantage for PG
    advantage = -1.0 * tf.stop_gradient(expect_earnings - true_earnings)

    # actor loss
    loss_actor = policy_gradient_loss(advantage, log_prob)

    # critic loss
    loss_critic = rms_loss(expect_earnings, true_earnings)

    # loss weight
    loss_wt = [1.0, 1.0]
    loss = loss_wt[0] * loss_actor + loss_wt[1] * loss_critic

    return loss, loss_critic, loss_actor



