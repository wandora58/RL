
import tensorflow as tf


# Create critic optimizer
def build_critic_optimizer(val_loss, val_lr):

    val_optimizer = tf.train.RMSPropOptimizer(val_lr)

    val_optim = val_optimizer.minimize(val_loss)
    val_grad = val_optimizer.compute_gradients(val_loss)

    return val_optim, val_grad


# Create actor optimizer
def build_actor_optimizer(pol_loss, pol_lr):

    pol_optimizer = tf.train.RMSPropOptimizer(pol_lr)

    pol_optim = pol_optimizer.minimize(pol_loss)
    pol_grad = pol_optimizer.compute_gradients(pol_loss)

    return pol_optim, pol_grad
