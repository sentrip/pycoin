import tensorflow as tf
import tensorflow.contrib as ctb


class A3CNetwork:
    def __init__(self, state_size, n_classes, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            hidden1 = ctb.layers.fully_connected(self.inputs, 1024)
            #hidden2 = ctb.layers.fully_connected(hidden1, 2048)
            output = ctb.layers.fully_connected(hidden1, 1024)

            # Output layers for policy and value estimations
            self.policy = ctb.layers.fully_connected(output, n_classes, activation_fn=tf.nn.softmax)
            self.value = ctb.layers.fully_connected(output, n_classes)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, n_classes, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.rewards)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
