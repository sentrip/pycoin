import random
from threading import Thread

import gym
import numpy as np
import scipy.signal
import tensorflow as tf

from stockgym.a3c import A3CNetwork


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Agent(Thread):
    def __init__(self, sess, trainer, coord, gamma):
        super(Agent, self).__init__()
        self.sess = sess
        self.coord = coord
        self.gamma = gamma
        state_dim = 9
        state_size = 111 * state_dim
        self.possible_actions = list(range(110 * state_dim))

        self.env = gym.make('StockGym-v0')
        self.local_AC = A3CNetwork(state_size, len(self.possible_actions), self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []

    def run_episode(self, el):
        state = self.env.reset()
        done = False
        c=0
        while not done:
            a_dist, value = self.sess.run([self.local_AC.policy, self.local_AC.value],
                                          feed_dict={self.local_AC.inputs: [state]})

            if np.random.random() < el:
                action = random.choice(self.possible_actions)
            else:
                action = np.random.choice(self.possible_actions, p=a_dist[0])

            next_state, reward, done, _ = self.env.step(action)
            if c % 1 == 0:
                print('{}, {:3d}, {:.2f}, {:.2f}'.format(self.env.id, action, reward, self.env.agent_value))
            self.observations.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.values.append(value)
            state = next_state
            c += 1

    def train(self):
        n = len(self.observations)
        observations = np.array(self.observations)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)

        discounted_rewards = discount(rewards, self.gamma)

        run_list = [self.local_AC.value_loss, self.local_AC.policy_loss,
                    self.local_AC.entropy, self.local_AC.grad_norms,
                    self.local_AC.var_norms, self.local_AC.apply_grads]

        feed_dict = {self.local_AC.target_v: values.reshape(-1),
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.rewards: discounted_rewards}

        value_loss, policy_loss, entropy, grads, var_norms, _ = self.sess.run(run_list, feed_dict=feed_dict)
        return value_loss / n, policy_loss / n, entropy / n, grads, var_norms

    def run(self):
        el = 0.99
        while not self.coord.should_stop():
            self.run_episode(el)
            if len(self.observations) > 0:
                _ = self.train()
            self.sess.run(self.update_local_ops)
            el *= 0.99
