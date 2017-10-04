from threading import Thread

import gym
import numpy as np
import scipy.misc
import scipy.signal
import tensorflow as tf
from stockgym.ac3 import AC3Network


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Agent(Thread):
    def __init__(self, name, state_size, n_classes, trainer, model_path):
        super(Agent, self).__init__()
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("bin/train_" + str(self.number))

        self.rewards_plus = None
        self.value_plus = None

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC3Network(state_size, n_classes, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.actions = list(range(330))
        self.env = None
        self.work_args = []

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss, self.local_AC.policy_loss,
                                               self.local_AC.entropy, self.local_AC.grad_norms,
                                               self.local_AC.var_norms, self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def run(self):
        self.env = gym.make('StockGym-v0')
        self.env.reset()
        max_episode_length, gamma, sess, coord, saver = self.work_args
        total_steps = 0
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                state = self.env.reset().reshape(-1, 999)
                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, value = sess.run([self.local_AC.policy, self.local_AC.value],
                                             feed_dict={self.local_AC.inputs: state})

                    action = np.random.choice(a_dist[0], p=a_dist[0])
                    action = int(np.argmax(a_dist == action))
                    next_state, reward, done, _ = self.env.step(self.actions[action])
                    # reward /= 100.0
                    next_state = next_state.reshape(-1, 999)
                    print(value)
                    episode_buffer.append([state, action, reward, next_state, done, value[0][action]])
                    episode_values.append(value[0][action])

                    episode_reward += reward
                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not done and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, feed_dict={self.local_AC.inputs: state})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if done:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)
                #
                # # Periodically save gifs of episodes, model parameters, and summary statistics.
                # if episode_count % 5 == 0 and episode_count != 0:
                #     if episode_count % 250 == 0 and self.name == 'worker_0':
                #         saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                #         print("Saved Model")
                #
                #     mean_reward = np.mean(self.episode_rewards[-5:])
                #     mean_length = np.mean(self.episode_lengths[-5:])
                #     mean_value = np.mean(self.episode_mean_values[-5:])
                #     summary = tf.Summary()
                #     summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                #     summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                #     summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                #     summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                #     summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                #     summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                #     summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                #     summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                # episode_count += 1

    def run_work(self, max_episode_length, gamma, sess, coord, saver):
        self.work_args = max_episode_length, gamma, sess, coord, saver
        self.start()
