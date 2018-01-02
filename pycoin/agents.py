import json
import os
import random

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


class BaseAgent(ABC):
    ##################################
    # REQUIRED OVER-WRITABLE METHODS #
    ##################################

    @abstractmethod
    def act(self, state, deterministic=True, filtered=False):
        raise NotImplementedError

    ##################################
    # OPTIONAL OVER-WRITABLE METHODS #
    ##################################

    def __init__(self, max_pos=5):
        self.pos = 0
        self.max_pos = max_pos

    def filter(self, action):
        if action == 1 and self.pos == self.max_pos:
            return 0
        elif action == 2 and self.pos == -self.max_pos:
            return 0
        self.pos += 1 if action == 1 else -1 if action == 2 else 0
        return action

    def preprocess_state(self, state):
        return state

    def reset(self):
        self.pos = 0


class RLAgent(BaseAgent):
    episodes = 0
    episode_reward = 0.
    discount = 0.99

    ##################################
    # REQUIRED OVER-WRITABLE METHODS #
    ##################################

    @abstractmethod
    def build_model(self):
        raise NotImplementedError("You forgot to build your model!")

    @abstractmethod
    def exploration_probability(self):
        raise NotImplementedError('You did not implement exploration_probability')

    ##################################
    # OPTIONAL OVER-WRITABLE METHODS #
    ##################################

    def __init__(self,
                 session=None, checkpoint_path=None,
                 save_every=50, save_summaries=False,
                 memory_size=10000, batch_size=64, **kwargs):
        super(RLAgent, self).__init__(**kwargs)

        self.session = session
        self.save_every = save_every

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is None:
            self.model_path = None
        else:
            self.model_path = os.path.join(checkpoint_path, 'model')
            if not os.path.isdir(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)

        self.callbacks = []
        if save_summaries and isinstance(self, RLAgent):
            self.callbacks.append(TensorBoard(log_dir=self.checkpoint_path))

        self.model = self.build_model()
        self.session.run(tf.global_variables_initializer())

    def reset(self):
        super(RLAgent, self).reset()
        self.episode_reward = 0.

    def load(self):
        if self.model_path:
            if os.path.isfile(self.model_path + '.hdf5'):
                self.model.load_weights(self.model_path + '.hdf5')

    def save(self):
        if self.model_path:
            self.model.save_weights(self.model_path + '.hdf5')

    def predict(self, states):
        return self.session.run(self.model.output, feed_dict={self.model.input: states})

    def act(self, state, deterministic=True, filtered=False):
        if not deterministic and self.exploration_probability > random.random():
            action = random.randint(0, self.model.output.shape[-1] - 1)
        else:
            action = np.argmax(self.predict([self.preprocess_state(state)])[0])
        return self.filter(action) if filtered else action

    def observe(self, state, action, reward, next_state, terminal=False):
        state, next_state = self.preprocess_state(state), self.preprocess_state(next_state)
        self.memory.append((state, action, reward, next_state, terminal))
        self.episode_reward += reward
        if terminal:
            self.train()
            self.episodes += 1
            if self.episodes % self.save_every == 0:
                self.save()

    def train(self):
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = list(zip(*mini_batch))
        y_target = self.predict(states)
        next_predictions = self.predict(next_states)
        for i, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if not done:
                reward += self.discount * np.max(next_predictions[i])
            y_target[i, action] = reward
        self.model.fit(np.array(states), np.array(y_target), batch_size=len(mini_batch),
                       verbose=0, callbacks=self.callbacks)


class TensorForceAgent(RLAgent):
    Agent = None
    kwargs = None

    _viable_agents = [
        'VPGAgent',
        'TRPOAgent',
        'PPOAgent',
        'DQNAgent',
        'DDQNAgent',
        'DQNNstepAgent',
        'DQFDAgent',
        'NAFAgent'
    ]

    def __init__(self, **kwargs):
        super(TensorForceAgent, self).__init__(**kwargs)
        if isinstance(self.kwargs, str):
            with open(self.kwargs) as f:
                self.kwargs = json.load(f)  # Load from json file if string
        elif not isinstance(self.kwargs, dict):
            raise ValueError('self.kwargs must be a dict or a string')

        assert self.Agent is not None, 'Missing Agent, try one of these: %s' % str(self._viable_agents)
        assert 'states_spec' in self.kwargs, 'Missing states_spec'
        assert 'actions_spec' in self.kwargs, 'Missing actions_spec'
        assert 'network_spec' in self.kwargs, 'Missing network_spec'

        if kwargs.get('summary_spec', None) is not None:
            self.kwargs.update(
                {'summary_spec': {
                    'directory': self.checkpoint_path, 'steps': 50,
                    'labels': ['losses', 'total-loss', 'rewards']}})

        self._agent = self.Agent(**self.kwargs)

    def build_model(self):
        return

    @property
    def exploration_probability(self):
        return 0

    def train(self):
        pass

    def load(self):
        if self.model_path:
            self._agent.restore_model(file=self.model_path)

    def save(self):
        if self.model_path:
            self._agent.save_model(file=self.model_path, append_timestep=False)

    def reset(self):
        super(RLAgent, self).reset()
        self.episode_reward = 0.
        self._agent.reset()

    def act(self, state, deterministic=True, filtered=False):
        action = self._agent.act(self.preprocess_state(state), deterministic=deterministic)
        return self.filter(action) if filtered else action

    def observe(self, state, action, reward, next_state, terminal=False):
        super(TensorForceAgent, self).observe(state, action, reward, next_state, terminal=terminal)
        self._agent.observe(reward=reward, terminal=terminal)


# TODO WIP
# class ActorCriticAgent(RLAgent):
#     actor = None
#     critic = None
#     critic_action_input = None
#     critic_state_input = None
#
#     def __init__(self, lr=1e-3, **kwargs):
#         super(ActorCriticAgent, self).__init__(**kwargs)
#
#         self.learning_rate = lr
#
#         self.actor_critic_grad = tf.placeholder(tf.float32, [None, 2])
#
#         actor_weights = self.actor.trainable_weights
#         self.actor_grads = tf.gradients(self.actor.output, actor_weights, -self.actor_critic_grad)
#         grads = zip(self.actor_grads, actor_weights)
#         self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
#
#         self.critic_grads = tf.gradients(self.critic.output, self.critic_action_input)
#
#     @abstractmethod
#     def build_actor(self):
#         raise NotImplementedError
#
#     @abstractmethod
#     def build_critic(self):
#         raise NotImplementedError
#
#     def build_model(self):
#         self.actor = self.build_actor()
#         self.critic = self.build_critic()
#
#     def predict(self, states):
#         return self.session.run(self.actor.output, feed_dict={self.actor.input: states})
#
#     def train_actor(self, samples):
#         states, actions, rewards, next_states, _ = list(zip(*samples))
#         states = np.array(states)
#         predictions = self.predict(states)
#         grads = self.session.run(self.critic_grads,
#                                  feed_dict={self.critic_state_input: states,
#                                             self.critic_action_input: predictions})[0]
#         self.session.run(self.optimize,
#                          feed_dict={self.actor.input: states,
#                                     self.actor_critic_grad: grads})
#
#     def train_critic(self, samples):
#         states, actions, rewards, next_states, dones = list(zip(*samples))
#         rewards = list(rewards)
#         predictions = self.predict(next_states)
#         future_rewards = self.critic.predict([np.array(next_states), predictions])
#         y_target = np.zeros([len(samples), self.actor.output.shape[-1]])
#         for i, (action, reward, future_reward, done) in enumerate(zip(actions, rewards, future_rewards, dones)):
#             rewards[i] += self.discount * np.max(future_reward)
#             y_target[i][action] = 1
#         self.critic.fit([np.array(states), y_target], np.array(rewards), verbose=0)
#
#     def train(self):
#         mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
#         self.train_critic(mini_batch)
#         self.train_actor(mini_batch)

