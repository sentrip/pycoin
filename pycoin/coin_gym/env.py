import gym
import gym.spaces
import numpy as np

from gym.utils import seeding

from .market import Market
from .agent import Agent


class CoinGym(gym.Env):
    def __init__(self):
        self.market = Market()
        self.agent = Agent()
        self.action_space = gym.spaces.Discrete(len(self.agent.trade_ratios))
        self.observation_space = gym.spaces.Box(
            np.zeros([len(self.state)]),
            np.array([np.finfo(np.float32).max] * len(self.state))
        )

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def state(self):
        return np.array([
            self.agent.balance, self.agent.coin,
            *self.market.history[-1],
            *self.market.sma_crossover
        ])

    def _reset(self):
        self.market.reset()
        self.agent.reset()
        return self.state

    def _step(self, action):
        anti_reward = self.anti_reward(action)
        self.agent.order(action - 1, self.market.price)
        self.market.step()
        reward = self.reward(action)
        final_reward = min(anti_reward, reward)
        return self.state, final_reward, self.agent.broke, {}

    def anti_reward(self, action):
        if self.agent.broke:
            return -100
        if action == 2:
            if self.agent.balance < self.market.price:
                return -0.1
            elif self.agent.recently_bought:
                return -0.02
        elif action == 0:
            if self.agent.coin < 0.0001:
                return -0.1
            elif self.agent.recently_sold:
                return -0.02
        elif action == 1:
            if self.agent.did_nothing:
                return -10
        return 10000

    def reward(self, action):
        if self.agent.last_investment['order']:
            reward = self.agent.value / self.agent.last_investment['value'] - 1
            if action == 0:
                reward *= 10
        else:
            reward = 0
        return round(reward, 3)
