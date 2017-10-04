import random
from collections import deque
from functools import partial
from multiprocessing.connection import Client

import bottleneck as bn
import empyrical
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

"""
Add this to `gym/envs/__init__.py`:

register(
    id='StockGym-v0',
    entry_point='env.stock:StockGym',
)

"""


class StockGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.n_companies = 110
        self.initial_balance = 100000.
        self.bankrupt_threshold = 0.5
        self.history_length = 5000
        self.history = deque(maxlen=self.history_length)
        self.portfolio = np.array([0] * self.n_companies, dtype=np.int32)
        self.initial_prices = np.zeros([self.n_companies])
        self._returns = deque(maxlen=self.history_length)
        self._benchmark_returns = deque(maxlen=self.history_length)
        self.N = 50
        self.balance = 0.
        for _ in range(3):
            self._returns.append(0.)
            self._benchmark_returns.append(0.)
        # Buy/sell ratios
        self.ratios = [-1., 0.75, 0.5, 0.25, 0., 0.25, 0.5, 0.75, 1.]
        # (n_owned, price, red_sent, twit_sent, art_sent, blog_sent, long_SMA, lse_derivative, spline_est)
        self.n_company_metrics = 9
        # (balance, agent_value, market_value, max_drawdown, alpha, beta, calmar, sharpe, sortino)
        self.n_portfolio_metrics = 9
        # MOVING AVERAGES
        self.sma = partial(self.moving_average, 1)
        self.mma = partial(self.moving_average, 2)
        self.lma = partial(self.moving_average, 3)

        self.id = str(random.randint(10000000, 90000000))
        # Sell ratios for each company
        action_space_size = self.n_companies * len(self.ratios)
        self.action_space = spaces.Discrete(action_space_size)
        # +- infinity for all observations
        obs_space_size = np.array([np.finfo(np.float32).max] * action_space_size, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_space_size, obs_space_size)
        self._seed()

        self.incoming = Client(('localhost', 6200), authkey=b'veryscrape')

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.update_history()
        self.state = self.build_state()

        self.balance = self.initial_balance
        self.portfolio = np.array([0] * self.n_companies, dtype=np.int32)
        self.initial_prices = self.price_history[-1].flatten()

        self._returns = deque(maxlen=self.history_length)
        self._benchmark_returns = deque(maxlen=self.history_length)
        for _ in range(3):
            self._returns.append(0.)
            self._benchmark_returns.append(0.)

        return self.state

    def _step(self, action):
        print(self.id, action)
        # Act
        self.submit_order(action)
        # Observe
        self.update_history()

        self._returns.append(self.agent_return)
        self._benchmark_returns.append(self.benchmark_return)

        self.state = self.build_state()

        # Evaluate
        reward = self.agent_market_ratio
        return self.state, reward, self.agent_is_broke, {}

    def _close(self):
        self.incoming.close()

    def build_state(self):
        portfolio_metrics = np.array([
            self.balance, self.agent_value, self.market_value,
            self.alpha, self.beta, self.max_drawdown,
            self.downside_risk, self.sharpe_ratio, self.sortino_ratio]).reshape(-1, self.n_portfolio_metrics)
        calculated_metrics = self.history[-1]
        for i, f in enumerate([self.sma, self.mma, self.lma, self.least_square_derivative]):
            new = np.apply_along_axis(f, 0, np.arange(0, self.n_companies)).reshape(-1, 1)
            calculated_metrics = np.concatenate([calculated_metrics, new], axis=1)
        state = np.concatenate([portfolio_metrics, calculated_metrics], axis=0).flatten()
        return state

    def update_history(self):
        data = self.incoming.recv()
        self.history.append(data)

    def submit_order(self, action):
        """Submits a buy or send order, changing balance and stock number for requested company"""
        ind = int(action / len(self.ratios))
        order = self.ratios[action - ind * len(self.ratios)]
        # Commission - 0.1% of trade volume
        tc = 0.001
        current_price = self.history[-1][ind][4]
        if order > 0:
            commission = int(self.balance * order / current_price) * current_price * tc
            n = int((self.balance - commission) * order / current_price)
        else:
            n = int(self.portfolio[ind] * order)
        cost = current_price * n
        self.balance -= (cost + tc * abs(cost))
        self.portfolio[ind] += n

    @property
    def returns(self):
        return np.array(self._returns, dtype=np.float32)

    @property
    def benchmark_returns(self):
        return np.array(self._benchmark_returns, dtype=np.float32)

    @property
    def price_history(self):
        return np.array(self.history)[:, :, 4].reshape(-1, self.n_companies)

    @property
    def agent_value(self):
        """Current value of all investments and money"""
        ownership = self.price_history[-1].flatten() * self.portfolio
        return np.sum(ownership) * 0.999 + self.balance

    @property
    def agent_is_broke(self):
        """Current value of all investments and money are less than a threshold percentage of the initial balance"""
        return self.agent_value < self.initial_balance * self.bankrupt_threshold

    @property
    def market_value(self):
        """Average of inital_price/price ratios for all companies multiplied by initial investment"""
        return self.initial_balance * bn.nanmean(self.price_history[-1].flatten() / (self.initial_prices + 1e-7))

    @property
    def agent_market_ratio(self):
        """Scaling factor for agent's performence in comparison to the market"""
        return min(self.agent_return / self.benchmark_return, 1000)

    @property
    def agent_return(self):
        """Current returns of agent from entire portfolio as excess percentage"""
        return round(self.agent_value / self.initial_balance - self.returns[-1] - 1, 4) + 1e-9

    @property
    def benchmark_return(self):
        """Current returns of market as excess percentage"""
        return round(self.market_value / self.initial_balance - self.benchmark_returns[-1] - 1, 4) + 1e-9

    @property
    def alpha(self):
        """Current alpha value of portfolio"""
        return bn.nansum(self.returns - self.beta * self.benchmark_returns)

    @property
    def beta(self):
        """Current beta value (volatility) of portfolio"""
        joint = np.vstack([self.returns, self.benchmark_returns])
        cov = np.cov(joint, ddof=0)
        cov[1, 1] = max(cov[1, 1], 1e-30)
        return cov[0, 1] / cov[1, 1]

    @property
    def downside_risk(self):
        """Current downside risk of porfolio"""
        returns = self.returns.copy()
        returns[returns > 0] = 0
        return np.sqrt(bn.nansum(np.square(returns)))

    @property
    def max_drawdown(self):
        """Current maximum drawdown of portfolio"""
        cumulative = empyrical.cum_returns(self.returns, starting_value=100)
        max_return = np.fmax.accumulate(cumulative)
        max_return[np.round(max_return, 4) == 0] = 1e-7
        drawdown = (cumulative - max_return) / max_return
        return drawdown[-1]

    @property
    def sharpe_ratio(self):
        """Current risk-adjusted return ratio of portfolio"""
        return bn.nanmean(self.returns) / max(bn.nanstd(self.returns, ddof=1), 1e-7) * np.sqrt(self.returns.shape[0])

    @property
    def sortino_ratio(self):
        """Current sortino ratio of portfolio"""
        return bn.nansum(self.returns) / max(self.downside_risk, 1e-7)

    def moving_average(self, factor, company_ind):
        """Simple moving average for previous factor*N stock prices"""
        return bn.nansum(self.price_history[-self.N * factor:, company_ind], axis=0)

    def least_square_derivative(self, company_ind):
        """Derivative of linear least-squares estimates for previous N stock prices"""
        n, s_x, s_y, s_xy, s_x2, s_y2 = 0, 0, 0, 0, 0, 0
        for i, j in enumerate(reversed(self.price_history[-self.N:, company_ind])):
            if n >= self.N:
                break
            n, s_x, s_y, s_xy, s_x2, s_y2 = n + 1, s_x + i, s_y + j, s_xy + i * j, s_x2 + i ** 2, s_y2 + j ** 2
        div = n * s_x2 - s_x ** 2
        return (n * s_xy - s_x * s_y) / (div if round(div, 2) != 0 else 1e-3)
