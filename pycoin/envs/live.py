from datetime import timedelta
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from btfx_trader import CoinAPI

from .training import Plotter
from .testing import BackTest
from ..gens import historical


class LivePlotter(Plotter):
    def __init__(self, env):
        super(LivePlotter, self).__init__(env)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt='%H:%M'))
        self.fig.autofmt_xdate()
        self.ax.set_xlabel('Time', fontsize=12, labelpad=15)
        self.start_plot = 0
        self.end_plot = None

    @property
    def x_axis(self):
        return [i.to_pydatetime() for i in self.env.df['datetime']]

    @property
    def prices(self):
        return self.env.price

    @property
    def patch_start_end(self):
        return mdates.date2num(self.x_axis[-max(self.env.step_length, 6)]), mdates.date2num(self.x_axis[-1]) #- 0.5

    @property
    def xaxis_limit(self):
        return pd.to_datetime(self.env.df['datetime'].iloc[0]), \
               pd.to_datetime(self.env.df['datetime'].iloc[-1] + timedelta(seconds=10))

    @property
    def title_text(self):
        width = 75
        header = ('{:%ds}{:6s}{:%ds}\n' % (width, width - 6)).format('', self.env.symbol, '')
        val = 'Value                : {:10s}'.format('%.2f' % self.env.value)
        bal = 'Balance            : {:10s}'.format('%.2f' % self.env.balance)
        cnn = 'Coin                  : {:10s}'.format('%.6f' % self.env.coin)
        return header + val + self.blank(val, width) + bal + self.blank(bal, width) + cnn + self.blank(cnn, width)


class LiveTrading(BackTest):
    _id = 'live-v1'
    plotter = LivePlotter
    api = CoinAPI  # For testing purposes

    def __init__(self, symbol, history_length=2, step_length=1, fee=0., max_position=5, time_fee=0.):
        self.symbol = symbol.upper() + ('' if symbol.endswith('USD') else 'USD')
        self.data_length = max(360, history_length * 3, step_length * 180)
        data = historical(self.symbol, period='d').iloc[-self.data_length:]
        super(LiveTrading, self).__init__(data, history_length=history_length, step_length=step_length,
                                          fee=fee, max_position=max_position, time_fee=time_fee)
        self.api = self.api(symbols=[self.symbol], data_types=['tickers'])
        self.api.connect()
        self.df = data

        self.obs_features = self.df[self.using_feature].as_matrix()
        self.price = self.df['price'].as_matrix()
        self.price_mean = self.df['price'].rolling(center=False, window=self.history_length).mean().as_matrix()
        self.reset()

    def get_observation(self):
        new_data = []
        for _ in range(self.step_length):
            d = self.api.get(self.api.symbols[0], 'tickers')
            d['datetime'], d['price'], d['symbol'] = d['time'], d['last_price'], self.symbol
            del d['time'], d['last_price']
            self._price = d['price']
            new_data.append(d)
        idx = list(range(self.df.last_valid_index()+1, self.df.last_valid_index() + self.step_length + 1))
        self.df = self.df.append(pd.DataFrame(new_data, index=idx)).iloc[-self.data_length:]
        self.obs_features = self.df[self.using_feature].as_matrix()
        self.price = self.df['price'].as_matrix()
        self.price_mean = self.df['price'].rolling(center=False, window=self.history_length).mean().as_matrix()

        self.pos = np.roll(self.pos, -self.step_length)
        self.pos_variation = np.roll(self.pos_variation, -self.step_length)
        self.pos_entry_cover = np.roll(self.pos_entry_cover, -self.step_length)
        self.reward = np.roll(self.reward, -self.step_length)
        self.reward_make_real = np.roll(self.reward_make_real, -self.step_length)

        self.pos[-self.step_length:] = self.chg_pos
        self.pos_variation[-self.step_length:] = self.chg_pos_var
        self.pos_entry_cover[-self.step_length:] = self.chg_pos_entry_cover
        self.reward[-self.step_length:] = self.chg_reward
        self.reward_make_real[-self.step_length:] = self.chg_make_real

        self.obs_state = self.obs_features[-self.history_length:]
        self.obs_price_mean = self.price_mean[-self.history_length:]
        self.obs_pos = self.pos[-self.history_length:]
        self.obs_pos_var = self.pos_variation[-self.history_length:]
        self.obs_pos_entry_cover = self.pos_entry_cover[-self.history_length:]
        self.obs_make_real = self.reward_make_real[-self.history_length:]
        self.obs_price = self.price[-self.history_length:]
        self.obs_reward = self.reward[-self.history_length:]

    def reset(self):
        self.pos = np.zeros_like(self.price)
        self.pos_variation = np.zeros_like(self.price)
        self.pos_entry_cover = np.zeros_like(self.price)
        # self.price_mean = np.zeros_like(self.price) todo check which of these is correct
        self.price_mean = self.df['price'].rolling(center=False, window=self.history_length).mean().as_matrix()

        self.reward = np.zeros_like(self.price)
        self.reward_make_real = np.zeros_like(self.price)

        self.chg_price = np.array(self.price[-self.step_length:])
        self.chg_price_mean = np.array(self.price_mean[-self.step_length:])
        self.chg_pos = np.zeros_like(self.chg_price)
        self.chg_pos_var = np.zeros_like(self.chg_price)
        self.chg_pos_entry_cover = np.zeros_like(self.chg_price)
        self.chg_reward = np.zeros_like(self.chg_price)
        self.chg_make_real = np.zeros_like(self.chg_price)

        self.obs_state = self.obs_features[-self.history_length:]
        self.obs_price = self.price[-self.history_length:]
        self.obs_price_mean = self.price_mean[-self.history_length:]
        self.obs_pos = np.zeros_like(self.obs_price)
        self.obs_pos_var = np.zeros_like(self.obs_price)
        self.obs_pos_entry_cover = np.zeros_like(self.obs_price)
        self.obs_reward = np.zeros_like(self.obs_price)
        self.obs_make_real = np.zeros_like(self.obs_price)

        self._price = self.obs_price[-1]
        self.balance = self.initial_value / 2
        self.coin = self.initial_value / 2 / self._price
        self.initial_price = self._price
        return self.state

    def step(self, action):
        self.chg_price = np.array(self.price[-self.step_length:])
        self.chg_price_mean = np.array(self.price_mean[-self.step_length:])
        self.chg_pos = np.zeros_like(self.chg_price)
        self.chg_pos_var = np.zeros_like(self.chg_price)
        self.chg_pos_entry_cover = np.zeros_like(self.chg_price)
        self.chg_reward = np.zeros_like(self.chg_price)
        self.chg_make_real = np.zeros_like(self.chg_price)

        price_mean = self.price_mean[-1]
        mkt_position = self.pos[-1]
        self.execute_action(mkt_position, price_mean, action)
        self.order(self.trade_ratios[action])
        self.get_observation()

        reward = self.chg_reward.sum() - self.time_fee - (self.fee * int(action > 0))
        self.total_reward += reward
        return self.state, reward, False, {}
