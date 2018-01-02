import random
import gym.spaces
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import create_trading_figure


class Plotter:
    price_plot = None
    target_box = None
    trade_plot_buy = None
    trade_plot_sell = None

    def __init__(self, env):
        self.env = env
        self.fig, self.ax = create_trading_figure([0.1, 0.1, 0.8, 0.75], 'Step', 'Price ( $ )')
        self.plot_trading()
        self.reset_axis()
        self.set_title()
        plt.ion()
        plt.show()

    @property
    def x_axis(self):
        return list(range(len(self.env.price[:self.env.step_ + self.env.history_length])))

    @property
    def prices(self):
        return self.env.price[:self.env.step_ + self.env.history_length]

    @property
    def patch_start_end(self):
        return self.env.step_, self.env.step_ + self.env.history_length - 0.5

    @property
    def xaxis_limit(self):
        return 0, len(self.env.price[:self.env.step_ + self.env.history_length]) + 200

    @property
    def title_text(self):
        width = 75
        header = ('{:%ds}{:6s}{:%ds}\n' % (width, width - 6)).format(
            '', self.env.df_sample['symbol'].iloc[0], '')
        pos = 'Position             : {:1s}'.format('%d' % int(self.env.pos[self.env.step_ + self.env.history_length - 1]))
        tot = 'Total reward     : {:10s}'.format('%.2f' % self.env.total_reward)
        cur = 'Current reward : {:10s}'.format('%.2f' % self.env.obs_reward.sum())
        return header + pos + self.blank(pos, width) + tot + self.blank(tot, width) + cur + self.blank(cur, width)

    @staticmethod
    def blank(like_var, width):
        return ' ' * max(0, width - len(like_var)) + '\n'

    def trade_color(self, ind,
                    long=(1, 0, 0, 0.5), long_cover=(1, 1, 1, 0.5), short=(0, 1, 0, 0.5), short_cover=(1, 1, 1, 0.5)):
        if self.env.pos_variation[ind] > 0 and self.env.pos_entry_cover[ind] > 0:
            return long
        elif self.env.pos_variation[ind] > 0 > self.env.pos_entry_cover[ind]:
            return long_cover
        elif self.env.pos_variation[ind] < 0 < self.env.pos_entry_cover[ind]:
            return short
        elif self.env.pos_variation[ind] < 0 and self.env.pos_entry_cover[ind] < 0:
            return short_cover

    def reset_axis(self):
        self.ax.set_xlim(*self.xaxis_limit)

    def plot_patch(self, start, end,
                   pos_colors=((0., 0.95, 0., 0.8), (0., 1., 0., 0.5)),
                   neg_colors=((0.95, 0., 0., 0.8), (1., 0., 0., 0.5))):
        mx, mn = self.env.obs_price.max(), self.env.obs_price.min()
        ec, fc = pos_colors if self.env.obs_price[-1] >= mx - (mx - mn) / 2 else neg_colors
        self.target_box = self.ax.add_patch(patches.Rectangle((start, mn), end - start, mx - mn, edgecolor=ec,
                                                              facecolor=fc, linestyle='-', linewidth=1.5, fill=True))

    def plot_trading(self):
        self.price_plot = self.ax.plot(self.x_axis, self.prices, c=(0, 0.68, 0.95, 0.9), zorder=1)
        self.plot_patch(*self.patch_start_end)
        trade_x = self.env.pos_variation.nonzero()[0]
        trade_x_buy = [i for i in trade_x if self.env.pos_variation[i] > 0]
        trade_x_sell = [i for i in trade_x if self.env.pos_variation[i] < 0]
        self.trade_plot_buy = self.ax.scatter(x=[self.x_axis[i] for i in trade_x_buy],
                                              y=[self.env.price[i] for i in trade_x_buy],
                                              c=[self.trade_color(i) for i in trade_x_buy],
                                              s=100, marker='^', edgecolor=(1, 0, 0, 0.9), zorder=2)
        self.trade_plot_sell = self.ax.scatter(x=[self.x_axis[i] for i in trade_x_sell],
                                               y=[self.env.price[i] for i in trade_x_sell],
                                               c=[self.trade_color(i) for i in trade_x_sell],
                                               s=100, marker='v', edgecolor=(0, 1, 0, 0.9), zorder=2)

    def set_title(self):
        self.fig.suptitle(self.title_text, x=0.1,
                          horizontalalignment='left',
                          fontsize=16)

    def plot(self):
        for line in self.price_plot:
            self.ax.lines.remove(line)
        self.target_box.remove()
        self.trade_plot_buy.remove()
        self.trade_plot_sell.remove()
        self.plot_trading()
        self.reset_axis()
        self.set_title()
        plt.pause(0.00001)


class Trading:
    _id = 'training-v1'
    plotter = Plotter
    first_render = True

    df = None
    step_ = 0
    df_sample = None
    price = None

    pos = None
    pos_variation = None
    pos_entry_cover = None
    price_mean = None
    reward = None
    reward_flux = None
    reward_make_real = None

    obs_features = None
    obs_state = None
    obs_pos = None
    obs_pos_var = None
    obs_pos_entry_cover = None
    obs_price = None
    obs_price_mean = None
    obs_reward = None
    obs_make_real = None

    chg_pos = None
    chg_pos_var = None
    chg_pos_entry_cover = None
    chg_price = None
    chg_price_mean = None
    chg_reward = None
    chg_reward_flux = None
    chg_make_real = None

    def __init__(self, data, *,
                 history_length=2, step_length=1,  fee=0., time_fee=0., max_position=5,
                 price_column_name='price',  feature_names=None):

        if isinstance(data, pd.DataFrame):
            data = [data]
        if feature_names is None:
            feature_names = data[0].columns

        for i in range(len(data)):
            if 'serial_number' not in data[i].columns:
                if 'live' in self._id:
                    data[i]['serial_number'] = pd.Series(range(len(data[i])), index=data[i].index)
                else:
                    data[i] = self.add_date_serial(data[i])
        self.data = data
        self.df = random.choice(self.data)
        self.history_length = history_length
        self.step_length = step_length

        self.price_name = price_column_name
        self.using_feature = [price_column_name] + [i for i in feature_names if i not in ['datetime', 'serial_number',
                                                                                          'symbol',  price_column_name]]
        self.fee = fee / 100
        self.time_fee = time_fee
        self.max_position = max_position
        self.total_reward = 0.

        self.begin_fs = self.df[self.df['serial_number'] == 0]
        self.date_length = len(self.begin_fs)
        self.feature_length = len(feature_names)
        
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = (self.history_length, len(self.using_feature) + 6)

    @staticmethod
    def add_date_serial(dataframe):
        day = dataframe['datetime'][0].day
        serials = []
        serial = 0
        index = 0
        while index < len(dataframe):
            if dataframe['datetime'][index].day > day:
                serial = 0
                day = dataframe['datetime'][index].day
            serials.append({'serial_number': serial})
            index += 1
            serial += 1
        return pd.concat([dataframe, pd.DataFrame(serials)], axis=1)

    def get_observation(self):
        s, e = self.step_, self.step_ + self.history_length
        self.obs_state = self.obs_features[s: e]
        self.obs_price_mean = self.price_mean[s: e]

        self.obs_pos = self.pos[s: e]
        self.obs_pos_var = self.pos_variation[s: e]
        self.obs_pos_entry_cover = self.pos_entry_cover[s: e]
        self.obs_make_real = self.reward_make_real[s: e]

        self.obs_price = self.price[s: e]
        self.obs_reward = self.reward[s: e]

    @property
    def state(self):
        return np.concatenate((
            self.obs_state,
            # Recent change ratio
            1 - self.obs_price[:, np.newaxis] / self.obs_price_mean[:, np.newaxis],
            # Position ratio
            self.obs_pos[:, np.newaxis] / self.max_position,
            self.obs_pos_var[:, np.newaxis],
            self.obs_pos_entry_cover[:, np.newaxis],
            self.obs_make_real[:, np.newaxis]
        ), axis=1)

    def random_sample(self):
        random_int = np.random.randint(self.date_length)
        if random_int == self.date_length - 1:
            begin_point = self.begin_fs.index[random_int]
            end_point = None
        else:
            begin_point, end_point = self.begin_fs.index[random_int: random_int + 2]
        return begin_point, end_point

    def _long(self, open_pos, enter_price, price_mean, mkt_position):
        if open_pos:
            self.chg_price_mean[:] = enter_price
            self.chg_pos[:] = 1
            self.chg_pos_var[:1] = 1
            self.chg_pos_entry_cover[:1] = 1
        else:
            after_act_mkt_position = mkt_position + 1
            self.chg_price_mean[:] = (price_mean * mkt_position + enter_price) / after_act_mkt_position
            self.chg_pos[:] = after_act_mkt_position
            self.chg_pos_var[:1] = 1
            self.chg_pos_entry_cover[:1] = 2

    def _short(self, open_pos, enter_price, price_mean, mkt_position):
        if open_pos:
            self.chg_price_mean[:] = enter_price
            self.chg_pos[:] = -1
            self.chg_pos_var[:1] = -1
            self.chg_pos_entry_cover[:1] = 1
        else:
            after_act_mkt_position = mkt_position - 1
            self.chg_price_mean[:] = (price_mean * abs(mkt_position) + enter_price) / abs(after_act_mkt_position)
            self.chg_pos[:] = after_act_mkt_position
            self.chg_pos_var[:1] = -1
            self.chg_pos_entry_cover[:1] = 2

    def _short_cover(self, price_mean, mkt_position):
        self.chg_price_mean[:] = price_mean
        self.chg_pos[:] = mkt_position + 1
        self.chg_make_real[:1] = 1
        diff = self.chg_price - self.chg_price_mean + 1e-30
        ratio_after_fee = (1 - (abs(diff) / diff) * self.fee)
        self.chg_reward[:] = diff / np.max(self.price) * ratio_after_fee * mkt_position #* \
                             #abs(mkt_position) * ratio_after_fee * self.chg_make_real
        self.chg_pos_var[:1] = 1
        self.chg_pos_entry_cover[:1] = -1

    def _long_cover(self, price_mean, mkt_position):
        self.chg_price_mean[:] = price_mean
        self.chg_pos[:] = mkt_position - 1
        self.chg_make_real[:1] = 1
        diff = self.chg_price - self.chg_price_mean + 1e-30
        ratio_after_fee = (1 - (abs(diff) / diff) * self.fee)
        self.chg_reward[:] = diff / np.max(self.price) * ratio_after_fee * mkt_position #* \
                             #abs(mkt_position) * ratio_after_fee * self.chg_make_real
        self.chg_pos_var[:1] = -1
        self.chg_pos_entry_cover[:1] = -1

    def _stay(self, price_mean, mkt_position):
        self.chg_pos[:] = mkt_position
        self.chg_price_mean[:] = price_mean

    def execute_action(self, mkt_position, price_mean, action):
        enter_price = self.chg_price[0]
        if action == 1 and self.max_position > mkt_position >= 0:
            open_pos = (mkt_position == 0)
            self._long(open_pos, enter_price, price_mean, mkt_position)

        elif action == 2 and -self.max_position < mkt_position <= 0:
            open_pos = (mkt_position == 0)
            self._short(open_pos, enter_price, price_mean, mkt_position)

        elif action == 1 and mkt_position == self.max_position:
            action = 0
        elif action == 2 and mkt_position == -self.max_position:
            action = 0

        elif action == 1 and mkt_position < 0:
            self._short_cover(price_mean, mkt_position)

        elif action == 2 and mkt_position > 0:
            self._long_cover(price_mean, mkt_position)

        if action == 0:
            if mkt_position != 0:
                self._stay(price_mean, mkt_position)

    def reset(self):
        self.df = random.choice(self.data)
        begin_point, end_point = self.random_sample()
        self.df_sample = self.df.iloc[begin_point: end_point]

        self.step_ = 0
        self.price = self.df_sample[self.price_name].as_matrix()

        self.obs_features = self.df_sample[self.using_feature].as_matrix()
        self.pos = np.zeros_like(self.price)
        self.pos_variation = np.zeros_like(self.pos)
        self.pos_entry_cover = np.zeros_like(self.pos)

        self.price_mean = self.df_sample[self.price_name].\
            rolling(center=False, window=self.history_length).mean().as_matrix()
        self.price_mean[:self.history_length - 1] = pd.Series(
            [self.price_mean[self.history_length - 1]] * (self.history_length - 1))
        self.reward_make_real = np.zeros_like(self.price)
        self.reward = np.zeros_like(self.price)

        self.get_observation()
        self.total_reward = 0.
        return self.state

    def step(self, action):
        price_mean = self.price_mean[self.step_ + self.history_length - 1]
        mkt_position = self.pos[self.step_ + self.history_length - 1]

        self.step_ += self.step_length
        self.get_observation()

        # change part
        self.chg_pos = self.obs_pos[-self.step_length:]
        self.chg_pos_var = self.obs_pos_var[-self.step_length:]
        self.chg_pos_entry_cover = self.obs_pos_entry_cover[-self.step_length:]
        self.chg_price = self.obs_price[-self.step_length:]
        self.chg_price_mean = self.obs_price_mean[-self.step_length:]
        self.chg_make_real = self.obs_make_real[-self.step_length:]
        self.chg_reward = self.obs_reward[-self.step_length:]

        done = False
        if self.step_ + self.history_length + self.step_length >= len(self.price):
            done = True
            action = -1
            if mkt_position != 0:
                self.chg_price_mean[:] = price_mean
                self.chg_pos[:] = 0
                self.chg_pos_var[:1] = -mkt_position
                self.chg_pos_entry_cover[:1] = -2
                self.chg_make_real[:1] = 1
                diff = self.chg_price - self.chg_price_mean + 1e-30
                ratio_after_fee = (1 - (abs(diff) / diff) * self.fee)
                self.chg_reward[:] = diff / np.max(self.price) * ratio_after_fee * mkt_position #* \
                                     #abs(mkt_position) * ratio_after_fee * self.chg_make_real

        self.execute_action(mkt_position, price_mean, action)
        reward = self.chg_reward.sum() - self.time_fee - (self.fee * int(action > 0))
        self.total_reward += reward
        return self.state, reward, done, {}

    def render(self):
        if 'backtest' in self._id:
            raise AttributeError('Cannot render a back-test')
        else:
            if self.first_render:
                self.plotter = self.plotter(self)
                self.first_render = False
            self.plotter.plot()
