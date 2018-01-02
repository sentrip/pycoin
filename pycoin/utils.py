import importlib
import os
import pyclbr
import random

from collections import OrderedDict
from functools import partial

import pyfolio.timeseries as timeseries
import pyfolio.utils as utils
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from empyrical import cum_returns

from .gens import step_signal, wavy_signal, historical


STAT_FUNCS_PCT = [
    'Hourly return', 'Cumulative returns', 'Hourly volatility', 'Max drawdown', 'Tick value at risk', 'Tick turnover'
]


def plot_data(data, points_to_show=500):
    if points_to_show >= len(data):
        start, end = None, None
    else:
        start = random.randint(0, len(data) - points_to_show - 1)
        end = start + points_to_show
    x = [i.to_pydatetime() for i in data['datetime'][start: end].tolist()]
    y = data['price'][start: end].tolist()
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()
    fmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.show()


def create_trading_figure(axis_bounds, xlabel, ylabel):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes(axis_bounds)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.grid(color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=15)
    return fig, ax


def plot_returns(returns, factor_returns, transactions=None):
    fig, ax = create_trading_figure([0.1, 0.2, 0.8, 0.7], 'DateTime', 'Return ( % )')
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt='%d-%m-%Y %H:%M'))
    fig.autofmt_xdate()

    returns = cum_returns(returns, starting_value=1) * 100
    factor_returns = cum_returns(factor_returns, starting_value=1) * 100
    x = [i.to_pydatetime() for i in returns.index]
    ex_x = [x[0]] + x + [x[-1]]
    ex_returns = pd.Series([100]).append(returns.append(pd.Series([100])))
    ex_factor_returns = pd.Series([100]).append(factor_returns.append(pd.Series([100])))
    ax.fill(ex_x, ex_returns, color=(0, 1, 0, 0.3), zorder=0)
    ax.fill(ex_x, ex_factor_returns, color=(1, 0, 0, 0.3), zorder=0)
    ax.plot(x, returns, color='g', zorder=2)
    ax.plot(x, factor_returns, color='r', zorder=1)

    ax.plot(x, [100] * len(x), color='w', zorder=2, linewidth=3)

    if transactions is not None:
        buys_sells = transactions.iloc[transactions['amount'].nonzero()[0]]
        buys = buys_sells['amount'][buys_sells['amount'] > 0]
        sells = buys_sells['amount'][buys_sells['amount'] < 0]
        ax.scatter(x=[i.to_pydatetime() for i in buys.index], y=[returns[i] for i in buys.index],
                   c=[(0, 1, 0, 0.5)] * len(buys), s=100, marker='v', edgecolor=(0, 1, 0, 0.9), zorder=2)
        ax.scatter(x=[i.to_pydatetime() for i in sells.index], y=[returns[i] for i in sells.index],
                   c=[(1, 0, 0, 0.5)] * len(buys), s=100, marker='^', edgecolor=(1, 0, 0, 0.9), zorder=2)
        fig.suptitle('Strategy returns %s\n%s to %s' %
                     (transactions['symbol'].iloc[0],
                      returns.index[0].strftime('%d-%m-%Y'), returns.index[-1].strftime('%d-%m-%Y')))
    else:
        fig.suptitle('Strategy returns\n%s to %s' %
                     (returns.index[0].strftime('%d-%m-%Y'), returns.index[-1].strftime('%d-%m-%Y')))

    ax.set_xlim(x[0], x[-1])
    d1, d2 = max(max(returns), max(factor_returns), 100) - 100, 100 - min(min(returns), min(factor_returns), 100)
    ax.set_ylim(100 - d2 * 1.1, 100 + d1 * 1.1)
    plt.show()


def strategy_performance(returns=None, factor_returns=None, positions=None, transactions=None, details=None):
    if details is not None:
        returns, factor_returns = details['return'], details['factor_return']
        positions, transactions = details[[details['symbol'].iloc[0], 'cash']], details[['symbol', 'price', 'amount']]
    new_perf_stats = OrderedDict()
    new_perf_stats['Backtest days'] = round(len(returns) / (360 * 24), 4)
    if transactions is not None:
        new_perf_stats['Trades/hour'] = round(len(transactions.iloc[transactions['amount'].nonzero()[0]]) / 360, 4)
    perf_stats_all = timeseries.perf_stats(
        returns, factor_returns=factor_returns, positions=positions, transactions=transactions)

    for k in perf_stats_all.keys():
        j = ('Hourly' + k.replace('Annual', '')) if 'Annual' in k \
            else ('Tick' + k.replace('Daily', '')) if 'Daily' in k else k
        new_perf_stats[j] = perf_stats_all[k]

    perf_stats = pd.DataFrame(pd.Series(new_perf_stats), columns=['Results'])
    for column in perf_stats.columns:
        for stat, value in perf_stats[column].iteritems():
            if stat in STAT_FUNCS_PCT:
                perf_stats.loc[stat, column] = str(round(value * 100, 1)) + '%'

    utils.print_table(perf_stats, fmt='{0:.2f}')
    plot_returns(returns, factor_returns,
                 transactions=transactions if new_perf_stats['Backtest days'] <= 0.05 else None)


def check_super(cls, class_name):
    for i in cls.super:
        try:
            if i.name == class_name:
                return cls
        except (TypeError, AttributeError, ValueError):
            pass
        if i == class_name:
            return cls
        elif not isinstance(i, (str, bytes, bytearray)):
            if len(i.super) > 0:
                return check_super(i, class_name)


def get_cls(module, class_name='Default'):
    d = pyclbr.readmodule(module)
    for k, v in d.items():
        r = check_super(v, class_name)
        if r is not None:
            return getattr(importlib.import_module(module), r.name)


def get_agent(agent_file, custom_class_name=None, **kwargs):
    agent_file = agent_file.split('/')[-1].replace('.py', '')

    current_path = os.getcwd()
    os.chdir(os.path.join(os.getcwd(), *[i for i in agent_file.split('/')[:-1] if i]))

    _agents = ['TensorForceAgent', 'RLAgent', 'BaseAgent']
    for agent_name in _agents:
        agent_class = get_cls(agent_file, class_name=agent_name)
        if agent_class is not None:
            break
    else:
        if custom_class_name is not None:
            agent_class = get_cls(agent_file, class_name=custom_class_name)
            if agent_class is not None:
                required_attrs = [
                    'act', 'reset', 'observe',
                    'max_pos', 'n_episodes', 'episode_reward',
                    'batch_size', 'session', 'checkpoint_path', 'save_every', 'memory_size'
                ]

                for attr in required_attrs:
                    assert hasattr(agent_class, attr), 'Custom agent did not implement `%s`' % attr
        else:
            raise AttributeError('Could not locate a viable agent in %s' % agent_file)
    os.chdir(current_path)

    return partial(agent_class, **kwargs)


def get_data(run_type, data_type, symbols=None, period=None):
    if data_type.startswith('h'):
        if run_type == 'test':
            symbols = symbols[:1]
        return [historical(i, period=period or 'd') for i in symbols]
    else:
        prices = np.geomspace(1, 15000, 50) + np.geomspace(1, 15000, 50) * 0.3 * (np.random.rand(50) * 2 - 1)
        prices[0], prices[-1] = max(0.5, prices[0]), min(prices[-1], 15000)
        if run_type == 'test':
            prices = [random.choice(prices)]
        if data_type.startswith('w'):
            kwargs_list = [dict(period_1=random.randint(50, 100),
                                period_2=random.randint(100, 200),
                                epsilon=0.05 + random.random() * (0.3 - 0.05),
                                volatility=0.01 + random.random() * (0.04 - 0.01)) for _ in range(len(prices))]
        else:
            kwargs_list = [dict(period_1=random.randint(30, 60),
                                volatility=0.01 + random.random() * (0.04 - 0.01)) for _ in range(len(prices))]
        f = wavy_signal if data_type.startswith('w') else step_signal
        data = list(map(lambda p, kwargs: f(p, **kwargs), prices, kwargs_list))
        return [i.iloc[-360:] for i in data] if period == 'h' else data
