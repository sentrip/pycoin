import os
import logging
from collections import deque
from contextlib import suppress

import matplotlib.pyplot as plt

from btfx_trader import Trader
from .envs import Trading, BackTest, LiveTrading
from .utils import strategy_performance

log = logging.getLogger(__name__)


def train(env, agent, min_avg_score=200, log_every=10, render=True):
    assert isinstance(env, Trading), '`Trading` environment required for `train`'
    with suppress(AttributeError):
        agent.load()

    # Training
    scores = deque(maxlen=100)
    avg_score = 0
    actions = [0 for _ in range(env.action_space.n)]
    while avg_score < min_avg_score or len(scores) < scores.maxlen:
        try:
            agent.reset()
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                agent.observe(state, action, reward, next_state, terminal=done)
                state = next_state
                actions[action] += 1

            scores.append(agent.episode_reward)
            avg_score = sum(scores) / len(scores)
            if agent.episodes % log_every == 0:
                log.info('Episode %6d - Actions: [ {} ] - Score: %.2f'.format(', '.join(['%.2f'] * env.action_space.n)),
                         agent.episodes, *[i/sum(actions) for i in actions], avg_score)
                actions = [0 for _ in range(env.action_space.n)]

        except KeyboardInterrupt:
            break

    if render:
        # Final rendering
        input('Model done training, Avg Score: %.2f - press enter to see result:' % avg_score)
        agent.training = False
        agent.reset()
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            env.render()


def backtest(env, agent, render=True):
    assert isinstance(env, BackTest), '`BackTest` environment required for `backtest`'
    with suppress(AttributeError):
        agent.load()

    while True:
        try:
            agent.reset()
            state = env.reset()
            done, info = False, []
            while not done:
                action = agent.act(state, filtered=True)
                state, reward, done, info = env.step(action)
            if render:
                strategy_performance(details=info)
                plt.close()
        except KeyboardInterrupt:
            break


def live(env, agent, live_training=False, real_trading=False, render=True):
    assert isinstance(env, LiveTrading), '`LiveTrading` environment required for `live`'
    symbol = env.df['symbol'].iloc[0]
    if real_trading:
        trader = Trader(os.environ['BITFINEX_KEY'], os.environ['BITFINEX_SECRET'])
    else:
        trader = type('', (), {'order': lambda *a, **kw: None})  # todo improve dummy trader

    with suppress(AttributeError):
        agent.load()

    try:
        agent.reset()
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, deterministic=True, filtered=True)

            if real_trading:
                order = env.trade_ratios[action] * env.trade_ratio(env.trade_ratios[action])
                if order != 0.:
                    trader.order(symbol, env._price, percentage=order, pad_price=0.001, wait_execution=False)
                    env.balance = trader.cached_wallet['usd']
                    env.coin = trader.cached_wallet[symbol.lower().replace('usd', '')]

            next_state, reward, done, _ = env.step(action)
            if live_training:
                agent.observe(state, action, reward, next_state, terminal=done)
            state = next_state

            if render:
                env.render()
    except KeyboardInterrupt:
        return
