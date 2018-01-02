import argparse
import logging
import os
import sys

from functools import partial

import requests
import tensorflow as tf

from .agents import RLAgent, TensorForceAgent
from .envs import Trading, BackTest, LiveTrading
from .runners import train, backtest, live
from .utils import get_agent, get_data

# Disable unnecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT_LOGGER = logging.getLogger()
ROOT_LOGGER.setLevel('CRITICAL')
for h in ROOT_LOGGER.handlers:
    h.setLevel('CRITICAL')
    ROOT_LOGGER.removeHandler(h)


def env_options(option_parser):
    option_parser.add_argument('agent_file', type=str)
    option_parser.add_argument('--force-name', type=str, default=None)
    option_parser.add_argument('--log-level', type=str, default='INFO')
    option_parser.add_argument('--history-length', type=int, default=2)
    option_parser.add_argument('--step-length', type=int, default=1)
    option_parser.add_argument('--max-position', type=int, default=5)
    option_parser.add_argument('--fee', type=float, default=0.1)
    option_parser.add_argument('--time-fee', type=float, default=0.)


def data_options(option_parser):
    option_parser.add_argument('--data', type=str, default='wavy')
    option_parser.add_argument('--period', type=str, default='d')


def get_symbols(sym):
    if sym is None:
        s = list(filter(lambda i: i not in ['BCCUSD', 'BT1USD', 'BT2USD'],
                        map(lambda q: q.upper(),
                            filter(lambda i: 'usd' in i,
                                   requests.get('https://api.bitfinex.com/v1/symbols').json()))))
    else:
        s = sym.split(',')
    return [i.upper() + ('' if 'usd' in i.lower() else 'USD') for i in s]


# Command line parsing
parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers()

# Training options
train_parser = sub_parsers.add_parser('train', help='Train an agent')
#   - Env options
train_parser.add_argument('--symbols', type=str, default=None)
train_parser.add_argument('--min-score', type=float, default=50000.)
data_options(train_parser)
env_options(train_parser)
#   - Agent options
train_parser.add_argument('--memory-size', type=int, default=10000)
train_parser.add_argument('--checkpoint-path', type=str, default=None)
train_parser.add_argument('--save-summaries', action='store_true')
train_parser.add_argument('--save-every', type=int, default=100)
train_parser.add_argument('--log-every', type=int, default=100)

# Testing options
test_parser = sub_parsers.add_parser('test', help='Back-test an agent')
test_parser.add_argument('--symbol', type=str, default='btc')
data_options(test_parser)
env_options(test_parser)

# Live-trading options
live_parser = sub_parsers.add_parser('live', help='Live trade an agent')
live_parser.add_argument('--symbol', type=str, default='btc')
live_parser.add_argument('-train', action='store_true')
live_parser.add_argument('-real-trading', action='store_true')
env_options(live_parser)

args = parser.parse_args()

# Setup logging
log = logging.getLogger('pycoin')
log.setLevel(args.log_level)
stream = logging.StreamHandler(stream=sys.stdout)
stream.setLevel(args.log_level)
fmt = logging.Formatter(fmt='[ %(asctime)-15s ] %(levelname)-5s: %(name)-15s: %(message)s',
                        datefmt='%D %H:%M:%S')
stream.setFormatter(fmt)
log.addHandler(stream)

# Env setup
assert len(sys.argv) > 2, \
    "Please provide environment type and agent file: python3 -m pycoin [TYPE] [AGENT_FILE] --args"
env_type = sys.argv[1]
env_kwargs = dict(
    history_length=args.history_length,
    step_length=args.step_length,
    max_position=args.max_position,
    fee=args.fee,
    time_fee=args.time_fee
)


# Data setup
_symbols = get_symbols(args.symbols if env_type == 'train' else args.symbol)
_data = get_data(env_type,
                 'historical' if env_type == 'live' else args.data,
                 period='d' if env_type == 'live' else args.period,
                 symbols=_symbols)


# Environment selection
if env_type == 'train':
    env = Trading(_data, **env_kwargs)
    runner = partial(train,
                     min_avg_score=args.min_score,
                     log_every=args.log_every)
else:
    if env_type == 'test':
        env = BackTest(_data, **env_kwargs)
        runner = backtest
    else:
        env = LiveTrading(_symbols[0], **env_kwargs)
        runner = partial(live,
                         live_training=args.train,
                         real_trading=args.real_trading)

# Agent setup
AgentClass = get_agent(args.agent_file, custom_class_name=args.force_name, max_pos=args.max_position)
session = None
train_kwargs = {}

# Tensorflow and RL setup
reinforcement_learning = any(i in AgentClass.func.__bases__ for i in [RLAgent, TensorForceAgent])
if reinforcement_learning:
    batch_size = 64  # Todo get batch size from env
    session = tf.Session()
    train_kwargs.update(dict(
        session=session,
        checkpoint_path=args.checkpoint_path,
        save_every=args.save_every,
        save_summaries=args.save_summaries,
        memory_size=args.memory_size,
        batch_size=batch_size
    ))


# Create agent
agent = AgentClass(**train_kwargs)

# Run environment with agent
runner(env, agent)

# Cleanup
if reinforcement_learning:
    session.close()
