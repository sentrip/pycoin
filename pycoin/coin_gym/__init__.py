import warnings
import gym
from .env import CoinGym

try:
    gym.envs.register(id='CoinGym-v0', entry_point='coin_gym.env:CoinGym')
except gym.error.Error:
    pass


def make_gym(log_level='CRITICAL'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gym.logger.setLevel(log_level)
        return gym.make('CoinGym-v0')
