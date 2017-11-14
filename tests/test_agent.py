import unittest

from pycoin.coin_gym.agent import Agent


class TestCaseAgent(unittest.TestCase):

    def setUp(self):
        self.price = 105
        self.agent = Agent()
        self.agent.reset()

    def test_order(self):
        self.agent.order(0, self.price)
        assert self.agent.balance == self.agent.initial_balance, \
            'Ordering nothing with full balance failed, %.1f' % self.agent.balance
        assert self.agent.balance >= 0, 'Agent balance fell below 0'
        assert self.agent.coin >= 0, 'Agent balance fell below 0'
        for _ in range(2):
            self.agent.order(1, self.price)
        assert self.agent.balance < self.price, 'Did not full balance worth of coin'
        assert self.agent.coin > int(self.agent.initial_balance / self.price), 'Did not buy all coin possible'
        assert self.agent.balance >= 0, 'Agent balance fell below 0'
        assert self.agent.coin >= 0, 'Agent balance fell below 0'
        self.agent.order(0, self.price)
        assert self.agent.balance < self.price, 'Ordering nothing with empty balance failed, %.1f' % self.agent.balance
        for _ in range(2):
            self.agent.order(-1, self.price)
        assert self.agent.balance > 99999, 'Balance not restored'
        assert self.agent.coin == 0., 'Coin did not return to 0'
        assert self.agent.balance >= 0, 'Agent balance fell below 0'
        assert self.agent.coin >= 0, 'Agent balance fell below 0'

    def test_recent_trade(self):
        self.agent.order(0, self.price)
        assert not self.agent.recently_bought, 'Recently bought true when agent didnt buy'
        assert not self.agent.recently_sold, 'Recently sold true when agent didnt sell'
        self.agent.order(1, self.price)
        assert self.agent.recently_bought, 'Recently bought false when agent bought'
        assert not self.agent.recently_sold, 'Recently sold true when agent didnt sell'
        for _ in range(5):
            self.agent.order(0, self.price)
        assert not self.agent.recently_bought, 'Recently bought true when agent didnt buy in last 5 ticks'
        self.agent.order(-1, self.price)
        assert not self.agent.recently_bought, 'Recently bought true when agent didnt buy'
        assert self.agent.recently_sold, 'Recently sold false when agent sold'
        for _ in range(5):
            self.agent.order(0, self.price)
        assert not self.agent.recently_sold, 'Recently sold true when agent didnt sell in last 5 ticks'
