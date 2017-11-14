from threading import Thread
import unittest

from pycoin.coin_gym.market import Market
from tests import connect_and_send


class TestCaseMarket(unittest.TestCase):

    def setUp(self):
        self.sender = Thread(target=connect_and_send)
        self.sender.start()
        self.market = Market()

    def tearDown(self):
        self.sender.join()

    def test_step(self):
        price = self.market.step()
        assert price == self.market.price, "Returned price does not match current price"
        assert price == self.market.data[-1, 3], "Returned price does not match market data"

    def test_reset(self):
        self.market.reset()
        assert self.market.data[-self.market.minimum_required_length, 0] != 0, \
            "Data not gathered"
        assert self.market.data[-(self.market.minimum_required_length + 1), 0] == 0, \
            "More data points gathered than required"
        assert round(self.market.initial_price, 1) == self.market.__dict__.get('start', None) or 50, \
            'Initial price was not reset, %.2f, 50' % self.market.initial_price

    def test_prices(self):
        l = len(self.market.prices(100))
        assert l == self.market.backtrack, "prices returned incorrect default array, %d" % l
        assert all(i == self.market.initial_price for i in self.market.prices(100)), \
            'prices in default array are not all initial price'
        self.market.reset()
        l = len(self.market.prices(100))
        assert l == 100, "prices returned an array with incorrect length, %d, %d" % (l, 100)

    def test_sma(self):
        s1, s2 = self.market.sma_crossover
        assert s1 == s2 == self.market.initial_price
        self.market.reset()
        for _ in range(50):
            self.market.step()
            s1, s2 = self.market.sma_crossover
            assert s1 != s2, 'SMAs are the same'
            assert s1 > 0 and s2 > 0, 'SMAs are not both > 0, %.2f, %.2f' % (s1, s2)
            assert s1 != self.market.initial_price != s2, 'Averages are initial prices after reset'
