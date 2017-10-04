# todo write stockgym tests so they work
# import time
# import gym
# import unittest
# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing.connection import Listener
# import numpy as np
#
#
# class TestGym(unittest.TestCase):
#     pool = ThreadPoolExecutor(1)
#
#     def acc(self):
#         for _ in range(3):
#             self.c = self.sender.accept()
#             print('cc')
#
#     def setUp(self):
#         self.data = np.array([0.5, 0.5, 0.5, 0.5, 150.] * 110).reshape(110, 5)
#         self.sender = Listener(('localhost', 6200), authkey=b'veryscrape')
#         self.c = None
#         self.pool.submit(self.acc)
#         time.sleep(0.1)
#         self.env = gym.make('StockGym-v0')
#         time.sleep(0.01)
#
#         self.c.send(self.data)
#         print('sent')
#         self.state = self.env.reset()
#         print('reset')
#
#     def tearDown(self):
#         self.env.close()
#         self.c.close()
#         self.sender.close()
#
#     def test_initialize_gym(self):
#         assert len(self.state) == 110 * self.env.n_company_metrics + self.env.n_portfolio_metrics, "State shape incorrect"
#         assert self.env.initial_balance == 100000, "Initial balance incorrect"
#
#     def test_100_steps_no_sell(self):
#         for _ in range(100):
#             self.c.send(self.data)
#             _ = self.env.step(int(len(self.env.ratios)/2))
#         assert self.env.balance == self.env.initial_balance, 'Balance was changed when it shouldnt be'
#         assert sum(self.env.portfolio) == 0, 'Stocks were bought when they shoudnt be'
#
#     def test_100_half_buy_half_sell(self):
#         i = 1
#         for _ in range(100):
#             self.c.send(self.data)
#             _ = self.env.step(int(len(self.env.ratios)/2) + i * 4)
#             i *= -1
#         ex = round(self.env.initial_balance * 0.999**100, -2)
#         assert round(self.env.balance, -2) == ex, 'Balance is not correct, {:.2f}, {:.2f}'.format(self.env.balance, ex)
#         assert sum(self.env.portfolio) == 0, 'Stocks remain in the account'
