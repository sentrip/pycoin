import random
import unittest

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from pycoin.envs import Trading
from pycoin.gens import step_signal


class TestBasicUsage(unittest.TestCase):
    def setUp(self):
        self.state_features = 4
        self.period = 10
        df = step_signal(50, period_1=self.period)
        self.env = Trading(df, fee=1)

    def test_add_date_serial(self):
        for _ in range(5):
            data = []
            serialized_data = []
            start_time = datetime.now() - timedelta(days=10)
            for day in range(5):
                for n in range(random.randint(1, 200)):
                    d = {
                        'datetime': start_time + timedelta(days=day) + timedelta(seconds=n * 100),
                        'price': 50 + random.random()
                    }
                    data.append(d)
                    q = d.copy()
                    q.update(serial_number=n)
                    serialized_data.append(q)
            df = pd.DataFrame(data)
            serial_df = pd.DataFrame(serialized_data)
            for row, (i, j) in enumerate(zip(self.env.add_date_serial(df), serial_df)):
                self.assertEqual(i, j, 'Row %d unequal, %s, %s' % (row, str(i), str(j)))

    def test_reset(self):
        # Test step before first reset
        with self.assertRaises(TypeError) as err:
            self.env.step(0)
        # Test first reset
        state = self.env.reset()
        for i in state:
            self.assertEqual(np.sum(i[-self.state_features:]), 0, 'Position and reward features non zero after reset')
        for _ in range(100):
            self.env.step(random.randint(0, 2))
        # Test reset after acting
        state = self.env.reset()
        for i in state:
            self.assertEqual(np.sum(i[-self.state_features:]), 0, 'Position and reward features non zero after reset')

    @staticmethod
    def assertAllInBetween(arr, mn, mx, msg):
        truth_arr = arr[arr >= mn] <= mx
        assert np.all(truth_arr), msg
        assert truth_arr.shape[0] == arr.shape[0], msg

    def test_step(self):
        self.env = Trading(step_signal(50, period_1=self.period), fee=1, history_length=20, max_position=50)
        state = self.env.reset()
        for _ in range(50):
            action = random.randint(0, 2)
            state, reward, _, _ = self.env.step(action)
            if action == 0:
                self.assertEqual(0., state[-1, -1], 'Position make real not 0 when action 0')
                self.assertEqual(0., state[-1, -2], 'Position entry cover not 0 when action 0')
                self.assertEqual(0., state[-1, -3], 'Position variation not 0 when action 0')
            elif action == 1:
                self.assertEqual(1., state[-1, -3], 'Position variation not 1 when action 1')
            elif action == 2:
                self.assertEqual(-1., state[-1, -3], 'Position variation not -1 when action 2')

            self.assertAllInBetween(state[:, -1], 0, 1, 'Position make real not between 0, 1')
            self.assertAllInBetween(state[:, -2], -2, 2, 'Position entry cover not between -2, 2')
            self.assertAllInBetween(state[:, -3], -1, 1, 'Position variation not between -1, 1')
            self.assertAllInBetween(state[:, -4], -1, 1, 'Position ratio not between -1, 1')
            self.assertAllInBetween(state[:, -5], -1, 1, 'Recent price change ration not between -1, 1')

        self.assertNotEqual(np.sum(np.sum(np.abs(state[:, self.state_features:]), axis=1)), 0,
                            'Position and reward features zero after acting')


class TestRewards(unittest.TestCase):
    def setUp(self):
        self.period = 10
        df = step_signal(50, period_1=self.period)
        self.env = Trading(df, fee=1, time_fee=0.1)

    def test_buy_low_sell_high(self):
        self.env.reset()
        for _ in range(10):
            state, reward, _, _ = self.env.step(0)
            assert reward == -self.env.time_fee, 'Non-zero reward for doing nothing'
        state, reward, _, _ = self.env.step(1)
        assert reward >= -self.env.fee - self.env.time_fee, 'Negative reward for buying'
        for _ in range(10):
            state, reward, _, _ = self.env.step(0)
            assert reward == -self.env.time_fee, 'Non-zero reward for doing nothing'
        state, reward, _, _ = self.env.step(2)
        assert reward >= -self.env.fee - self.env.time_fee, \
            'Negative reward for selling high after buying low'

    def test_buy_high_sell_low(self):
        self.env.reset()
        state, reward, _, _ = self.env.step(1)
        assert reward >= -self.env.fee - self.env.time_fee, \
            'Negative reward for buying'
        for _ in range(10):
            state, reward, _, _ = self.env.step(0)
            assert reward == -self.env.time_fee, 'Non-zero reward for doing nothing'
        state, reward, _, _ = self.env.step(2)
        assert reward <= 0, 'Positive reward for selling low after buying high'

    def test_sell_low_buy_high(self):
        self.env.reset()
        for _ in range(10):
            state, reward, _, _ = self.env.step(0)
            assert reward == -self.env.time_fee, 'Non-zero reward for doing nothing'
        state, reward, _, _ = self.env.step(2)
        assert reward >= -self.env.fee - self.env.time_fee, \
            'Negative reward for selling'
        for _ in range(10):
            state, reward, _, _ = self.env.step(0)
            assert reward == -self.env.time_fee, 'Non-zero reward for doing nothing'
        state, reward, _, _ = self.env.step(1)
        assert reward <= 0, 'Positive reward for buying high after selling low'

    def test_sell_high_buy_low(self):
        self.env.reset()
        state, reward, _, _ = self.env.step(2)
        assert reward >= -self.env.fee - self.env.time_fee, \
            'Negative reward for buying'
        for _ in range(10):
            state, reward, _, _ = self.env.step(0)
            assert reward == -self.env.time_fee, 'Non-zero reward for doing nothing'
        state, reward, _, _ = self.env.step(1)
        assert reward >= -self.env.fee - self.env.time_fee, \
            'Negative reward for buying low after selling high'

    def test_reward_constrained(self):
        self.env.reset()
        # Positive rewards
        for _ in range(10):
            self.env.step(0)

        for _ in range(self.env.max_position):
            state, reward, _, _ = self.env.step(1)
            assert 1 >= reward >= -self.env.fee - self.env.time_fee, \
                'Reward not between 0-1 for buying, %.2f' % reward

        for _ in range(10):
            state, reward, _, _ = self.env.step(0)

        for _ in range(self.env.max_position):
            state, reward, _, _ = self.env.step(2)
            assert 1 >= reward >= -self.env.fee - self.env.time_fee, \
                'Reward not between 0-1 for selling, %.2f' % reward

        for _ in range(10):
            state, reward, _, _ = self.env.step(0)

        state, reward, _, _ = self.env.step(2)
        assert 1 >= reward >= -self.env.fee - self.env.time_fee, \
            'Reward not between 0-1 for selling, %.2f' % reward

        for _ in range(10):
            state, reward, _, _ = self.env.step(0)

        state, reward, _, _ = self.env.step(1)
        assert 1 >= reward >= -self.env.fee - self.env.time_fee, \
            'Reward not between 0-1 for buying, %.2f' % reward

        # Negative rewards
        for _ in range(8):
            self.env.step(0)

        for _ in range(self.env.max_position):
            state, reward, _, _ = self.env.step(1)
            assert -1 <= reward <= 0, 'Reward not between -1-0 for buying, %.2f' % reward

        for _ in range(10):
            state, reward, _, _ = self.env.step(0)

        for _ in range(self.env.max_position):
            state, reward, _, _ = self.env.step(2)
            assert -1 <= reward <= 0, 'Reward not between -1-0 for selling, %.2f' % reward

        for _ in range(10):
            state, reward, _, _ = self.env.step(0)

        state, reward, _, _ = self.env.step(2)
        assert -1 <= reward <= 0, 'Reward not between -1-0 for selling, %.2f' % reward

        for _ in range(10):
            state, reward, _, _ = self.env.step(0)

        state, reward, _, _ = self.env.step(1)
        assert -1 <= reward <= 0, 'Reward not between -1-0 for buying, %.2f' % reward
