import unittest

from pycoin.envs import BackTest
from pycoin.gens import step_signal


class TestBasicUsage(unittest.TestCase):
    def setUp(self):
        self.period = 20
        df = step_signal(80, period_1=self.period)
        self.env = BackTest(df, fee=0)

    def test_order_no_fee(self):
        self.env.reset()
        for i in range(1, self.env.max_position + 1):
            self.env.order(1)
            expected_balance = self.env.value / 2 * (1 - i / self.env.max_position)
            expected_coin = (self.env.initial_value - expected_balance) / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after buy')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after buy')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

        for i in range(1, self.env.max_position * 2 + 1):
            self.env.order(-1)
            expected_balance = self.env.value / 2 / self.env.max_position * i
            expected_coin = (self.env.initial_value - expected_balance) / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after sell')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after sell')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

        for i in range(1, self.env.max_position + 1):
            self.env.order(1)
            expected_balance = self.env.value - self.env.value / 2 * (i / self.env.max_position)
            expected_coin = (self.env.initial_value - expected_balance) / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after buy')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after buy')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

    def test_order_fee(self):
        fee = 1
        self.env = BackTest(step_signal(80, period_1=100), fee=fee)
        self.env.initial_value = 2000000
        self.env.reset()
        expected_balance = self.env.balance
        expected_coin = self.env.coin
        for i in range(1, self.env.max_position + 1):
            old_balance = self.env.balance
            self.env.order(1)
            order_amount = abs(old_balance - self.env.balance)
            expected_balance -= order_amount
            expected_coin += order_amount * (1 - fee / 100) / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after buy')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after buy')
            self.assertLessEqual(self.env.value, self.env.initial_value, 'Value did not decrease or stay the same')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

        for i in range(1, self.env.max_position * 2 + 1):
            old_coin = self.env.coin
            self.env.order(-1)
            order_amount = abs(old_coin - self.env.coin) * self.env._price
            expected_balance += order_amount * (1 - fee / 100)
            expected_coin -= order_amount / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after sell')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after sell')
            self.assertLessEqual(self.env.value, self.env.initial_value, 'Value did not decrease or stay the same')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

        for i in range(1, self.env.max_position * 2 + 1):
            old_balance = self.env.balance
            self.env.order(1)
            order_amount = abs(old_balance - self.env.balance)
            expected_balance -= order_amount
            expected_coin += order_amount * (1 - fee / 100) / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after buy')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after buy')
            self.assertLessEqual(self.env.value, self.env.initial_value, 'Value did not decrease or stay the same')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

        for i in range(1, self.env.max_position * 2 + 1):
            old_coin = self.env.coin
            self.env.order(-1)
            order_amount = abs(old_coin - self.env.coin) * self.env._price
            expected_balance += order_amount * (1 - fee / 100)
            expected_coin -= order_amount / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after sell')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after sell')
            self.assertLessEqual(self.env.value, self.env.initial_value, 'Value did not decrease or stay the same')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

        for i in range(1, self.env.max_position + 1):
            old_balance = self.env.balance
            self.env.order(1)
            order_amount = abs(old_balance - self.env.balance)
            expected_balance -= order_amount
            expected_coin += order_amount * (1 - fee / 100) / self.env._price
            self.assertAlmostEqual(self.env.balance, expected_balance, 2, 'Incorrect balance after buy')
            self.assertAlmostEqual(self.env.coin, expected_coin, 8, 'Incorrect coin after buy')
            self.assertLessEqual(self.env.value, self.env.initial_value, 'Value did not decrease or stay the same')
            assert self.env.balance >= 0, 'Balance less than 0'
            assert self.env.coin >= 0, 'Coin less than 0'

    def test_reset(self):
        self.env.reset()
        assert self.env.balance == self.env.initial_value / 2
        assert self.env.coin == self.env.initial_value / 2 / self.env._price

    def test_step(self):
        pass  # todo write test_step - returns and details returned

    def test_transaction_details(self):
        pass  # todo write transaction details test
