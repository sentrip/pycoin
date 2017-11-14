import unittest
from threading import Thread
from tests import connect_and_send
import pycoin.coin_gym as g


class TestCaseEnv(unittest.TestCase):

    def setUp(self):
        self.sender = Thread(target=connect_and_send, kwargs={'offset': 0., 'ln': 2000})
        self.sender.start()
        self.env = g.make_gym()
        self.env.reset()

    def tearDown(self):
        while True:
            try:
                self.env.market._in.recv()
            except EOFError:
                break
        self.env.close()
        self.sender.join()

    def test_aaa_make_gym(self):
        pass  # gym created in setup as it is needed in all tests, this test fails first if something is up

    def test_anti_reward(self):
        assert self.env.anti_reward(1) > 0, 'Returned negative reward for doing nothing'
        assert self.env.anti_reward(2) > 0, 'Returned negative reward for buying'
        self.env.agent.coin = 10
        assert self.env.anti_reward(0) > 0, 'Returned negative reward for selling'

    def test_anti_reward_recent_buy(self):
        assert self.env.anti_reward(2) > 0, 'Returned negative anti-reward for buying'
        self.env.step(2)
        assert self.env.anti_reward(2) < 0, 'Returned positive anti reward for buying recently with empty balance'
        self.env.step(0)
        assert self.env.anti_reward(2) < 0, 'Returned positive anti reward for buying recently with full balance'

    def test_anti_reward_recent_sell(self):
        self.env.step(2)
        assert self.env.anti_reward(0) > 0, 'Returned negative anti-reward for selling'
        self.env.step(0)
        assert self.env.anti_reward(0) < 0, 'Returned positive anti reward for buying recently with no coin'
        self.env.step(2)
        assert self.env.anti_reward(0) < 0, 'Returned positive anti reward for buying recently with full coin'

    def test_anti_reward_trade_broke(self):
        self.env.step(2)
        self.env.step(2)
        assert self.env.anti_reward(2) < 0, 'Returned positive anti reward for buying with empty balance'
        self.env.step(0)
        self.env.step(0)
        assert self.env.anti_reward(0) < 0, 'Returned positive anti reward for selling with no coin'

    ################
    # REWARD TESTS #
    ################
    def test_reward_no_action(self):
        for _ in range(100):
            self.env.step(1)
            assert self.env.reward(1) == 0, 'Returned non zero reward for doing nothing'

    def test_reward_initial_buy_then_no_action(self):
        s, r, *_ = self.env.step(2)
        assert r >= 0, 'Returned negative reward for buying on rise'
        # Price rising and invested
        for _ in range(100 - self.env.market.minimum_required_length - 1):  # Not 100 because of reset
            s, r, *_ = self.env.step(1)
            assert r >= 0, 'Returned negative reward when did nothing and invested while price is rising'
        # Price falling -- above initial -- and invested
        self.step_n(100, 1)
        # for _ in range(100):
        #     s, r, *_ = self.env.step(1)
        #     # todo - Ideally this should be strictly less than 0, never equal or above
        #     print(r)
        #     assert -1 < r <= 0, \
        #         'Returned positive reward when invested and price is falling but above initial/previous'
        # Price falling -- below initial -- and invested
        for _ in range(100):
            s, r, *_ = self.env.step(1)
            assert r <= 0, 'Returned positive reward when did nothing and invested ' \
                           'while price is falling and below initial/previous'

        # Price rising -- below initial -- and invested
        for _ in range(100):
            s, r, *_ = self.env.step(1)
            assert r > -1,\
                'Returned large negative reward for waiting while price is rising, but below initial/previous'
            assert r <= 0, \
                'Returned positive reward for being invested while price is below previous/initial and rising'

    def step_n(self, n, action):
        for _ in range(n):
            s, r, *a = self.env.step(action)
            assert r != 10000, 'Returned super high reward'

    def test_reward_buy_high_sell_low(self):
        self.step_n(100 - self.env.market.minimum_required_length - 1 + 10, 1)
        self.env.step(2)
        reward = 0
        self.step_n(1, 1)
        for _ in range(100):
            _, r, *a = self.env.step(1)
            assert r != 0, 'Returned zero reward when losing money'
            assert r <= reward, 'Reward is not decreasing when price is dropping while invested'
            reward = min(r, reward)
        self.env.step(0)
        for _ in range(100):
            _, r, *a = self.env.step(1)
            assert r != 0, 'Returned zero reward when losing money'
            assert r <= reward, 'Reward is not decreasing when sold low, bought high'
            reward = min(r, reward)

    def test_reward_buy_low_sell_high(self):
        self.step_n(300 - self.env.market.minimum_required_length - 1 + 10, 1)
        self.env.step(2)
        reward = 0
        self.env.step(1)
        for _ in range(98):
            _, r, *a = self.env.step(1)
            assert r != 0, 'Returned zero reward when gaining money'
            assert r >= reward, 'Reward is not increasing when price is rising while invested'
            reward = max(r, reward)
        self.env.step(0)
        self.env.step(1)
        for _ in range(98):
            _, r, *a = self.env.step(1)
            assert r != 0, 'Returned zero reward when gaining money'
            assert r >= reward, 'Reward is not increasing when sold high, bought low'
            reward = max(r, reward)

    def test_reward_wears_off_buy_low_sell_high_then_no_action_once(self):
        self.step_n(300 - self.env.market.minimum_required_length - 1 + 10, 1)
        self.env.step(2)
        reward = 0
        self.env.step(1)
        for _ in range(98):
            _, r, *a = self.env.step(1)
            reward = max(r, reward)
        self.env.step(0)
        self.env.step(1)
        for _ in range(98):
            _, r, *a = self.env.step(1)
            reward = max(r, reward)

        for _ in range(300):
            _, r, *a = self.env.step(1)
            assert r >= 0, 'Returned negative reward when having gained money'
            assert r <= reward, 'Reward not decreasing a while after any money was made'
            reward = min(r, reward)

    def test_reward_wears_off_buy_low_sell_high_then_no_action_multiple(self):
        self.step_n(300 - self.env.market.minimum_required_length - 1 + 10, 1)
        reward = 0
        for _ in range(2):
            self.env.step(2)
            self.env.step(1)
            for _ in range(98):
                _, r, *a = self.env.step(1)
                reward = max(r, reward)
            self.env.step(0)
            self.env.step(1)
            for _ in range(98):
                _, r, *a = self.env.step(1)
                reward = max(r, reward)
            self.step_n(200, 1)

        for _ in range(300):
            _, r, *a = self.env.step(1)
            assert r >= 0, 'Returned negative reward when having gained money'
            assert r <= reward, 'Reward not decreasing a while after any money was made'
            reward = min(r, reward)

    def test_reward_mitigated_infinite_growth(self):
        self.step_n(300 - self.env.market.minimum_required_length - 1 + 10, 1)
        reward = 0
        rewards = []
        for _ in range(4):
            self.env.step(2)
            self.env.step(1)
            for _ in range(98):
                _, r, *a = self.env.step(1)
                reward = max(r, reward)
            self.env.step(0)
            self.env.step(1)
            for _ in range(98):
                _, r, *a = self.env.step(1)
                reward = max(r, reward)
            self.step_n(200, 1)
            rewards.append(reward)

        for i in range(1, 3):
            self.assertAlmostEqual(rewards[i + 1] / rewards[i], rewards[i] / rewards[i - 1], 2,
                                   'Did not return similar ratio of rewards for consecutive successful buy/sell epochs')
        assert reward <= 100, 'Returned more than 100 reward, unhealthy if continuous growth expected'
