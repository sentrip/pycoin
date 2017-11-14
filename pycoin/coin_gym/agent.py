from collections import deque


class Agent:
    initial_balance = 100000.
    bankrupt_threshold = 0.8
    trade_ratios = [-1., 0., 1.]

    def __init__(self):
        self.coin = 0.
        self.balance = self.initial_balance
        self.balances = deque(maxlen=200)
        self.actions = deque(maxlen=1000)
        self.recent_actions = deque(maxlen=5)

        self.current_price = 0.

    def reset(self):
        self.coin = 0.
        self.balance = self.initial_balance
        self.balances = deque(maxlen=200)
        self.actions = deque(maxlen=1000)
        self.recent_actions = deque(maxlen=5)

    def order(self, order, price):
        self.current_price = price
        self.balances.append(self.value)
        n = int(((self.balance / price) if order > 0 else self.coin) * 10000) / 10000 * order
        action = order + 1. if n != 0 else 1.
        self.recent_actions.append(action)
        self.actions.append(action)
        self.balance -= price * n
        self.coin += n

    @property
    def value(self):
        """Current value of all investments and money"""
        return self.current_price * self.coin + self.balance

    @property
    def broke(self):
        """Current value of all investments and money are less than a threshold percentage of the initial balance"""
        return self.value < self.initial_balance * self.bankrupt_threshold

    @property
    def previous_value(self):
        """Agent value 200 ticks ago, normalized to initial balance (if investments go awry)"""
        return max(self.initial_balance, self.balances[0])

    @property
    def recently_bought(self):
        return 2. in self.recent_actions

    @property
    def recently_sold(self):
        return 0. in self.recent_actions

    @property
    def did_nothing(self):
        return all(i == 1 for i in self.actions) and len(self.actions) == self.actions.maxlen

