import pandas as pd

from .training import Trading


class BackTest(Trading):
    _id = 'backtest-v1'
    initial_value = 2e3
    trade_ratios = [0., 1., -1.]

    def __init__(self, data, **kwargs):
        assert isinstance(data, pd.DataFrame) or len(data) == 1, 'Only a single product can be traded in back testing'
        if not isinstance(data, pd.DataFrame):
            data = data[0]
        super(BackTest, self).__init__(data, **kwargs)

        self.amount = 0
        self._price = data['price'].iloc[0]
        self.balance = self.initial_value / 2
        self.coin = self.initial_value / 2 / self._price
        self.initial_price = self._price
        self.transaction_details = []

    def random_sample(self):
        return None, None  # Selects all data (no random sampling for back test)

    def reset(self):
        self.amount = 0
        self.transaction_details = []
        state = super(BackTest, self).reset()
        self._price = self.obs_price[-1]
        self.balance = self.initial_value / 2
        self.coin = self.initial_value / 2 / self._price
        self.initial_price = self._price
        return state

    def order(self, order):
        if abs(self.obs_pos[-1] + order) > self.max_position:
            order = 0
        trade_value = self.value / self.max_position / 2
        n = trade_value / self._price * order
        trade_cost = n * self._price * self.fee * order
        buy_fee = trade_cost * int(order < 0)
        sell_fee = trade_cost / self._price * int(order > 0)
        self.balance = max(self.balance - self._price * n - buy_fee, 0)
        self.coin = max(self.coin + n - sell_fee, 0)
        self.amount = n - sell_fee * self._price

    @property
    def value(self):
        return self._price * self.coin + self.balance

    def step(self, action):
        previous_value = self.value
        previous_price = self._price
        self.order(self.trade_ratios[action])
        symbol = self.df_sample['symbol'].iloc[0]
        details = {'amount': self.amount, 'price': self._price, 'symbol': symbol,
                   symbol: self.coin * self._price, 'cash': self.balance}
        state, reward, done, _ = super(BackTest, self).step(action)
        self._price = self.obs_price[-1]
        _return = (self.value - previous_value) / previous_value
        factor_return = (self._price - previous_price) / previous_price
        details.update({'return': _return, 'factor_return': factor_return})
        self.transaction_details.append(details)
        info = {} if not done else pd.DataFrame(self.transaction_details,
                                                index=self.df_sample['datetime'][-len(self.transaction_details):])
        return state, reward, done, info
