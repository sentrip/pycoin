from multiprocessing.connection import Client
import random

import numpy as np
import bottleneck as bn


def sin_distribute(initial_price):
    t = 0
    dt = np.pi / np.random.randint(50, 500)
    while True:
        cv = np.sin(t) + np.sin(5 * t) * 0.3
        price = initial_price * (1 + 0.2 * cv)
        t += dt
        yield [price] * 4 + [50 * (1 + 0.5 * cv)]


class SinMarket:
    def __init__(self):
        self.start = random.random() * 2000 + 50
        self.gen = sin_distribute(self.start)

    def recv(self):
        return next(self.gen)


class Market:
    def __init__(self):
        try:
            self._in = Client(('localhost', 25000))
        except ConnectionRefusedError:
            self._in = SinMarket()

        self.history_length = 5000
        self.backtrack = 100    # number of ticks to look back for price indicators
        self.data = np.zeros([self.history_length, 5])
        self.initial_price = 0

        self.added_data_points = 0
        self.minimum_lengths = {
            'sma': 10
        }
        self.minimum_required_length = min(list(self.minimum_lengths.values()))

    @property
    def history(self):
        return self.data[-max(min(self.added_data_points, self.history_length), 1):, :]

    @property
    def price(self):
        return self.data[-1, 3]

    def prices(self, n):
        return np.concatenate([
            np.array([self.initial_price] * max(self.backtrack - len(self.history), 0)),
            self.data[-max(min(self.added_data_points, n), 1):, 3]
        ])

    def add(self, data):
        self.added_data_points += 1
        self.data = np.roll(self.data, -1, axis=0)
        for i, j in enumerate(data):
            self.data[self.history_length-1][i] = j

    def step(self):
        data = self._in.recv()
        self.add(data)
        return data[3]

    def reset(self):
        count = 0
        while self.added_data_points < self.minimum_required_length:
            price = self.step()
            if count == 0:
                self.initial_price = price
            count += 1

    ######################
    #  PRICE INDICATORS ##
    ######################

    @property
    def sma_crossover(self):
        if self.history.shape[0] < self.minimum_lengths['sma']:
            return self.price, self.price
        short = bn.nanmean(self.prices(int(self.backtrack / self.minimum_lengths['sma'])))
        long = bn.nanmean(self.prices(self.backtrack))
        return short, long
