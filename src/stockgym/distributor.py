import os
from multiprocessing.connection import Listener, Client

import numpy as np

linux_path, windows_path = "/home/djordje/veryscrape/vs/documents", "C:/users/djordje/desktop/lib/documents"
BASE_DIR = linux_path if os.path.isdir(linux_path) else windows_path


def load_query_dictionary(file_name):
    """Loads query topics and corresponding queries from disk"""
    queries = {}
    with open(os.path.join(BASE_DIR, '%s.txt' % file_name), 'r') as f:
        lns = f.read().splitlines()[10:13]
        for l in lns:
            x, y = l.split(':')
            queries[x] = y.split(',')
    return queries


class Distributor:
    def __init__(self, n_nodes=50):
        super(Distributor, self).__init__()
        self.n = n_nodes
        self.client = Client(('localhost', 6100), authkey=b'veryscrape')
        self.server = Listener(('localhost', 6200), authkey=b'veryscrape')
        self.types = ['article', 'blog', 'reddit', 'twitter', 'stock']

    def create_matrix(self, data):
        m = np.zeros([len(data['stock']), len(self.types)])
        for j, q in enumerate(data):
            for i, t in enumerate(data[q]):
                m[i][j] = data[q][t]
        return m

    def run(self):
        conns = []
        for _ in range(self.n):
            conns.append(self.server.accept())

        while True:
            data = self.client.recv()
            mat = self.create_matrix(data)
            for c in conns:
                c.send(mat)


class SinDistributor:
    def __init__(self, n_nodes=50):
        self.n = n_nodes
        self.gen = self.company_data_generator()
        self.server = Listener(('localhost', 6200), authkey=b'veryscrape')
        self.types = ['article', 'blog', 'reddit', 'twitter', 'stock']

    @staticmethod
    def sin_with_noise(t, off, n=0, step=0.01):
        ts = t + off - n * step
        main = np.sin(ts)
        sub = np.sin(5 * ts) * 0.3
        return main + sub

    def company_data_generator(self):
        companies = sorted(list(load_query_dictionary('query_topics.txt')))
        np.random.seed(10000)
        pairs = np.random.choice(np.array(companies), size=(2, 2), replace=False).tolist()
        prices = [[50. + np.random.rand(1)[0] * 450. for _ in range(2)] for _ in pairs]
        offsets = []
        for _ in pairs:
            offsets.append(np.random.rand(1)[0] * 4 * np.pi - 2 * np.pi)

        step = 0.01
        t = 0
        while True:
            data = {q: {} for q in ['article', 'blog', 'reddit', 'twitter', 'stock']}
            for price_pair, off, pair in zip(prices, offsets, pairs):
                for c, price in zip(pair, price_pair):
                    data['stock'][c] = self.sin_with_noise(t, off) * price * 0.1 + price
                    for qq, n in zip(['article', 'blog', 'reddit', 'twitter', 'stock'], [3, 5, 10, 20]):
                        data[qq][c] = self.sin_with_noise(t, off, n) * 0.25 + 0.5
            t += step
            yield data

    def create_matrix(self, data):
        m = np.zeros([len(data['stock']), len(self.types)])
        for j, q in enumerate(data):
            for i, t in enumerate(data[q]):
                m[i][j] = data[q][t]
        return m

    def run(self):
        conns = []
        for _ in range(self.n):
            conns.append(self.server.accept())

        while True:
            data = next(self.gen)
            mat = self.create_matrix(data)
            for c in conns:
                c.send(mat)
