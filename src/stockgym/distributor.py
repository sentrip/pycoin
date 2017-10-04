from multiprocessing.connection import Listener, Client

import numpy as np


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
