from multiprocessing.connection import Listener
import numpy as np


class Distributor:
    def __init__(self, queue, n_nodes=50):
        super(Distributor, self).__init__()
        self.n = n_nodes
        self.queue = queue
        self.server = Listener(('localhost', 6200), authkey=b'veryscrape')
        self.types = ['article', 'blog', 'reddit', 'twitter', 'stock']

    def create_matrix(self, data):
        m = np.zeros([len(data), len(self.types)])
        for i, t in enumerate(data):
            for j, q in enumerate(self.types):
                m[i][j] = data[q][t]
        return m

    def run(self):
        conns = []
        for _ in range(self.n):
            conns.append(self.server.accept())

        while not any(c.closed for c in conns):
            data = self.queue.get()
            mat = self.create_matrix(data)
            for c in conns:
                c.send(mat)
