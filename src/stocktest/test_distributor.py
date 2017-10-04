# todo write distributor test
# if __name__ == '__main__':
#     q = Queue()
#     p = Distributor(q, 4)
#     mock = {k: {t: 0.1 for t in p.topics} for k in p.types}
#     p.start()
#     cs = []
#     for _ in range(4):
#         cs.append(Client(('localhost', 6200), authkey=b'veryscrape'))
#
#     while True:
#         input()
#         q.put(mock)
#         for c in cs:
#             print(c.recv().shape)
# import time
# import unittest
# from multiprocessing import Queue
# from multiprocessing.connection import Client
#
# import numpy as np
#
# from stockgym import Distributor
#
#
# class TestDistributor(unittest.TestCase):
#
#     def test_distribute(self):
#         temp_dict = {'AAPL': {'twitter': 1, 'blog': 0.5, 'reddit': 0.1, 'article': 0.9, 'stock': 150.2}}
#         q = Queue()
#         d = Distributor(q)
#         q.put(temp_dict)
#         d.start()
#         cs = []
#         try:
#             for i in range(50):
#                 c = Client(('localhost', 6200), authkey=b'vs')
#                 cs.append(c)
#             time.sleep(0.01)
#             d.running = False
#             q.put(temp_dict)
#             d.join()
#             for i, c in enumerate(cs):
#                 try:
#                     assert c.poll(), 'No incoming item for client {}'.format(i)
#                     item = c.recv()
#                     assert isinstance(item, np.ndarray), 'Received item not array {}'.format(i)
#                     assert item.shape == (110, 5), 'Incorrect shape returned'
#                 finally:
#                     c.close()
#         finally:
#             d.server.close()
