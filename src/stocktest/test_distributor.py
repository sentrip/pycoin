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
