import math
import time
from multiprocessing.connection import Listener


def gen_data(x):
    start = [0, 0]
    types = ['article', 'blog', 'reddit', 'twitter', 'stock']
    data = {}
    for t in types:
        data[t] = {}
        for i in range(110):
            q = i % 25
            if i and q == 0:
                start[0] += 1
            st = ''.join(map(chr, [65 + start[i] % 26 for i in range(2)]))
            if i < 60:
                res = math.sin(x)
            else:
                res = 0.5
            if t == 'stock':
                res *= 150
            data[t][st] = res
            start[1] += 1
    return data


def send_data():
    conn = Listener(('localhost', 6100), authkey=b'veryscrape').accept()
    t = 0.001
    while True:
        conn.send(gen_data(t))
        time.sleep(2)
        t += 1


if __name__ == '__main__':
    send_data()
