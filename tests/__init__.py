from multiprocessing.connection import Listener
import numpy as np


def connect_and_send(offset=np.pi/2, ln=400):
    server = Listener(('localhost', 25000))
    c = server.accept()
    dt = 2 * np.pi / 400
    for i in range(ln):
        price = 100 + 100 * 0.5 * np.sin(dt * i - offset)
        data = ([price] * 4) + [price / 10]
        c.send(data)
    c.close()
    server.close()
