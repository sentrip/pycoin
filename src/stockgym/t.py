from multiprocessing.connection import Listener
from threading import Thread
import gym
def acc():
    Listener(('localhost', 6200), authkey=b'veryscrape').accept()
Thread(target=acc).start()
env = gym.make('StockGym-v0')
