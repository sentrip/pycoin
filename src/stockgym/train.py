from threading import Thread

import tensorflow as tf

from stockgym.a3c import A3CNetwork
from stockgym.agent import Agent
from stockgym.distributor import Distributor

if __name__ == '__main__':
    num_workers = 128
    gamma = .99
    state_size = 999
    n_classes = 990
    max_episode_length = 100000

    distributor = Distributor(num_workers)
    Thread(target=distributor.run).start()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        with tf.device("/cpu:0"):
            trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = A3CNetwork(state_size, n_classes, 'global', None)
            workers = [Agent(sess, trainer, coord, gamma) for _ in range(num_workers)]
        sess.run(tf.global_variables_initializer())
        for worker in workers:
            worker.start()
        coord.join(workers)
