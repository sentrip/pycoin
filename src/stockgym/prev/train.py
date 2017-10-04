import os
from threading import Thread
from time import sleep

import tensorflow as tf
from stockgym.ac3 import AC3Network

from stockgym.distributor import Distributor
from stockgym.prev.agent import Agent

if __name__ == '__main__':
    max_episode_length = 10000
    gamma = .99  # discount rate for advantage estimation and reward discounting
    state_size = 999
    n_classes = 9
    load_model = False
    model_path = './bin/model'
    tf.reset_default_graph()

    num_workers = 1
    distributor = Distributor(num_workers)
    Thread(target=distributor.run).start()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.device("/cpu:0"):
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC3Network(state_size, n_classes, 'global', None)  # Generate global network
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Agent(i, state_size, n_classes, trainer, model_path))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        for worker in workers:
            worker.run_work(max_episode_length, gamma, sess, coord, saver)
            sleep(0.5)
        coord.join(workers)
