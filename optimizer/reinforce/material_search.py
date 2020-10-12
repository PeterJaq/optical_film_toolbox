import os 
import sys
import csv

sys.path.append("/home/peterjaq/Project/optical_film_toolbox")

from optimizer.reinforce.envs.material_env import MaterialEnv
from optimizer.reinforce.models.A3C import ACNet, Worker

import multiprocessing
import threading
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import shutil


def run_search():

    tf.disable_eager_execution()

    # GAME = 'CartPole-v0'
    OUTPUT_GRAPH = True
    LOG_DIR = './log'
    N_WORKERS = multiprocessing.cpu_count()
    MAX_GLOBAL_EP = 1000
    GLOBAL_NET_SCOPE = 'Global_Net'
    UPDATE_GLOBAL_ITER = 10
    GAMMA = 0.9
    ENTROPY_BETA = 0.001
    LR_A = 0.001    # learning rate for actor
    LR_C = 0.001    # learning rate for critic
    GLOBAL_RUNNING_R = []
    GLOBAL_EP = 0

    # env = gym.make(GAME)
    env = MaterialEnv()

    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n
    #env.destroy()

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()



if __name__ == "__main__":
    # maze game
    env = FilmEnv()


    run_search()
    RL.plot_cost()