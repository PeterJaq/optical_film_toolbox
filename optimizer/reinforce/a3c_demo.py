import sys
import os 

sys.path.append("/home/peterjaq/Project/optical_film_toolbox")

import threading

import tensorflow as tf
import pandas as pd
import numpy as np
import gym
import matplotlib.pyplot as plt

from optimizer.reinforce.models.A3C_keras import A3CAgent, ActorCriticNet, GlobalCounter, Step
from optimizer.reinforce.envs.material_env import MaterialEnv


MONITOR_DIR = "/home/peterjaq/Project/optical_film_toolbox/logs"

def main():

    ACTION_SPACE = 2

    NUM_AGENTS = 8

    N_STEPS = 50000

    with tf.device("/cpu:0"):

        global_counter = GlobalCounter()

        global_history = []

        global_ACNet = ActorCriticNet(ACTION_SPACE)

        global_ACNet.build(input_shape=(None, 4))

        agents = []

        for agent_id in range(NUM_AGENTS):

            agent = A3CAgent(agent_id=f"agent_{agent_id}",
                             env=gym.envs.make("CartPole-v1"),
                             global_counter=global_counter,
                             action_space=ACTION_SPACE,
                             global_ACNet=global_ACNet,
                             gamma=0.99,
                             global_history=global_history,
                             global_steps_fin=N_STEPS)

            agents.append(agent)

    coord = tf.train.Coordinator()
    agent_threads = []
    for agent in agents:
        target_func = (lambda: agent.play(coord))
        thread = threading.Thread(target=target_func)
        thread.start()
        agent_threads.append(thread)

    coord.join(agent_threads, stop_grace_period_secs=300)


    print(global_history)

    plt.plot(range(len(global_history)), global_history)
    plt.plot([0, len(global_history)], [195, 195], "--", color="darkred")
    plt.xlabel("episodes")
    plt.ylabel("Total Reward")
    plt.savefig(MONITOR_DIR / "a3c_cartpole-v1.png")

    df = pd.DataFrame()
    df["Total Reward"] = global_history
    df.to_csv(MONITOR_DIR / "a3c_cartpole-v1.csv", index=None)

if __name__ == "__main__":

    main()