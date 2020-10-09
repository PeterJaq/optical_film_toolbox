import numpy as np
import tensorflow as tf
import model.optical_model_env as optical_model_env
from model import ActorCritic

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = optical_model_env.optical_film_env()

N_F = env.n_features
N_A = env.n_actions

sess = tf.Session()

actor = ActorCritic.Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = ActorCritic.Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s, mean_abs = env.init_Device()
    t = 0
    track_r = []
    while True:

        a = actor.choose_action(s)

        s_, r, done, info = env.run_simulate(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done :
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            print("episode:", i_episode, "  reward:", int(running_reward))
            break

        print('[%d episode-%d step], the final observation:%s, abs:%f, reward:%f, TD_error:%f' % (i_episode, t, s, info, r, td_error))