import os 
import sys
import csv

sys.path.append("/home/peterjaq/Project/optical_film_toolbox")

from optimizer.reinforce.envs.film_env import FilmEnv
from optimizer.reinforce.models.DQN import DeepQNetwork

def run_maze():
    step = 0
   
    write_temp = []
    write = []
    max_abs = 0
    for episode in range(500):
        # initial observation
        observation, mean_abs = env.init_thickness()
        print("The init Device is: %s abs:%f" % (observation, mean_abs))

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, mean_abs = env.run_simulate(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
            write_temp.append(mean_abs)

            print('%d step, the final observation:%s, abs:%f, reward:%f' % (step, observation, mean_abs, reward))

        
        if mean_abs > max_abs:
            max_abs = mean_abs
        write.append(write_temp)
        write_temp = []
    print("The best result is : %f" % max_abs)
     # end of game
    fileobj=open('answer_DQN.csv','w')
    writer = csv.writer(fileobj)
    for row in write:
        writer.writerow(row) 
    #env.destroy()


if __name__ == "__main__":
    # maze game
    env = FilmEnv()

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True 
                      )
    run_maze()
    RL.plot_cost()