import optical_model_env
from model import policy_gridient_softmax
import csv

env = optical_model_env.optical_film_env()

RL = policy_gridient_softmax.PolicyGradient(
    env.n_actions, 
    env.n_features,
    learning_rate=0.02,
    reward_decay=0.995)



step = 0
write_temp = []
write = []
max_abs = 0
write_file_name = "./Policy_gridient.txt"

for i_episode in range(30):

    observation, mean_abs = env.init_Device()
    f = open(write_file_name,"w")
    while True:
        #if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, mean_abs = env.run_simulate(action)   # reward = -1 in all cases

        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()  # train

            break
        step += 1
        write_temp.append(mean_abs)

        print('%d step, the final observation:%s, abs:%f, reward:%f' % (step, observation, mean_abs, reward))

        observation = observation_

        if mean_abs > max_abs:
            max_abs = mean_abs
    write.append(write_temp)
    write_temp = []
    print("The best result is : %f" % max_abs)

fileobj=open('answer_policy_gridient_.csv','w')
writer = csv.writer(fileobj)
for row in write:
    writer.writerow(row) 
