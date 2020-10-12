import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
from dataclasses import dataclass


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
import numpy as np


class ActorCriticNet(tf.keras.Model):

    def __init__(self, action_space):

        super(ActorCriticNet, self).__init__()

        self.action_space = action_space

        self.dense1 = kl.Dense(100, activation="relu")
        self.dense2 = kl.Dense(100, activation="relu")
        self.values = kl.Dense(1, name="value")

        self.policy_logits = kl.Dense(action_space)

    @tf.function
    def call(self, x):

        x1 = self.dense1(x)
        logits = self.policy_logits(x1)

        x2 = self.dense2(x)
        values = self.values(x2)

        return values, logits

    def sample_action(self, state):

        state = tf.convert_to_tensor(np.atleast_2d(state), dtype=tf.float32)

        _, logits = self(state)
        action_probs = tf.nn.softmax(logits)
        cdist = tfp.distributions.Categorical(probs=action_probs)
        action = cdist.sample()

        return action.numpy()[0]

    def compute_grads(self, states, discounted_rewards):
        """
           loss =  MSE(discouted_rewards - V(state)) = MSE(Advantages)
        """

        with tf.GradientTape() as tape:

            estimated_values = self(states)

            loss = tf.reduce_mean(
                tf.square(discounted_rewards - estimated_values))

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)

@dataclass
class Step:

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class GlobalCounter:

    n: int = 0

class A3CAgent:

    MAX_TRAJECTORY = 5

    def __init__(self, agent_id, env,
                 global_counter, action_space,
                 global_ACNet,
                 gamma, global_history, global_steps_fin):

        self.agent_id = agent_id

        self.env = env

        self.global_counter = global_counter

        self.action_space = action_space

        self.global_ACNet = global_ACNet

        self.local_ACNet = ActorCriticNet(self.action_space)

        self.gamma = gamma

        self.global_history = global_history

        self.global_steps_fin = global_steps_fin

        self.optimizer = tf.keras.optimizers.Adam(lr=0.0004)

    def play(self, coord):

        self.total_reward = 0

        self.state = self.env.reset()

        try:
            while not coord.should_stop():

                trajectory = self.play_n_steps(N=self.MAX_TRAJECTORY)

                states = [step.state for step in trajectory]

                actions = [step.action for step in trajectory]

                if trajectory[-1].done:
                    R = 0
                else:
                    values, _ = self.local_ACNet(
                        tf.convert_to_tensor(np.atleast_2d(trajectory[-1].next_state),
                                             dtype=tf.float32))
                    R = values[0][0].numpy()

                discounted_rewards = []
                for step in reversed(trajectory):
                    R = step.reward + self.gamma * R
                    discounted_rewards.append(R)
                discounted_rewards.reverse()

                with tf.GradientTape() as tape:

                    total_loss = self.compute_loss(states, actions, discounted_rewards)

                grads = tape.gradient(
                    total_loss, self.local_ACNet.trainable_variables)

                self.optimizer.apply_gradients(
                    zip(grads, self.global_ACNet.trainable_variables))

                self.local_ACNet.set_weights(self.global_ACNet.get_weights())

                if self.global_counter.n >= self.global_steps_fin:
                    coord.request_stop()

        except tf.errors.CancelledError:
            return

    def play_n_steps(self, N):

        trajectory = []

        for _ in range(N):

            self.global_counter.n += 1

            action = self.local_ACNet.sample_action(self.state)

            next_state, reward, done, info = self.env.step(action)

            step = Step(self.state, action, reward, next_state, done)

            trajectory.append(step)

            if done:
                print(f"Global step {self.global_counter.n}")
                print(f"Total Reward: {self.total_reward}")
                print(f"Agent: {self.agent_id}")
                print()

                self.global_history.append(self.total_reward)

                self.total_reward = 0

                self.state = self.env.reset()

                break

            else:
                self.total_reward += reward
                self.state = next_state

        return trajectory

    def compute_loss(self, states, actions, discounted_rewards):

        states = tf.convert_to_tensor(
            np.vstack(states), dtype=tf.float32)

        values, logits = self.local_ACNet(states)

        discounted_rewards = tf.convert_to_tensor(
            np.vstack(discounted_rewards), dtype=tf.float32)

        advantages = discounted_rewards - values

        value_loss = advantages ** 2

        actions_onehot = tf.one_hot(actions, self.action_space, dtype=tf.float32)

        action_probs = tf.nn.softmax(logits)

        log_action_prob = actions_onehot * tf.math.log(action_probs + 1e-20)

        log_action_prob = tf.reduce_sum(log_action_prob, axis=1, keepdims=True)

        entropy = -1 * tf.reduce_sum(
            action_probs * tf.math.log(action_probs + 1e-20),
            axis=1, keepdims=True)

        policy_loss = tf.reduce_sum(
            log_action_prob * tf.stop_gradient(advantages),
            axis=1, keepdims=True)

        policy_loss += 0.01 * entropy
        policy_loss *= -1

        total_loss = tf.reduce_mean(0.5 * value_loss + policy_loss)

        return total_loss