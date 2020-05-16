# -*- coding: utf-8 -*-
import random, time
import gym
import numpy as np
from collections import deque
import tensorflow as tf

EPISODES = 100

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.2  # exploration rate
        self.learning_rate = 1e-5
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        inpx = tf.keras.layers.Input(shape=(self.state_size))
        x = inpx
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.keras.layers.Dense(self.action_size, activation='linear')(x)

        model = tf.keras.Model(inputs=inpx, outputs=x)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(state, training=training)
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma*np.amax(self.model(next_state)[0]))
            target_f = np.array(self.model(state))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env_name = 'LunarLander-v2'

if __name__ == "__main__":
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    # print(env.action_space.sample())
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    agent.load(f"./model-{env_name}.h5")

    done = False
    batch_size = 256

    train=False
    # train=True
    if train:
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            cumReward = 0
            for time in range(5000):
                # env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                # reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                cumReward += reward
                state = next_state
                if done:
                    print("episode: {}/{}, steps: {}, e: {:.2}, reward: {}".format(e, EPISODES, time, agent.epsilon, cumReward))
                    break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        agent.save(f"./model-{env_name}.h5")
    else:
        env = gym.wrappers.Monitor(env, './videos/dqn-' + str(time.time()) + '/')
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        cumReward = 0
        for time in range(5000):
            env.render()
            action = agent.act(state, False)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            cumReward += reward
            state = next_state
            if done:
                # print(f"reward: {cumReward}")
                print(f"steps:{time} reward: {cumReward}")
                break
