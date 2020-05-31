# -*- coding: utf-8 -*-
import random, time
import gym
import numpy as np
from collections import deque
import tensorflow as tf

class QAgent:
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
        # using SELU as main activation 
        # following
        #   https://arxiv.org/abs/1706.02515

        inpx = tf.keras.layers.Input(shape=(self.state_size))
        x = inpx
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu', use_bias=False)(x)
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu', use_bias=False)(x)
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu', use_bias=False)(x)
        x = tf.keras.layers.GaussianDropout(0.1)(x)


        def deepDueling(x, layers=[64,64,64]):
            layer = x
            for neurons in layers:
                layer = tf.keras.layers.Dense(neurons, kernel_initializer='lecun_normal', activation='selu')(layer)
            return layer

        # Dueling network architectire
        # following 
        #   https://medium.com/analytics-vidhya/introduction-to-dueling-double-deep-q-network-d3qn-8353a42f9e55
        #   https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751

        actions = deepDueling(x)
        actions = tf.keras.layers.Dense(self.action_size, activation='softmax')(actions)
        
        value = deepDueling(x)
        value = tf.keras.layers.Dense(1, activation='linear')(value)

        # dueling X
        x = tf.keras.layers.Flatten()(actions*value)

        # Compiled model, using Huber loss
        model = tf.keras.Model(inputs=inpx, outputs=x)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        # if training and np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
        act_values = self.model(state, training=training)
        # if not training:
            # print(act_values[0])
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        next_states = []
        for state, action, reward, next_state, done in minibatch:
            states.append(state[0])
            next_states.append(next_state[0])
        states = np.array(states)
        next_states = np.array(next_states)

        predState = np.array(self.model(states, training=False))
        predNextState = np.array(self.model(next_states, training=False))
        # minibatch = random.sample(self.memory, batch_size)
        targets = []
        for i in range(len(minibatch)):
        # for state, action, reward, next_state, done in minibatch:
            state, action, reward, next_state, done = minibatch[i]
            target = reward
            nextSample = predNextState[i]
            Qmax = np.amax(nextSample)
            target_f = predState[i]
            if not done:
                target = (reward + self.gamma*Qmax)
            target_f[action] = target
            targets.append(target_f)

        targets = np.array(targets)
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
        history = self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
        return history.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env_name = 'LunarLander-v2'
EPISODES = 100

if __name__ == "__main__":
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    # print(env.action_space.sample())
    action_size = 4
    agent = QAgent(state_size, action_size)
    agent.load(f"./model-{env_name}.h5")

    done = False
    batch_size = 128

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
                    break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                print(f"episode: {e: 4d}/{EPISODES:4d}, steps: {time: 4d}, reward: {cumReward:+7.2f}, loss = {loss:.2f}")
        agent.save(f"./model-{env_name}.h5")
    else:
        # env = gym.wrappers.Monitor(env, './videos/dqn-' + str(time.time()) + '/')
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        cumReward = 0
        for time in range(5000):
            env.render()
            action = agent.act(state, False)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            cumReward += reward
            state = next_state
            if done:
                # print(f"reward: {cumReward}")
                print(f"steps:{time} reward: {cumReward}")
                break
