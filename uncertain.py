# -*- coding: utf-8 -*-
import random, time
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow_probability as tfp

EPISODES = 1000

# @tf.function
def ProbLoss(y_true, y_pred):
    # print(y_pred)
    q = tfp.distributions.Normal(loc=y_pred[:,:,0], scale=y_pred[:,:,1]).prob(y_true)
    # print(losses)
    return tf.reduce_mean(-tf.math.log(tf.maximum(1e-6, tf.reduce_mean(q, axis=-1))))
    # return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
    # tf.reduce_mean

class DQNUncertainAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        # self.epsilon = 0.2  # exploration rate
        self.learning_rate = 1e-5
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        inpx = tf.keras.layers.Input(shape=(self.state_size))
        x = inpx
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='selu')(x)
        # x = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        x = tf.keras.layers.GaussianDropout(0.1)(x)


        eluplus = lambda x: tf.nn.elu(x)+1

        def headdistr(inp, prelayers=[24,24,24]):
            layer = inp
            for ls in prelayers:
                layer = tf.keras.layers.Dense(units=ls, kernel_initializer='lecun_normal', activation='selu')(layer)
            mu = tf.keras.layers.Dense(1, activation='linear')(layer)
            # mu._name = mu.name + str("_mu")
            sigma = tf.keras.layers.Dense(1, activation=eluplus)(layer)
            # sigma._name = sigma.name + str("_sig")
            # return mu, sigma
            # return tf.keras.layers.concatenate([mu, sigma],axis=1)
            return tf.expand_dims(tf.keras.layers.concatenate([mu, sigma],axis=1), 1)

        actQ = tf.keras.layers.concatenate([headdistr(x, prelayers=[]) for _ in range(self.action_size)], axis=1)

        model = tf.keras.Model(inputs=inpx, outputs=actQ)
        model.compile(loss=ProbLoss, optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getQSamples(self, state, training=True):
        act_values = np.array(self.model(state, training=training))[0]
        # print("Q predict:",act_values)

        rewards_sample = []
        for Q_sample in act_values:
            rewards_sample.append(np.random.normal(Q_sample[0], Q_sample[1], 1)[0])
        # print("Q samples:",rewards_sample)        
        return rewards_sample

    def getFullQSamples(self, state):
        act_values = np.array(self.model(state, training=False))
        # print("Q predict:",act_values)

        batches = []
        for batch in act_values:
            rewards_sample = []
            for Q_sample in batch:
                rewards_sample.append(np.random.normal(Q_sample[0], Q_sample[1], 1)[0])
            batches.append(rewards_sample)
        # print("Q samples:",rewards_sample)        
        return np.array(batches)

    def act(self, state, training=True):
        # if training and np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
        samples = self.getQSamples(state, training)
        if not training:
            print(samples)
        return np.argmax(samples)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        next_states = []
        for state, action, reward, next_state, done in minibatch:
            states.append(state[0])
            next_states.append(next_state[0])
        states = np.array(states)
        next_states = np.array(next_states)
        # print(states.shape)
        # print(next_states.shape)
        predState = self.getFullQSamples(states)
        predNextState = self.getFullQSamples(next_states)

        targets = []
        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            target = reward
            nextSample = predNextState[i]
            Qmax = np.amax(nextSample)
            if not done:
                target = (reward + self.gamma*Qmax)
            target_f = predState[i]
            target_f[action] = target
            # target_f = np.array([target_f])
            targets.append(target_f)
            # print("X:", state)
            # print("Y:", target_f)
        targets = np.array(targets)

        # print("X:", states.shape)
        # print("Y:", targets.shape)
        history = self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
        return history.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env_name = 'LunarLander-v2'

if __name__ == "__main__":
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]

    action_size = env.action_space.n
    agent = DQNUncertainAgent(state_size, action_size)
    # agent.load(f"./model-{env_name}-un2.h5")

    done = False
    batch_size = 128

    # train=False
    train=True
    if train:
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            cumReward = 0
            for time in range(5000):
                # env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                cumReward += reward
                state = next_state
                if done:
                    break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                print(f"episode: {e: 4d}/{EPISODES:4d}, steps: {time: 4d}, reward: {cumReward:+7.2f}, loss = {loss:.2f}")
        agent.save(f"./model-{env_name}-un2.h5")
    else:
        # env = gym.wrappers.Monitor(env, './videos/dqn-un-' + str(time.time()) + '/')
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
                print(f"steps:{time} reward: {cumReward}")
                break
