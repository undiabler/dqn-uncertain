# -*- coding: utf-8 -*-
import random, time
import gym
import numpy as np
from collections import deque
import tensorflow as tf

from memory import ReplayBuffer

class QAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # self.memory = deque(maxlen=2000)
        self.memory = ReplayBuffer(input_shape=(state_size,))

        self.gamma = 0.95   # discount rate
        self.epsilon = 0.2  # exploration rate
        self.learning_rate = 1e-5
        self.model = self._build_model()
        self.oldmodel = self._build_model()

        self.activeReplays = 0
        self.updateReplays = 10

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
        # actions = tf.keras.layers.Dense(self.action_size, activation='softmax')(actions)
        actions = tf.keras.layers.Dense(self.action_size)(actions)
        
        value = deepDueling(x)
        value = tf.keras.layers.Dense(1, activation='linear')(value)

        # dueling X
        # x = tf.keras.layers.Flatten()(actions*value)

        # Combine streams into Q-Values, custom layer for reduce mean
        reduce_mean = tf.keras.layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

        x = tf.keras.layers.Add()([value, tf.keras.layers.Subtract()([actions, reduce_mean(actions)])])

        # Compiled model, using Huber loss
        model = tf.keras.Model(inputs=inpx, outputs=x)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # self.memory.append((state, action, reward, next_state, done))
        self.memory.add_experience(state[0], action, reward, done, next_state[0])

    def update_model(self):
        # Update the target Q network
        self.oldmodel.set_weights(self.model.get_weights())
        print("model updated")

    def act(self, state, training=True):
        # if training and np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
        act_values = self.model(state, training=training)
        # if not training:
            # print(act_values[0])
        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        # minibatch = random.sample(self.memory, batch_size)
        # states = []
        # next_states = []
        # for state, action, reward, next_state, done in minibatch:
        #     states.append(state[0])
        #     next_states.append(next_state[0])
        # states = np.array(states)
        # next_states = np.array(next_states)

        minibatch, prior, indx = self.memory.get_minibatch()
        states, actions, rewards, next_states, dones = minibatch

        predState = np.array(self.model(states, training=False))
        predNextState = np.array(self.oldmodel(next_states, training=False))
        
        targets = []
        for i in range(len(indx)):
        # for i in range(len(minibatch)):
        # for state, action, reward, next_state, done in minibatch:
            # state, action, reward, next_state, done = minibatch[i]
            state, _, reward, next_state, done = states[i], actions[i], rewards[i], next_states[i], dones[i]
            # print(state)
            action = np.argmax(predState[i])
    
            Qmax = predNextState[i][action]
            # Qmax = np.amax(predNextState[i])
            # target_f = predState[i]

            target = reward + self.gamma*Qmax*(1-done)
            # target_f[action] = target
            # targets.append(target_f)
            targets.append(target)

        targets = np.array(targets)
        # print(states.shape)
        # print(targets.shape)
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay

        # history = self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)
        # loss = history.history['loss'][0]

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.model(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.action_size, dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            # print(q_values)
            # print(Q)
            error = Q - targets
            loss = tf.keras.losses.Huber()(targets, Q)

            # if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                # loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))

        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
        # Target network 
        # print(indx,error.numpy())
        self.memory.set_priorities(indx, error.numpy())
        self.activeReplays +=1
        if self.activeReplays>self.updateReplays:
            self.update_model()
            self.activeReplays = 0

        return loss

    def load(self, name):
        self.model.load_weights(name)
        self.update_model()

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
    # agent.load(f"./model-{env_name}.h5")

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
        agent.load(f"./model-{env_name}.h5")
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
