import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the DQN model
def create_dqn_model():
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=env.observation_space.shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(env.action_space.n, activation='linear')
    ])
    return model

# Define the DQN agent
class DQNAgent:
    def __init__(self, env, buffer_size=10000):  # Add buffer_size parameter
            self.env = env
            self.model = create_dqn_model()
            self.target_model = create_dqn_model()
            self.target_model.set_weights(self.model.get_weights())
            self.replay_buffer = deque(maxlen=buffer_size)  # Use deque for replay buffer
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            self.batch_size = 32
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def remember(self, state, action, reward, next_state, done):
            self.replay_buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploration (random action)
        else:
            q_values = self.model.predict(state[np.newaxis, :])
            return np.argmax(q_values[0])  # Exploitation (choose best action)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = np.array(random.sample(self.replay_buffer, self.batch_size))
        states = np.vstack(batch[:, 0]).astype(np.float32)
        actions = np.array(batch[:, 1], dtype=np.int32)
        rewards = np.array(batch[:, 2], dtype=np.float32)
        next_states = np.vstack(batch[:, 3]).astype(np.float32)
        dones = np.array(batch[:, 4], dtype=np.bool)

        # Compute target Q-values
        target_q_values = rewards + self.gamma * np.max(self.target_model.predict(next_states), axis=1) * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values_actions = tf.reduce_sum(q_values * tf.one_hot(actions, self.env.action_space.n), axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values_actions))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# Initialize the DQN agent
agent = DQNAgent(env)

print('Training the DQN agent...')
num_episodes = 500
update_frequency = 32  # Set the update frequency to 32 samples (batch size)

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    if np.shape(state) == (2,):
        state = state[0]

    for t in range(1000):
        env.render()
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if np.shape(state) == (2,):
            state = state[0]
        if np.shape(obs) == (2,):
            obs = obs[0]
        agent.remember(state, action, reward, obs, terminated)
        state = obs

        # Check if it's time to update the model
        if len(agent.replay_buffer) >= agent.batch_size and len(agent.replay_buffer) % update_frequency == 0:
            for _ in range(update_frequency):  # Perform multiple updates at once
                agent.train()
                agent.update_epsilon()

        if terminated:
            agent.update_target_model()
            print(f"Episode {episode + 1}/{num_episodes}, Total reward: {total_reward}")
            break

#save model
agent.model.save('model.h5')


env.close()