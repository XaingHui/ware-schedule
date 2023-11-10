import time

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

from src.env.env import WarehouseEnvironment


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    env = WarehouseEnvironment(width=75, height=153, number=50)

    state_size = len(env.get_state())
    action_size = 3  # 代表上移、下移和不执行动作
    agent = DQNAgent(state_size, action_size)

    episodes = 1000

    for episode in range(episodes):
        state = env.get_state()
        print(env.step(1))
        print(env.current_time)
        time.sleep(24)
        # 从字典中获取数据
        tuple_data = np.array(state['agent_position'])
        list_data_1 = np.array(state['target_positions'])
        list_data_2 = np.array(state['item_positions'])
        nested_list_data = np.array(state['interference_positions'])
        print(env.step(1))
        print(env.current_time)
        print(tuple_data)
        print(list_data_1)
        print(list_data_2)
        print(nested_list_data)
        # 合并数据成一个多维数组
        # Convert positions to NumPy arrays
        agent_position = np.array(state['agent_position'])
        target_positions = np.array(state['target_positions'])
        item_positions = np.array(state['item_positions'])
        interference_positions = np.array(state['interference_positions'])

        # Concatenate and reshape the data
        reshaped_state = np.concatenate(
            [agent_position, target_positions.flatten(), item_positions.flatten(), interference_positions.flatten()])
        state = np.reshape(reshaped_state, [1, 4])

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            total_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    env.render()


if __name__ == "__main__":
    main()
