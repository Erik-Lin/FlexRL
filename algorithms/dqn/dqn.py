import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 定义DQN神经网络模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 7x7
        # self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=3)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, output_size)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x = torch.relu(self.conv1(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = self.dropout(x)  # 添加丢弃率
        return self.fc4(x)

# 定义DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, gamma=0.99, lr=0.003, batch_size=128, epsilon_decay=0.95, epsilon_min=0.05):
        self.state_size = state_size
        self.action_size = action_size
        # self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.count = 0  # 计数器,记录更新次数
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.last_loss = 0.0  # 添加一个属性用于存储最后一次的损失值

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)
        for state, action, reward, next_state, done in batch:

            state = torch.tensor([state], dtype=torch.float32).squeeze()
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state = torch.tensor(np.array([next_state]), dtype=torch.float32).squeeze()
            done = torch.tensor(done, dtype=torch.float32)

            current_q_values = self.q_network(state).gather(0, torch.tensor([action.item()])).squeeze()
            next_q_values = self.target_network(next_state).max(0)[0].detach()
            target_q_values = reward + self.gamma * (1 - done) * next_q_values

            loss = nn.MSELoss()(current_q_values, target_q_values)
            self.last_loss = loss.item()  # 将损失值赋给last_loss属性
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def target_train(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, model_path):
        # 保存神经网络模型的权重
        torch.save(self.q_network.state_dict(), model_path)

    def load_model(self, model_path):
        # 加载神经网络模型的权重
        self.q_network.load_state_dict(torch.load(model_path))
    
    def get_last_loss(self):
        # 返回最后一次计算的损失值
        return self.last_loss