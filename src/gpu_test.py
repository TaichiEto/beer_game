import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# === 実験設定 ===
time_units = ["week", "day", "month"]
goals = ["cost_min", "profit_max", "env_min", "weighted"]
num_episodes = 2000
batch_size = 512
config = {"gpu": True}  # GPUを使用するか（True: GPU, False: CPU）

device = torch.device("cuda" if config["gpu"] and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === ビールゲームの環境 ===
class BeerGameEnv:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.inventory = 10
        self.backlog = 0
        self.order = 5
        self.demand = 5
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return np.array([self.inventory, self.backlog, self.order, self.demand], dtype=np.float32)

    def step(self, action):
        self.order = max(0, int(action))
        received = self.order

        self.inventory += received - self.demand
        if self.inventory < 0:
            self.backlog += abs(self.inventory)
            self.inventory = 0
        else:
            self.backlog = max(0, self.backlog - self.inventory)

        self.demand = max(1, int(self.demand + np.random.randint(-2, 3)))

        holding_cost = self.inventory * 0.1
        backlog_cost = self.backlog * 0.5
        profit = max(0, self.demand * 2 - self.order * 1.5)
        env_impact = self.order * 0.2

        if config["goal"] == "cost_min":
            reward = - (holding_cost + backlog_cost)
        elif config["goal"] == "profit_max":
            reward = profit
        elif config["goal"] == "env_min":
            reward = -env_impact
        else:
            weights = config.get("reward_weights", {"cost_min": 1.0, "profit_max": 0.0, "env_min": 0.0})
            reward = - (weights["cost_min"] * (holding_cost + backlog_cost)) + \
                     (weights["profit_max"] * profit) - (weights["env_min"] * env_impact)

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.get_state(), reward, done

# === DQN エージェント（PyTorch, GPU対応） ===
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=2000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# === 全パターンの実験実行 ===
for time_unit in time_units:
    for goal in goals:
        config["reward_weights"] = {"cost_min": 1.0, "profit_max": 0.0, "env_min": 0.0}
        config["goal"] = goal
        env = BeerGameEnv()
        agent = DQNAgent(state_size=4, action_size=10)
        
        reward_history = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            for step in range(env.max_steps):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                agent.train(batch_size)
                state = next_state
                total_reward += reward
                if done:
                    break
            if episode % 10 == 0:
                agent.update_target_network()
                print(f"[{time_unit} - {goal}] Episode {episode}: Total Reward: {total_reward:.2f}")
            reward_history.append(total_reward)
        
        plt.plot(reward_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Training Progress of PyTorch DQN in Beer Game ({time_unit} - {goal})")
        plt.savefig(f'training_curve_{time_unit}_{goal}.png')
