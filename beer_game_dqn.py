import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# === CUDA Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Output Directory Setup ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === Configuration ===
config = {
    "time_unit": "week",  # Choose from "week", "day", "month"
    "goal": "cost_min",   # Choose from "cost_min", "profit_max", "env_min", "weighted"
    "reward_weights": {"cost_min": 1.0, "profit_max": 0.0, "env_min": 0.0},
    "batch_size": 64,     # Increased batch size
    "num_episodes": 500,
    "learning_rate": 0.0005,  # Small learning rate
    "discount_factor": 0.99,  # Higher discount factor to focus on long-term rewards
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,   # Higher minimum to ensure exploration
    "epsilon_decay": 0.998  # Slower decay rate
}

# Save configuration to file
with open(os.path.join(output_dir, "option.txt"), "w") as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

# === Beer Game Environment ===
class BeerGameEnv:
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()
        self.reward_history = []  # Record rewards for each step

    def reset(self):
        self.inventory = 10
        self.backlog = 0
        self.order = 5
        self.demand = 5
        self.step_count = 0
        self.reward_history = []
        return self.get_state()

    def get_state(self):
        return np.array([self.inventory, self.backlog, self.order, self.demand], dtype=np.float32)

    def step(self, action):
        self.order = max(0, int(action))
        received = self.order

        # Calculate next demand (with more stable demand variation)
        self.demand = max(1, min(10, int(self.demand + np.random.randint(-1, 2))))
        
        # Update inventory and backlog
        self.inventory += received - self.demand
        if self.inventory < 0:
            self.backlog -= self.inventory  # Add negative inventory to backlog
            self.inventory = 0
        else:
            # Process backlogged orders
            fulfilled = min(self.backlog, self.inventory)
            self.backlog -= fulfilled
            self.inventory -= fulfilled

        # Calculate costs and rewards
        holding_cost = self.inventory * 0.1
        backlog_cost = self.backlog * 0.5
        profit = max(0, self.demand * 2 - self.order * 1.5)
        env_impact = self.order * 0.2
        
        if config["goal"] == "cost_min":
            # Normalize reward to range -10 to 0
            reward = - min(10, (holding_cost + backlog_cost))
        elif config["goal"] == "profit_max":
            reward = min(10, profit)  # Bound reward to range 0 to 10
        elif config["goal"] == "env_min":
            reward = - min(10, env_impact)  # Bound reward to range -10 to 0
        else:
            weights = config["reward_weights"]
            reward = (- weights["cost_min"] * min(10, (holding_cost + backlog_cost)) + 
                      weights["profit_max"] * min(10, profit) - 
                      weights["env_min"] * min(10, env_impact))
            # Bound weighted reward to range -10 to 10
            reward = max(-10, min(10, reward))
        
        self.reward_history.append(reward)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Return richer state information
        info = {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "demand": self.demand,
            "holding_cost": holding_cost,
            "backlog_cost": backlog_cost,
            "profit": profit,
            "env_impact": env_impact
        }
        
        return self.get_state(), reward, done, info

# === DQN Agent (CUDA compatible, improved version) ===
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = config["discount_factor"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_end"]
        self.lr = config["learning_rate"]
        
        # Deeper network architecture
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        self.target_model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(device)
        
        # Adam optimizer with adjusted parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.memory = deque(maxlen=10000)  # Increased memory size
        
        # Initialize target network
        self.update_target_network()

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
        print("Target network updated")

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0  # Not enough samples for training
            
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Create numpy arrays first, then convert to tensors (to avoid warnings)
        states = np.array([experience[0] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor([experience[1] for experience in batch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([experience[2] for experience in batch]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor([experience[4] for experience in batch]).unsqueeze(1).to(device)
        
        # Calculate Q values
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Use Huber loss (robust to outliers)
        loss = nn.SmoothL1Loss()(q_values, targets)
        
        # Calculate gradients and optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping (stabilizes training)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

# === Run Training (CUDA compatible) ===
env = BeerGameEnv()
agent = DQNAgent(state_size=4, action_size=10)
reward_history = []
loss_history = []
avg_reward_history = []

# Progress display
print("Starting training...")

with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
    for episode in range(config["num_episodes"]):
        state = env.reset()
        total_reward = 0
        episode_steps = 0
        episode_loss = []
        
        for step in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train if enough samples are collected
            if len(agent.memory) >= config["batch_size"]:
                loss = agent.train(config["batch_size"])
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Post-episode processing
        reward_history.append(total_reward)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        avg_reward_history.append(avg_reward)
        
        # Calculate average loss
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        loss_history.append(avg_loss)
        
        # Periodically update target network
        if episode % 10 == 0:
            agent.update_target_network()
            
            # Log output
            log_msg = f"Episode {episode}/{config['num_episodes']}: Total Reward: {total_reward:.2f}, Avg Reward(100ep): {avg_reward:.2f}, Steps: {episode_steps}, Epsilon: {agent.epsilon:.4f}, Loss: {avg_loss:.6f}"
            print(log_msg)
            log_file.write(log_msg + "\n")
            
            # Save intermediate results every 100 episodes
            if episode % 100 == 0 and episode > 0:
                # Save model
                torch.save(agent.model.state_dict(), os.path.join(output_dir, f"model_ep{episode}.pth"))
                
                # Save intermediate graphs
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(reward_history, alpha=0.4, label='Episode Reward')
                plt.plot(avg_reward_history, label='Avg Reward (100 ep)')
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title("Training Progress")
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(loss_history, label='Loss')
                plt.xlabel("Episode")
                plt.ylabel("Loss")
                plt.title("Training Loss")
                plt.legend()
                plt.grid(True)
                
                plt.savefig(os.path.join(output_dir, f"progress_ep{episode}.png"))
                plt.close()

print("Training complete")

# Save final model
torch.save(agent.model.state_dict(), os.path.join(output_dir, "final_model.pth"))

# Visualize training results
plt.figure(figsize=(15, 10))

# Reward graph
plt.subplot(2, 1, 1)
plt.plot(reward_history, alpha=0.4, label='Episode Reward')
plt.plot(avg_reward_history, label='Avg Reward (100 ep)')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.legend()
plt.grid(True)

# Loss graph
plt.subplot(2, 1, 2)
plt.plot(loss_history, label='Loss')
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_results.png"))
plt.close()

# Test phase (evaluation)
print("Testing agent...")
test_episodes = 10
test_rewards = []
all_test_info = []

for i in range(test_episodes):
    state = env.reset()
    test_reward = 0
    episode_info = []
    
    for step in range(env.max_steps):
        # Testing uses epsilon=0 (fully exploit the learned model)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(agent.model(state_tensor)).item()
        next_state, reward, done, info = env.step(action)
        
        # Record information for each step
        step_info = {
            "step": step,
            "action": action,
            "reward": reward,
            "inventory": info["inventory"],
            "backlog": info["backlog"],
            "demand": info["demand"]
        }
        episode_info.append(step_info)
        
        state = next_state
        test_reward += reward
        if done:
            break
            
    test_rewards.append(test_reward)
    all_test_info.append(episode_info)
    print(f"Test episode {i+1}/{test_episodes}: reward {test_reward:.2f}")

avg_test_reward = sum(test_rewards) / len(test_rewards)
print(f"Average test reward: {avg_test_reward:.2f}")

with open(os.path.join(output_dir, "test_result.txt"), "w") as f:
    f.write(f"Average Test Reward: {avg_test_reward:.2f}\n")
    for i, r in enumerate(test_rewards):
        f.write(f"Test Episode {i}: {r:.2f}\n")

# Visualize test results
plt.figure(figsize=(15, 10))

# Test rewards
plt.subplot(2, 1, 1)
plt.bar(range(test_episodes), test_rewards)
plt.axhline(y=avg_test_reward, color='r', linestyle='-', label=f'Average: {avg_test_reward:.2f}')
plt.xlabel("Test Episode")
plt.ylabel("Total Reward")
plt.title("Test Results")
plt.legend()
plt.grid(True)

# Inventory and backlog trends (first test episode)
plt.subplot(2, 1, 2)
steps = [info["step"] for info in all_test_info[0]]
inventory = [info["inventory"] for info in all_test_info[0]]
backlog = [info["backlog"] for info in all_test_info[0]]
demand = [info["demand"] for info in all_test_info[0]]

plt.plot(steps, inventory, label='Inventory', marker='o')
plt.plot(steps, backlog, label='Backlog', marker='x')
plt.plot(steps, demand, label='Demand', marker='^')
plt.xlabel("Step")
plt.ylabel("Quantity")
plt.title("Inventory, Backlog, and Demand Trends (Test Episode 1)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "test_results.png"))
plt.close()

print(f"Results saved to {output_dir}")