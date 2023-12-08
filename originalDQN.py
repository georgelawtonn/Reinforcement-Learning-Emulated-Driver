import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from car_racing import CarRacing

# Initialize the CarRacing environment with specific parameters
env = CarRacing(render_mode="human", continuous=False, domain_randomize=False, obstacles=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the QNetwork class representing the neural network model
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(12544, 256),  # Adjusted input size to match the flattened output size
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 5)
        )

        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x



# Define the Agent class responsible for training and decision-making
class Agent():
    def __init__(self):
        self.qnetwork_local = QNetwork().to(device)
        self.qnetwork_target = QNetwork().to(device)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.batch_size = 5
        self.train_start = 3000
        self.counter_1 = 0
        self.counter_2 = 0
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return np.argmax(action_values.cpu().data.numpy()) if random.random() > self.epsilon else random.choice(np.arange(5))

    def step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.counter_2 = (self.counter_2 + 1) % 500
        self.counter_1 = (self.counter_1 + 1) % 4

        if self.counter_2 == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        if self.counter_1 == 0 and len(self.memory) >= self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
            self.learn(minibatch)

    def learn(self, batch):
        criterion = torch.nn.MSELoss()

        states = np.zeros((self.batch_size, 96, 96, 3))
        next_states = np.zeros((self.batch_size, 96, 96, 3))
        actions, rewards, dones = [], [], []

        for i, (state_i, action_i, reward_i, next_state_i, done_i) in enumerate(batch):
            states[i] = state_i
            next_states[i] = next_state_i
            actions.append(action_i)
            rewards.append(reward_i)
            dones.append(done_i)

        actions = torch.from_numpy(np.vstack(actions)).to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones)).to(device)

        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        predictions = self.qnetwork_local(torch.from_numpy(states).float().to(device)).gather(1, actions.type(torch.int64))

        with torch.no_grad():
            q_next = self.qnetwork_target(torch.from_numpy(next_states).float().to(device)).detach().max(1)[0].unsqueeze(1)

        targets = rewards + (self.gamma * q_next * (1 - dones.float()))
        targets = targets.float()
        loss = criterion(predictions, targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main training and testing loop
agent = Agent()
rewards = []
total_episodes = 1000
max_steps = 5000
speed_over_time = []
previous_position = None
previous_time = 0

for episode in tqdm(range(total_episodes)):
    state, _ = env.reset()
    cumulative_reward = 0
    previous_position = None
    previous_time = 0
    state_torch = torch.from_numpy(state).float().unsqueeze(0).to(device)
    Q_values = agent.qnetwork_local(state_torch)
    probabilities = F.softmax(Q_values, dim=1).squeeze(0)
    log_probabilities = torch.log(probabilities)

    for i in range(max_steps):
        env.render()
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)

        current_position = (
            env.previous_x,
            env.previous_y
        )

        current_time = i

        if previous_position is not None and previous_time is not None:
            prev_x, prev_y = previous_position
            curr_x, curr_y = current_position
            delta_position = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            delta_time = current_time - previous_time
            speed = delta_position / delta_time if delta_time != 0 else 0.0
            speed_over_time.append(speed)

        previous_position = current_position
        previous_time = current_time

        agent.step(state, action, reward, next_state, done)
        state = next_state
        cumulative_reward += reward

        if done or truncated:
            break

    rewards.append(cumulative_reward)
    print(f"Episode {episode}/{total_episodes}, Return = {cumulative_reward}, The epsilon now is: {agent.epsilon}")

torch.save(agent.qnetwork_local.state_dict(), 'model_weights.pth')

rolling_average = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]
plt.plot(rewards, label='Return')
plt.plot(rolling_average, label='10 Episode Rolling Average')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Average Return for Each Episode')
plt.legend()
plt.show()

plt.plot(speed_over_time)
plt.xlabel('Time steps')
plt.ylabel('Speed')
plt.title('Car Speed Over Time')
plt.show()
