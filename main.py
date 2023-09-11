import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

# Load starter data from JSON file
with open("starter_data.json", "r") as f:
    starter_data = json.load(f)

# Assuming we represent each state as a tensor (for simplicity)
# This is just a placeholder; you'll want to design a state representation that makes sense for your case
state_dim = 10


# Q-Network (again, very simplistic)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)  # Adjust the size as needed
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)  # Number of possible actions

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Initialize Q-network and optimizer
q_network = QNetwork(24, 24)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

frequency_dict = defaultdict(lambda: defaultdict(int))

# Dummy training loop (you'll need to replace this with your actual RL training logic)
for assignment in starter_data["assignments"]:

    # Step 1: Create frequency dictionary

    area = assignment["area"]
    coaches_pair = tuple(sorted([assignment["coaches"]["prime"], assignment["coaches"]["second"]]))

    frequency_dict[area][coaches_pair] += 1

    print(frequency_dict)

    # Step 2: Convert dictionary to array
    areas = sorted(starter_data["areas"])
    coach_pairs = sorted(
        [(c1, c2) for idx1, c1 in enumerate(starter_data["coaches"]) for idx2, c2 in enumerate(starter_data["coaches"])
         if idx1 < idx2])

    state_array = []

    for area in areas:
        for pair in coach_pairs:
            state_array.append(frequency_dict[area].get(pair, 0))

    print(state_array)

    # Initialize your state tensor
    # You'll need to write code to populate this dynamically based on your actual data
    state = torch.tensor(state_array, dtype=torch.float32)

    optimizer.zero_grad()

    q_values = q_network(state)
    best_action = torch.argmax(q_values).item()

    # Generate reward based on your rules for what makes a "good" assignment
    reward = 1

    # Update Q-values using Bellman equation
    loss = (reward + 0.99 * torch.max(q_values) - q_values[best_action]).pow(2)

    loss.backward()
    optimizer.step()

# To emit top 3 solutions, you could sort your Q-values and emit JSON (not shown here)

print("Model trained.")
