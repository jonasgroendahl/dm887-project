import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import random


# Convolutional neural network for the Mnih et al. (2015) architecture

# 84x84x4 input
# first hidden layer:
# 32 8x8 filters with stride 4, ReLU
# second hidden layer:
# 64 4x4 filters with stride 2, ReLU
# third hidden layer:
# 64 3x3 filters with stride 1, ReLU
# fully connected layer:
# 512 units, ReLU
# output layer:
# 1 for each action

# since in the NEC algorithm, we dont use the output of the network to predict the Q-value,
# we can use the output of the network to predict the latent space representation of the state
# in our case the input is the state for the envionment and the output is the latent space representation of the state


class CCN(nn.Module):
    def __init__(self, state_size, latent_space_size):
        super(CCN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(state_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Fully connected layer to produce latent representation
        self.fc = nn.Linear(64, latent_space_size)

    def forward(self, x):
        x = self.encoder(x)
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.conv_output_size)
        # Apply the fully connected layer to produce the latent representation
        latent_representation = F.relu(self.fc(x))
        return latent_representation


class Memory:
    def __init__(self):
        self.memory = {}

    def to_key(self, state_tensor):
        return tuple(state_tensor.squeeze().tolist())

    def add(self, action, latent_state_tensor, q_value):
        if action not in self.memory:
            self.memory[action] = {}

        self.memory[action][self.to_key(latent_state_tensor)] = q_value


game = gym.make("Hopper-v4")

latent_space_size = 5
state_size = game.observation_space.shape[0]
action_size = game.action_space.shape[0]

print("State size: ", state_size)
print("Action size: ", action_size)

ccn = CCN(state_size, latent_space_size)

for i in range(10):
    state = game.reset()
    done = False
    while not done:
        action = game.action_space.sample()
        next_state, reward, done, cool, cool2 = game.step(action)
        state = next_state
