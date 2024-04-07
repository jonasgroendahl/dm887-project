import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import Tensor
from torch.autograd import Variable
from replay_memory import ReplayMemory
import random


# important thing in NEC algorithm:

# The semi-tabular representation is an append-only memory that binds slow-changing keys to fast updating values
# and uses a context-based lookup on the keys to retrieve useful values during action selection by the agent
# A unique aspect of the memory in contrast to other neural memory architectures for reinforcement learning
# (explained in more detail in Section 3) is that the values retrieved from the memory can be updated much faster
# than the rest of the deep neural network.
# Instead, we elect to write all experiences to the memory, and allow it to grow very large compared
# to existing memory architectures
# Reading from this large memory is made efficient using kd-tree based nearest neighbour search

# 3 components
# 1. A neural network that extracts the embedding from the state
# 2. A differentiable neural dictionary (DND) that stores the key-value pairs
# 3. A final network that converts read-outs from the DND into Q-values


# questions so far:
# 1. what is the latent space representation of the state?
# 2. does it make sense to use CCN for this task? we are not dealing with image data like in the Atari game
# 3. is it ok to use 3rd party libraries for the k-tree implementation?
#  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
# 4. what is the difference between the DND and the replay memory?
# 5. with a continous action space, do we just need 1 DND with all the actions inside or for 1 each?

# GPT suggestions to 5):

# 1. Shared Memory Module: You could use a single memory module that stores episodic experiences across all actions.
# This shared memory module would be responsible for storing and retrieving episodic experiences,
# regardless of the specific action taken. This approach allows for more efficient memory utilization
# and can still capture valuable information across the continuous action space.

# 2. Function Approximation: Rather than storing episodic experiences for each action explicitly,
# you could use function approximation techniques to generalize across actions.
# For example, you could use a neural network to approximate the Q-value function directly from states and actions,
# without explicitly storing episodic memories.


class CCN(nn.Module):
    def __init__(self, state_size, latent_space_size):
        super(CCN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Fully connected layer to produce latent representation
        self.fc = nn.Sequential(nn.Linear(64, 512), nn.Linear(512, latent_space_size))

    def forward(self, x):
        x = self.conv(x)
        # Apply the fully connected layer to produce the latent representation
        latent_representation = F.relu(self.fc(x))
        return latent_representation


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DND:  # Differentiable Neural Dictionary
    def __init__(self):
        self.memory = {}

    def to_key(self, state_tensor):
        return tuple(state_tensor.squeeze().tolist())

    def add(self, latent_state_tensor, q_value):
        self.memory[self.to_key(latent_state_tensor)] = q_value

    # we get a key, now we need to find the nearest key in the memory (KDTree implementation p=30 neighbors)
    def lookup(self, key):
        # we get a key, now we need to find the nearest keys in the memory based on the key (KDTree implementation p=30 neighbors)
        all_indices_in_memory = []

        total_q_value = 0

        # sum up q value
        for key_in_memory in all_indices_in_memory:
            total_q_value += self.memory[key_in_memory]

        return total_q_value


game = gym.make("Hopper-v4")

state_size = game.observation_space.shape[0]
action_size = game.action_space.shape[0]

latent_space_size = state_size - 2

print("State size: ", state_size)
print("Action size: ", action_size)
print("Latent space size: ", latent_space_size)

ffn = FFN(state_size, latent_space_size, action_size)
dnd_list = {}
replay_memory = ReplayMemory(5000)


for i in range(10):
    state, info = game.reset()
    done = False
    total_reward = 0
    while not done:
        embedded_state = ffn(Tensor(state).unsqueeze(0))

        if len(dnd_list) < 10 or random.random() < 0.1:  # explore
            action = game.action_space.sample()
        else:
            # exploitation
            with torch.no_grad():
                q_estimates = [dnd.lookup(embedded_state) for dnd in dnd_list]
                action = torch.cat(q_estimates).max(0)[1].data[0]

        # do an action in the environment
        next_state, reward, terminated, truncated, _ = game.step(action)
        state = next_state

        # todo
        q_value = Tensor([reward])

        if dnd_list[action] is None:
            dnd_list[action] = DND()

        dnd_list[action].add(ffn(Tensor(state).unsqueeze(0)), q_value)
        replay_memory.push(state, action, q_value)

        done = terminated or truncated

        total_reward += reward

    print(f"{i} {total_reward}")


# todo - add more here
class NECAgent:
    def __init__(self, n_lookahead, q_learningrate):
        self.n_lookahead = n_lookahead
        self.q_learningrate = q_learningrate

    def discount(self, rewards, gamma):
        """
        Compute discounted sum of future values
        out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
        """
        discounted = []
        running_sum = 0
        for i in range(len(rewards) - 1, -1, -1):
            running_sum = rewards[i] + gamma * running_sum
            discounted.insert(0, running_sum)
        return discounted

    def n_step_q_value_estimate(self, t):
        """
        Return the N-step Q-value lookahead from time t in the transition queue
        """

        # sum(gamma^i * r_i) -- page 4
        lookahead = self.discount(
            [
                transition.reward
                for transition in self.transition_queue[t : t + self.n_lookahead]
            ],
            self.gamma,
        )[0]
        state = self.transition_queue[t + self.n_lookahead].state
        state_embedding = self.embedding_network(Tensor(state)).unsqueeze(0)

        # gamma^N * max(Q(s', a') + sum(gamma^i * r_i)  -- page 4
        return (
            self.gamma**self.n_lookahead
            * torch.cat([dnd.lookup(state_embedding) for dnd in self.dnd_list]).max()
            + lookahead
        )

    def q_update(self, q_initial, q_n):
        """
        Return the Q-update for DND updates
        """

        # q_i = q_i + alpha * (q_n - q_i) -- page 4
        return q_initial + self.q_learningrate * (q_n - q_initial)
