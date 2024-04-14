import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Variable
from replay_memory import ReplayMemory
import random


def find_k_closest_vectors(vector, existing_vectors, k=5):

    sorted_vectors = sorted(
        ((np.linalg.norm(vector - x[0]), x) for x in existing_vectors),
        key=lambda x: x[0],
    )

    # Return the first k vectors along with their distances
    return sorted_vectors


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.std_dev_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)  # Mean of the action distribution
        log_std_dev = self.std_dev_layer(x)  # Log of standard deviation

        # Clip log_std_dev to prevent numerical instability
        log_std_dev = torch.clamp(log_std_dev, min=-20.0, max=2.0)
        std_dev = torch.exp(log_std_dev)  # Standard deviation

        return mean, std_dev


class Memory:  # Memory good for continuous action space
    def __init__(self):
        self.memory = []  # list of tuples (state, action, reward)

    def __len__(self):
        return len(self.memory)

    def add(self, s_a_v_tuple):
        self.memory.append(s_a_v_tuple)

    def lookup(self, state_vector):
        # Find the k closest keys in the memory
        closests = find_k_closest_vectors(state_vector, self.memory, k=30)

        return closests

    def update(self, oldState, state_reward_action_pair):
        # find index of old state in memory
        index = next(
            (
                i
                for i, (state, _, _) in enumerate(self.memory)
                if np.array_equal(state, oldState)
            ),
            -1,
        )
        if index == -1:
            print("Old state not found in memory.")
            return
        # update the memory with the new state
        self.memory[index] = state_reward_action_pair


class EpisodeMemory:
    def __init__(self):
        self.memory = []  # list of tuples (state, action, reward)

    def __len__(self):
        return len(self.memory)

    def add(self, s_a_r_tuple):
        self.memory.append(s_a_r_tuple)


game = gym.make("Hopper-v4")

state_size = game.observation_space.shape[0]
action_size = game.action_space.shape[0]

latent_space_size = state_size - 2

print("State size: ", state_size)
print("Action size: ", action_size)
print("Latent space size: ", latent_space_size)
replay_memory = ReplayMemory(5000)

mem = Memory()
actor_network = ActorNetwork(state_size, action_size)

amount_of_training_to_do = 1000
gamma = 0.99
close_neighbour_threshold = 1

# CEC Implementation: https://arxiv.org/pdf/2211.15183.pdf

# main loop
for i in range(amount_of_training_to_do):
    state, info = game.reset()
    done = False
    total_reward = 0
    end_state = None
    turns = 0
    episode_mem = EpisodeMemory()

    while not done:
        nearest_neighbours = mem.lookup(state)
        # filter off the far away neighbours
        filtered_neighbours = [
            x for x in nearest_neighbours if x[0] < close_neighbour_threshold
        ]

        if len(filtered_neighbours) > 0:
            state_tensor = Tensor(state)
            mean, std_deviation = actor_network(state_tensor)
            action = mean + torch.randn_like(std_deviation) * std_deviation
            action = action.detach().numpy()  # Convert to NumPy array
        else:
            action = game.action_space.sample()

        # do an action in the environment
        next_state, reward, terminated, truncated, _ = game.step(action)

        episode_mem.add((state, action, reward))

        state = next_state

        done = terminated or truncated

        if done:
            if terminated:
                end_state = "terminated"
            else:
                end_state = "truncated"

        turns += 1

        total_reward += reward

    v = 0
    while len(episode_mem) > 0:
        state, action, reward = episode_mem.memory.pop()
        # calculate v so we have (s,a,v)
        v = reward + gamma * v

        # find the closest state neighbours
        closest_state_neighbours = mem.lookup(state)

        the_closest_state_neighbour = None
        if len(closest_state_neighbours) > 0:
            the_closest_state_neighbour = closest_state_neighbours[0]

        # if there is no close neighbour or the closest neighbour is far away
        if (
            the_closest_state_neighbour is None
            or the_closest_state_neighbour[0] > close_neighbour_threshold
        ):
            # add the state to the memory, we are far away from the neighbours meaning we have new information
            mem.add((state, action, v))
        else:
            # check if the new value is better than the closest neighbour
            if the_closest_state_neighbour[1][2] < v:
                # update the closest neighbour
                mem.update(the_closest_state_neighbour[1][0], (state, action, v))

    print(
        f"Episode {i} after {turns} turns - {total_reward} - {end_state} - {len(mem)}"
    )
