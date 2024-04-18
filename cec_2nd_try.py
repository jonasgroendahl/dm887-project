import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Variable
from replay_memory import ReplayMemory
import random
import matplotlib.pyplot as plt

# defaults from https://gymnasium.farama.org/environments/mujoco/hopper/
healthy_state_range = (-100, 100)
healthy_z_range = (0.7, float("inf"))
healthy_angle_range = (-0.2, 0.2)


def check_termination_condition_hopper(state):
    # Check if any element of the observation is outside the healthy state range
    observation = state[1:]

    for value in observation:
        if not (healthy_state_range[0] <= value <= healthy_state_range[1]):
            return "Range issue"  # Hopper is unhealthy

    # Check if the height of the hopper is outside the healthy z range
    if not (healthy_z_range[0] <= state[0] <= healthy_z_range[1]):
        return "Height issue"  # Hopper is unhealthy

    # Check if the angle of the hopper is outside the healthy angle range
    angle_index = 1
    if not (healthy_angle_range[0] <= state[angle_index] <= healthy_angle_range[1]):
        return "Angle issue"  # Hopper is unhealthy

    return "Unsure"


def find_k_closest_vectors(vector, existing_vectors, k=5):

    sorted_vectors = sorted(
        ((np.linalg.norm(vector - x[0]), x) for x in existing_vectors),
        key=lambda x: x[0],
    )

    # Return the first k vectors along with their distances
    return sorted_vectors


def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# function to draw total_reward for each episode over time
def draw(total_rewards):
    plt.plot(total_rewards)
    plt.ylabel("Total reward")
    plt.xlabel("Episode")
    plt.show()


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

# hyperparameters
amount_of_training_to_do = 5000
eval_loops = 300
gamma = 0.99  # discount factor
close_neighbour_threshold_action_selection = (
    0.5  # threshold for selecting close neighbours
)
close_neighbour_threshold_memory = 0.5  # threshold for updating memory
noise_probability = 0.3  # higher value means more exploration (0-1), probability of adding noise to selected action
noise_std_dev = (
    0.2  # higher value means more exploration (0-1), more noise to selected action
)
softmax_temperature = 1.0  # higher value means more exploration (0-1), larger gap between values when calculating softmax

# not in use right now
hyperparameter_sets = [
    {
        "amount_of_training_to_do": 5000,
        "eval_loops": 300,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.5,
        "close_neighbour_threshold_memory": 0.5,
        "noise_probability": 0.3,
        "noise_std_dev": 0.2,
        "softmax_temperature": 1.0,
    },
    {
        "amount_of_training_to_do": 5000,
        "eval_loops": 300,
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.4,
        "close_neighbour_threshold_memory": 0.4,
        "noise_probability": 0.3,
        "noise_std_dev": 0.3,
        "softmax_temperature": 0.9,
    },
]

total_rewards = []
total_rewards_eval = []


# CEC Implementation: https://arxiv.org/pdf/2211.15183.pdf
# main loop
def train():
    for i in range(amount_of_training_to_do):
        state, info = game.reset()
        done = False
        total_reward = 0
        end_state = None
        turns = 0
        episode_mem = EpisodeMemory()

        reason_for_termination = None

        while not done:
            nearest_neighbours = mem.lookup(state)
            # filter off the far away neighbours
            filtered_neighbours = [
                x
                for x in nearest_neighbours
                if x[0] < close_neighbour_threshold_action_selection
            ]

            if len(filtered_neighbours) > 0:
                # Extract actions and values from filtered neighbors
                actions = np.array([x[1][1] for x in filtered_neighbours])
                values = np.array([x[1][2] for x in filtered_neighbours])

                # Softmax action selection
                action_probs = softmax(values, temperature=softmax_temperature)
                selected_action_index = np.random.choice(
                    len(filtered_neighbours), p=action_probs
                )
                action = actions[selected_action_index]

                # Add Gaussian noise to the selected action
                if np.random.rand() < noise_probability:
                    action += np.random.normal(0, noise_std_dev, size=action.shape)
            else:
                action = game.action_space.sample()

            # do an action in the environment
            next_state, reward, terminated, truncated, _ = game.step(action)

            episode_mem.add((state, action, reward))

            state = next_state

            done = terminated or truncated

            turns += 1

            total_reward += reward

            if done:
                if truncated:
                    end_state = "truncated"
                else:
                    end_state = "terminated"
                reason_for_termination = check_termination_condition_hopper(state)
                total_rewards.append(total_reward)

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
                or the_closest_state_neighbour[0] > close_neighbour_threshold_memory
            ):
                # add the state to the memory, we are far away from the neighbours meaning we have new information
                mem.add((state, action, v))
            else:
                # check if the new value is better than the closest neighbour
                if the_closest_state_neighbour[1][2] < v:
                    # update the closest neighbour
                    mem.update(the_closest_state_neighbour[1][0], (state, action, v))

        print(
            f"Episode {i} after {turns} turns - {total_reward} - {end_state} - {len(mem)} - {reason_for_termination if reason_for_termination is not None else state}"
        )


def evaluate():  # evaluation
    for i in range(eval_loops):
        state, info = game.reset()
        done = False
        total_reward = 0
        end_state = None
        turns = 0

        while not done:
            nearest_neighbours = mem.lookup(state)

            actions = np.array([x[1][1] for x in nearest_neighbours])
            values = np.array([x[1][2] for x in nearest_neighbours])

            # Softmax action selection
            action_probs = softmax(values)
            selected_action_index = np.random.choice(
                len(nearest_neighbours), p=action_probs
            )
            action = actions[selected_action_index]

            next_state, reward, terminated, truncated, _ = game.step(action)

            state = next_state

            done = terminated or truncated

            turns += 1

            total_reward += reward

            if done:
                if terminated:
                    end_state = "terminated"
                else:
                    end_state = "truncated"
                total_rewards_eval.append(total_reward)

        print(
            f"Evaluation: Episode {i} after {turns} turns - {total_reward} - {end_state} - {len(mem)}"
        )


train()
evaluate()
draw(total_rewards)
draw(total_rewards_eval)
