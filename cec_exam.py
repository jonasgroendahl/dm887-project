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
from sklearn.neighbors import KDTree
from torch import optim
from torch.utils.data import DataLoader
import pickle
import os
from datetime import datetime
from cec_exam_hyper_params import hyperparameter_sets

# defaults from https://gymnasium.farama.org/environments/mujoco/hopper/
healthy_state_range = (-100, 100)
healthy_z_range = (0.7, float("inf"))
healthy_angle_range = (-0.2, 0.2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)


# function to draw total_reward for each episode over time
def draw(total_rewards, name, config, use_mean=False):
    plt.figure(figsize=(10, 5))
    rewards = (
        [np.mean(total_rewards[i : i + 500]) for i in range(0, len(total_rewards), 500)]
        if use_mean
        else total_rewards
    )
    episodes = (
        range(500, len(total_rewards) + 1, 500)
        if use_mean
        else range(len(total_rewards))
    )
    plt.plot(episodes, rewards)
    plt.ylabel("Total reward")
    plt.xlabel("Episode")
    plt.title(f"Evaluation Results for {name}")

    info_text = f"Softmax: {config['softmax_temperature']}\nEpsilon: {config['epsilon']}\nNeighbor threshold: {config['close_neighbour_threshold_action_selection']}\nMemory threshold: {config['close_neighbour_threshold_memory_update']}\nNoise std dev: {config['noise_std_dev']}\nLatent dim: {config['latent_dim']}"
    plt.annotate(
        info_text,
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        xytext=(10, -30),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.savefig(f"./images/{name}.png")
    plt.clf()  # clear the plot


class TreeMemory:
    def __init__(
        self, autoencoder, max_memory_size=1000, leaf_size=5, use_autoencoder=True
    ):
        self.memory = []  # list of tuples (state, action, reward)
        self.max_memory_size = max_memory_size
        self.leaf_size = leaf_size
        self.kd_tree = None
        self.autoencoder = autoencoder
        self.use_autoencoder = use_autoencoder

    # Method to update the autoencoder used by the memory
    def update_autoencoder(self, autoencoder):
        self.autoencoder = autoencoder

    # Function to encode state observations using the trained autoencoder
    def encode_states(self, autoencoder, states):
        with torch.no_grad():
            inputs = torch.Tensor(states)
            encoded_states, _ = autoencoder(inputs)
            return encoded_states.numpy()

    def __len__(self):
        return len(self.memory)

    def add(self, s_a_v_tuple):
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)

        encoded_state = (
            self.encode_states(self.autoencoder, [s_a_v_tuple[0]])[0]
            if self.use_autoencoder
            else s_a_v_tuple[0]
        )

        self.memory.append((encoded_state, s_a_v_tuple[1], s_a_v_tuple[2]))
        # Rebuild the k-d tree whenever a new entry is added
        self._rebuild_kd_tree()

    def _rebuild_kd_tree(self):
        states = [x[0] for x in self.memory]
        self.kd_tree = KDTree(states, leaf_size=self.leaf_size)

    def lookup(self, state_vector, k=5):
        if len(self.memory) == 0:
            return []

        latent_state = (
            self.encode_states(self.autoencoder, [state_vector])[0]
            if self.use_autoencoder
            else state_vector
        )
        # Find the k nearest neighbors using the k-d tree
        dist, ind = self.kd_tree.query(
            [latent_state], k=len(self.memory) if len(self.memory) < k else k
        )
        neighbors = [(dist[0][i], self.memory[ind[0][i]]) for i in range(len(ind[0]))]

        return neighbors

    def lookup_nearest(self, state_vector):
        if len(self.memory) == 0:
            return None

        latent_state = (
            self.encode_states(self.autoencoder, [state_vector])[0]
            if self.use_autoencoder
            else state_vector
        )

        # Find the nearest neighbor using the k-d tree
        dist, ind = self.kd_tree.query([latent_state], k=1)

        # Nearest neighbor information
        nearest_neighbor_distance = dist[0][0]
        nearest_neighbor_index = ind[0][0]
        nearest_neighbor = (
            nearest_neighbor_distance,
            self.memory[nearest_neighbor_index],
            nearest_neighbor_index,
        )

        return nearest_neighbor

    def update(self, index, state_reward_action_pair):
        encoded_state = (
            self.encode_states(self.autoencoder, [state_reward_action_pair[0]])[0]
            if self.use_autoencoder
            else state_reward_action_pair[0]
        )

        # Update the memory with the new state
        self.memory[index] = (
            encoded_state,
            state_reward_action_pair[1],
            state_reward_action_pair[2],
        )

        # Rebuild the k-d tree after updating the memory
        self._rebuild_kd_tree()


# Define the autoencoder neural network architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim), nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoencoderThreeLayer(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoencoderThreeLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh(),
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Function to train the autoencoder
def train_autoencoder(
    data,
    input_dim,
    latent_dim,
    num_epochs=20,
    autoencoder_type="Standard",
    batch_size=32,
):

    autoencoder = (
        Autoencoder(input_dim, latent_dim)
        if autoencoder_type == "Standard"
        else AutoencoderThreeLayer(input_dim, latent_dim)
    )

    print("Autoencoder type: ", autoencoder_type)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    training_data = torch.Tensor(data)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch, X in enumerate(train_dataloader):

            optimizer.zero_grad()
            _, decoded = autoencoder(X)
            loss = criterion(decoded, X)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader)}"
        )

    return autoencoder


# Function to train the autoencoder
def train_existing_autoencoder(
    data,
    autoencoder,
    num_epochs=10,
    batch_size=32,
):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    training_data = torch.Tensor(data)
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch, X in enumerate(train_dataloader):

            optimizer.zero_grad()
            _, decoded = autoencoder(X)
            loss = criterion(decoded, X)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # print(f"Train existing autoencoder, Loss: {total_loss/len(train_dataloader)}")

    return autoencoder


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state):
        self.buffer.append(state)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class EpisodeMemory:
    def __init__(self):
        self.memory = []  # list of tuples (state, action, reward)

    def __len__(self):
        return len(self.memory)

    def add(self, s_a_r_tuple):
        self.memory.append(s_a_r_tuple)


# how the autoencoder is trained MATTERS a lot, some runs it shows bad results especially with low latent_dim

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10


# Calculate the decayed epsilon
def decayed_epsilon(step):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * step / EPS_DECAY)


hyperparameter_sets_ant = [
    {
        "amount_of_training_to_do": 100,  # 1000
        "eval_loops": 20,  # 100
        "gamma": 0.99,
        "close_neighbour_threshold_action_selection": 0.3,  # d - filter factor
        "close_neighbour_threshold_memory_update": 0.1,  # n - filter factor
        "epsilon": 0.2,
        "noise_std_dev": 0.3,
        "softmax_temperature": 1,
        "max_memory_size": 30000,
        "latent_dim": 8,
        "train_autoencoder": False,
        "autoencoder": "ThreeLayer",
    },
]


# CEC Implementation: https://arxiv.org/pdf/2211.15183.pdf


def train(game, mem, config):
    episode_mem = EpisodeMemory()
    total_rewards = []
    replay_buffer = ReplayBuffer(10000)

    for i in range(config["amount_of_training_to_do"]):
        state, info = game.reset()
        done = False
        total_reward = 0
        end_state = None
        turns = 0

        reason_for_termination = None

        while not done:
            neighbours_with_distance = mem.lookup(state)
            # filter off the far away neighbours
            filtered_neighbours = [
                x
                for x in neighbours_with_distance
                if x[0] < config["close_neighbour_threshold_action_selection"]
            ]

            if len(filtered_neighbours) > 0:
                # Extract actions and values from filtered neighbors
                actions = np.array([x[1][1] for x in filtered_neighbours])
                values = np.array([x[1][2] for x in filtered_neighbours])

                # Softmax action selection
                action_probs = softmax(
                    values, temperature=config["softmax_temperature"]
                )
                selected_action_index = np.random.choice(
                    len(filtered_neighbours), p=action_probs
                )
                action = actions[selected_action_index]

                # Decay epsilon
                # decayed_epsilon_value = decayed_epsilon(i)

                # Add Gaussian noise to the selected action
                if np.random.rand() < config["epsilon"]:
                    action += np.random.normal(
                        0, config["noise_std_dev"], size=action.shape
                    )
            else:
                action = game.action_space.sample()

            # do an action in the environment
            next_state, reward, terminated, truncated, _ = game.step(action)

            episode_mem.add((state, action, reward))

            state = next_state

            done = terminated or truncated

            turns += 1

            total_reward += reward
            replay_buffer.add(state)

            if done:
                if truncated:
                    end_state = "truncated"
                else:
                    end_state = "terminated"
                reason_for_termination = ""
                total_rewards.append(total_reward)

        # if i % 100 == 0 and len(replay_buffer) > 0:
        #     train = config["train_autoencoder"]
        #     print(f"{i} - AutoEncoder training data: {len(replay_buffer)} - {train}")

        #     if config["train_autoencoder"]:
        #         # train autoencoder on the memory
        #         autoencoder = train_existing_autoencoder(
        #             replay_buffer.sample(config["batch_size"]),
        #             mem.autoencoder,
        #         )
        #         mem.update_autoencoder(autoencoder)

        if config["train_autoencoder"] and len(replay_buffer) > config["batch_size"]:
            # train autoencoder on sample from the replay buffer memory
            autoencoder = train_existing_autoencoder(
                replay_buffer.sample(config["batch_size"]),
                mem.autoencoder,
            )
            mem.update_autoencoder(autoencoder)

        v = 0
        while len(episode_mem) > 0:
            state, action, reward = episode_mem.memory.pop()
            # calculate v so we have (s,a,v)
            v = reward + config["gamma"] * v

            # find the closest state neighbours
            closest_state_neighbour = mem.lookup_nearest(state)

            # if there is no close neighbour or the closest neighbour is far away
            if (
                closest_state_neighbour is None
                or closest_state_neighbour[0]
                > config["close_neighbour_threshold_memory_update"]
            ):
                # add the state to the memory, we are far away from the neighbours meaning we have new information
                mem.add((state, action, v))
            else:
                # check if the new value is better than the closest neighbour
                if closest_state_neighbour[1][2] < v:
                    # update the closest neighbour
                    mem.update(closest_state_neighbour[2], (state, action, v))
        if i % 100 == 0:
            print(
                f"Episode {i} after {turns} turns - {total_reward} - {end_state} - {len(mem)} - {reason_for_termination if reason_for_termination is not None else state}"
            )

    return total_rewards, mem


def evaluate(game, mem, config):  # evaluation
    total_rewards_eval = []
    for i in range(config["eval_loops"]):
        state, info = game.reset()
        done = False
        total_reward = 0
        end_state = None
        turns = 0

        while not done:
            closest_neighbour = mem.lookup_nearest(state)

            action = closest_neighbour[1][1]

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
    return total_rewards_eval


games = ["Hopper-v4"]

config = {
    "Ant-v4": {"num_episodes_for_autoencoder": 1000, "epochs": 25},
    "Hopper-v4": {"num_episodes_for_autoencoder": 20000, "epochs": 25},
    "HalfCheetah-v4": {"num_episodes_for_autoencoder": 1000, "epochs": 20},
}


def main():
    for env in games:
        for i, params in enumerate(hyperparameter_sets):
            print(f"Game {env} with hyperparameters {params}")
            game = gym.make(env)

            state_size = game.observation_space.shape[0]
            action_size = game.action_space.shape[0]

            print("State size: ", state_size)
            print("Action size: ", action_size)

            num_episodes_for_autoencoder = params[
                "num_episodes_for_autoencoder"
            ]  # config[env]["num_episodes_for_autoencoder"]
            autoencoder_data = []
            print("Collecting data for autoencoder")
            for _ in range(num_episodes_for_autoencoder):
                state, info = game.reset()
                done = False
                while not done:
                    action = game.action_space.sample()
                    next_state, reward, terminated, truncated, _ = game.step(action)
                    autoencoder_data.append(state)
                    state = next_state
                    done = terminated or truncated

            print("Training autoencoder")
            autoencoder = train_autoencoder(
                autoencoder_data,
                state_size,
                params["latent_dim"],
                params["epochs"],  # config[env]["epochs"],
                params["autoencoder"],
            )
            print("Training autoencoder finished")

            mem = TreeMemory(
                max_memory_size=params["max_memory_size"],
                autoencoder=autoencoder,
                use_autoencoder=params["use_autoencoder"],
            )

            total_rewards, trained_mem = train(game=game, mem=mem, config=params)

            total_rewards_eval = evaluate(game=game, mem=trained_mem, config=params)

            name = f"{env}_{i}_trained_memory.pkl"

            os.makedirs(f"./runs/{name}", exist_ok=True)

            with open(name, "wb") as file:
                pickle.dump(trained_mem, file)

            now = datetime.now()

            now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

            draw(total_rewards, f"{env}_{i}_training_{now_str}", params, use_mean=True)
            draw(total_rewards_eval, f"{env}_{i}_evaluation_{now_str}", params)


main()
