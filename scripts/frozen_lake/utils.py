"""
Module containing utility functions for FrozenLake environment.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from tqdm import trange

from pyrlux.utils.data_structures import MDP

color_map = {
    "S": "green",
    "F": "white",
    "H": "black",
    "G": "gold",
}


def plot_policy(policy: np.ndarray, shape: tuple[int, int]) -> None:
    """
    Plot a policy as a grid using triangle symbols

    Args:
        policy (np.ndarray): policy of each state
        shape (tuple): shape of the gridworld

    Returns:
        None

    Notes:
        Action mapping:
            - 0: Move left
            - 1: Move down
            - 2: Move right
            - 3: Move up
    """
    policy = policy.reshape(shape)

    _, ax = plt.subplots(figsize=(5, 5))
    _ = ax.imshow(policy, cmap="gray", interpolation="none", vmin=0, vmax=3)

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(shape[0]):
        for j in range(shape[1]):
            arrow = "\u2190\u2193\u2192\u2191"[policy[i, j]]
            ax.text(j, i, arrow, color="g", size=20, ha="center", va="center")

    ax.set_title("Policy")


def plot_value_function(values: np.ndarray, shape: tuple[int, int]) -> None:
    """
    Plot a value function as a grid

    Args:
        values (np.ndarray): value function of each state
        shape (tuple): shape of the gridworld

    Returns:
        None
    """
    values = values.reshape(shape)
    values[-1][-1] = values.max()

    _, ax = plt.subplots(figsize=(5, 5))
    if values.max() == 0:
        im = ax.imshow(values, cmap="YlOrRd", interpolation="none")
    else:
        im = ax.imshow(values, cmap="YlOrRd", interpolation="none", norm="log")

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.figure.colorbar(im, ax=ax, orientation="horizontal")

    ax.set_title("Value function")


def plot_deltas(deltas: list[float]) -> None:
    """
    Plot deltas of value iteration

    Args:
        deltas (list[float]): delta of each iteration

    Returns:
        None
    """
    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(deltas)

    ax.set_title("Deltas")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Delta")


def plot_map(map_string: list[str], shape: tuple[int, int]) -> None:
    """
    Plot a map as a grid

    Args:
        map (list[str]): map of the gridworld (S: start, F: frozen, H: hole, G: goal)
        shape (tuple): shape of the gridworld

    Returns:
        None
    """
    _, ax = plt.subplots(figsize=(5, 5))

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(
                j,
                shape[0] - 1 - i,
                map_string[i][j],
                color=color_map[map_string[i][j]],
                size=20,
                ha="center",
                va="center",
            )

    # Add title
    ax.set_title("Map")


def extract_mdp(env: FrozenLakeEnv, gamma: float = 0.9) -> MDP:
    """
    Extract MDP from FrozenLakeEnv

    Args:
        env (FrozenLakeEnv): FrozenLakeEnv

    Returns:
        MDP (MDP):
    """
    transition_matrix = np.zeros(
        (env.observation_space.n, env.action_space.n, env.observation_space.n)
    )
    reward_matrix = np.zeros(
        (env.observation_space.n, env.action_space.n, env.observation_space.n)
    )

    for i in range(env.observation_space.n):
        for j in range(env.action_space.n):
            for transition in env.P[i][j]:
                k = transition[1]
                transition_matrix[k, j, i] = transition[0]
                reward_matrix[k, j, i] = transition[2]

    mdp_model = MDP(
        transition_matrix=transition_matrix,
        reward_matrix=reward_matrix,
        gamma=gamma,
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
    )

    return mdp_model


def evaluate_policy(n: int, env: FrozenLakeEnv, policy: np.ndarray) -> None:
    """
    Evaluate a policy by running n episodes

    Args:
        n (int): number of episodes
        env (FrozenLakeEnv): FrozenLakeEnv
        policy (np.ndarray): policy

    Returns:
        None
    """
    wins = 0
    for _ in trange(n):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = policy[observation]
            observation, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 1.0:
                wins += 1

    print(f"Win rate: {wins/n:.2%}")
    env.close()


def show_run(slippery: bool, frozen_lake_map: list[str], policy: np.ndarray):
    """
    Show a run of the policy

    Args:
        slippery (bool): whether the environment is slippery
        frozen_lake_map (list[str]): map of the gridworld (S: start, F: frozen, H: hole, G: goal)
        policy (np.ndarray): policy

    Returns:
        None
    """
    env: FrozenLakeEnv = gym.make(
        "FrozenLake-v1",
        desc=frozen_lake_map,
        is_slippery=slippery,
        render_mode="human",
    )

    observation, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = policy[observation]
        observation, _, terminated, truncated, _ = env.step(action)
    env.close()
