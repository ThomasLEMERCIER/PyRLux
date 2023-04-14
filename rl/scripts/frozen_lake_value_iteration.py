import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

from rl.agents.value_iteration import value_iteration
from rl.markov_process.mdp import MDP


def plot_policy(policy: np.ndarray, shape: tuple):
    """ Plot a policy as a grid using triangle symbols

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

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(policy, cmap='gray', interpolation='none', vmin=0, vmax=3)

    # Add gridlines
    ax.set_xticks(np.arange(-.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    # Add arrows
    for y in range(policy.shape[0]):
        for x in range(policy.shape[1]):
            arrow = u"\u2190\u2193\u2192\u2191"[policy[y, x]]
            ax.text(x, y, arrow, color='g', size=20, ha='center', va='center')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], orientation='horizontal')
    cbar.ax.set_xticklabels(['Move left', 'Move down', 'Move right', 'Move up'])

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    ax.set_title('Policy')

def plot_value_function(values: np.ndarray, shape: tuple):
    """ Plot a value function as a grid

    Args:
        values (np.ndarray): value function of each state
        shape (tuple): shape of the gridworld

    Returns:
        None
    """
    values = values.reshape(shape)
    values[-1][-1] = values.max()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(values, cmap='YlOrRd', interpolation='none', norm="log")
    # Add gridlines
    ax.set_xticks(np.arange(-.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    ax.set_title('Value function')

def plot_deltas(deltas: list[float]):
    """ Plot deltas of value iteration

    Args:
        deltas (list[float]): delta of each iteration

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(deltas)

    # Add title
    ax.set_title('Deltas')

def plot_result(values: np.ndarray, policy: np.ndarray, deltas: list[float], shape: tuple):
    """ Plot result of value iteration

    Args:
        values (np.ndarray): value function of each state
        policy (np.ndarray): policy of each state
        deltas (np.ndarray): delta of each iteration
        shape (tuple): shape of the gridworld

    Returns:
        None
    """

    plot_deltas(deltas)
    plot_value_function(values, shape)
    plot_policy(policy, shape)

    plt.show()

def extract_mdp(env: FrozenLakeEnv):
    """ Extract MDP from FrozenLakeEnv

    Args:
        env (FrozenLakeEnv): FrozenLakeEnv

    Returns:
        MDP (MDP): 
    """
    transition_matrix = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    reward_matrix = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    for i in range(env.observation_space.n):
        for j in range(env.action_space.n):
            for transition in env.P[i][j]:
                k = transition[1]
                transition_matrix[k, j, i] = transition[0]
                reward_matrix[k, j, i] = transition[2]

    mdp_model = MDP(transition_matrix=transition_matrix, reward_matrix=reward_matrix, gamma=0.9, num_states=env.observation_space.n, num_actions=env.action_space.n)
    return mdp_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Frozen Lake Value Iteration")
    parser.add_argument("--n", type=int, default=8, help="Size of the gridworld")
    parser.add_argument("--p", type=float, default=0.8, help="Probability of a cell being traversable")
    parser.add_argument("--slippery", action="store_true", default=False, help="Whether the environment is slippery")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--theta", type=float, default=1e-3, help="Threshold for convergence")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device)
    n = args.n
    p = args.p
    slippery = args.slippery
    theta = args.theta
    num_iter = args.num_iter

    print(f"Parameters: n={n}, p={p}, slippery={slippery}, theta={theta}, num_iter={num_iter}, device={device}")

    frozen_lake_map = generate_random_map(size=n, p=p)
    env : FrozenLakeEnv = gym.make('FrozenLake-v1', desc=frozen_lake_map, is_slippery=slippery, render_mode="human")
    observation, info = env.reset()

    mdp_model = extract_mdp(env)

    Vs, Pis, deltas = value_iteration(mdp_model, theta=theta, num_iter=num_iter, device=device)

    plot_result(Vs[-1], Pis[-1], deltas, (n, n))

    for _ in range(1000):
        action = Pis[-1][observation]
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()