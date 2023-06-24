"""
Module for testing Q Learning algorithm on Frozen Lake environment.
"""

import argparse

import gymnasium as gym
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import matplotlib.pyplot as plt
import numpy as np

from pyrlux import QlearningAgent
from pyrlux.algorithms.exploration_policy import (
    EpsilonGreedy,
    EpsilonGreedyDecay,
    Softmax,
    UCB1,
)
from pyrlux.utils.data_structures import Transition, QLearningParams

from . import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen Lake Q Learning")
    parser.add_argument("-n", type=int, default=10, help="Number of runs to simulate")
    parser.add_argument("-s", type=int, default=6, help="Size of the gridworld")
    parser.add_argument(
        "-p", type=float, default=0.8, help="Probability of a cell being traversable"
    )
    parser.add_argument(
        "--slippery",
        action="store_true",
        default=False,
        help="Whether the environment is slippery",
    )
    parser.add_argument(
        "--show_run",
        action="store_true",
        default=False,
        help="Whether to show a run of the algorithm",
    )
    parser.add_argument(
        "--show_results",
        action="store_true",
        default=False,
        help="Whether to show the results of the algorithm",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10000,
        help="Number of episodes to train the agent",
    )
    parser.add_argument(
        "--exploration_policy",
        type=str,
        default="epsilon_greedy",
        help="Exploration policy, one of epsilon_greedy, epsilon_greedy_decay, softmax, ucb1",
    )
    parser.add_argument("-e", type=float, default=0.1, help="Epsilon")
    parser.add_argument("-d", type=float, default=0.999, help="Decay rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.2, help="Learning rate")
    args = parser.parse_args()

    n = args.n
    s = args.s
    p = args.p
    slippery = args.slippery
    show_run = args.show_run
    show_results = args.show_results
    num_iters = args.num_iters
    exploration_policy = args.exploration_policy
    e = args.e
    d = args.d
    gamma = args.gamma
    alpha = args.alpha

    print(
        f"Parameters: s={s}, p={p}, slippery={slippery}, num_iters={num_iters}, alpha={alpha}"
    )

    frozen_lake_map = generate_random_map(size=s, p=p)
    env: FrozenLakeEnv = gym.make(
        "FrozenLake-v1", desc=frozen_lake_map, is_slippery=slippery
    )

    if exploration_policy == "epsilon_greedy":
        exploration_policy = EpsilonGreedy(epsilon=e)
    elif exploration_policy == "epsilon_greedy_decay":
        exploration_policy = EpsilonGreedyDecay(epsilon=e, decay=d)
    elif exploration_policy == "softmax":
        exploration_policy = Softmax()
    elif exploration_policy == "ucb1":
        exploration_policy = UCB1(env.observation_space.n, env.action_space.n)
    else:
        raise ValueError("Invalid exploration policy")

    agent = QlearningAgent(
        params=QLearningParams(
            alpha=alpha,
            gamma=gamma,
            num_states=env.observation_space.n,
            num_actions=env.action_space.n,
        ),
        exploration_policy=exploration_policy,
    )

    exploration = np.zeros(env.observation_space.n)

    pbar = tqdm(total=num_iters, desc="Training")
    while num_iters > 0:
        observation, _ = env.reset()
        terminated, truncated = False, False
        while (num_iters > 0) and not (terminated or truncated):
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)

            exploration[observation] += 1

            transition = Transition(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                done=terminated,
            )

            agent.update(transition)
            num_iters -= 1
            pbar.update(1)

            observation = next_observation
    pbar.close()

    policy = agent.get_policy()

    if show_results:
        utils.plot_exploration(exploration, (s, s))
        utils.plot_policy(policy, (s, s))
        utils.plot_map(frozen_lake_map, (s, s))
        utils.plot_value_function(agent.get_value_function(), (s, s))
        plt.show()

    if n > 0:
        utils.evaluate_policy(n, env, policy)

    if show_run:
        input("Press enter to see a run of the algorithm")
        utils.show_run(slippery, frozen_lake_map, policy)
