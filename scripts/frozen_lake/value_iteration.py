"""
Module for testing Value Iteration algorithm on Frozen Lake environment.
"""

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map

from pyrlux import ValueIteration
from pyrlux.utils.data_structures import ValueIterationParams

from . import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen Lake Value Iteration")
    parser.add_argument("-n", type=int, default=10, help="Number of runs to simulate")
    parser.add_argument("-s", type=int, default=8, help="Size of the gridworld")
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
        default=100,
        help="Number of iterations of value iteration",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1e-3,
        help="Threshold for convergence",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu or cuda:0)"
    )
    args = parser.parse_args()

    n = args.n
    s = args.s
    p = args.p
    slippery = args.slippery
    show_run = args.show_run
    show_results = args.show_results
    num_iters = args.num_iters
    theta = args.theta
    gamma = args.gamma
    device = torch.device(args.device)

    print(
        f"Parameters: s={s}, p={p}, slippery={slippery}, theta={theta}, "
        f" num_iter={num_iters}, theta={theta}, device={device}"
    )

    frozen_lake_map = generate_random_map(size=s, p=p)
    env: FrozenLakeEnv = gym.make(
        "FrozenLake-v1", desc=frozen_lake_map, is_slippery=slippery
    )

    mdp_model = utils.extract_mdp(env, gamma=gamma)

    Vs, Pis, deltas = ValueIteration(
        mdp_model,
        ValueIterationParams(num_iters=num_iters, theta=theta, device=device),
    ).run()

    policy = Pis[-1]

    if show_results:
        utils.plot_map(frozen_lake_map, (s, s))
        utils.plot_value_function(Vs[-1], (s, s))
        utils.plot_policy(policy, (s, s))

        plt.show()

    if n > 0:
        utils.evaluate_policy(n, env, policy)

    if show_run:
        utils.show_run(slippery, frozen_lake_map, policy)
