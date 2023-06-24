"""
Data structures used in the project.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MDP:
    """
    Markov Decision Process.
    """

    transition_matrix: np.ndarray  # shape: (S', A, S)
    reward_matrix: np.ndarray  # shape: (S', A, S)
    gamma: float

    num_states: int
    num_actions: int


@dataclass(frozen=True)
class Transition:
    """
    Transition.
    """

    state: int
    action: int
    reward: float
    next_state: int
    done: bool


@dataclass(frozen=True)
class PolicyIterationParams:
    """
    Parameters for the Policy Iteration algorithm.
    """

    num_iters: int
    num_iters_eval: int
    theta_eval: float = 1e-3
    device: torch.device = torch.device("cpu")


@dataclass(frozen=True)
class ValueIterationParams:
    """
    Parameters for the Value Iteration algorithm.
    """

    num_iters: int
    theta: float = 1e-3
    device: torch.device = torch.device("cpu")


@dataclass(frozen=True)
class QLearningParams:
    """
    Parameters for the Q-Learning algorithm.
    """

    alpha: float
    num_states: int
    num_actions: int
    gamma: float = 0.9


@dataclass(frozen=True)
class SarsaParams:
    """
    Parameters for the SARSA algorithm.
    """

    alpha: float
    num_states: int
    num_actions: int
    gamma: float = 0.9
