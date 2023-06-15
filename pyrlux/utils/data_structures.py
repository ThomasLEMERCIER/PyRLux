from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MDP:
    transition_matrix: np.ndarray  # shape: (S', A, S)
    reward_matrix: np.ndarray  # shape: (S', A, S)
    gamma: float

    num_states: int
    num_actions: int


@dataclass(frozen=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


@dataclass(frozen=True)
class PolicyIterationParams:
    num_iters: int
    num_iters_eval: int
    theta_eval: float = 1e-3
    device: torch.device = torch.device("cpu")

@dataclass(frozen=True)
class ValueIterationParams:
    num_iters: int
    theta: float = 1e-3
    device: torch.device = torch.device("cpu")
