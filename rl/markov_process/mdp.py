import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class MDP:

    transition_matrix: np.ndarray # shape: (S, A, S)
    reward_matrix: np.ndarray # shape: (S, A, S)
    gamma: float

    num_states: int
    num_actions: int
