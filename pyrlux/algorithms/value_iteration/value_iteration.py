"""
Module for Value Iteration algorithm.
"""

import numpy as np
import torch
from tqdm import trange
from pyrlux.utils.data_structures import MDP, ValueIterationParams


class ValueIteration:
    """
    Value Iteration algorithm.
    """

    def __init__(
        self,
        mdp_model: MDP,
        params: ValueIterationParams,
    ):
        """
        Args:
            mdp_model (MDP): MDP model.
            params (ValueIterationParams): Parameters for the algorithm.
        """
        self.mdp_model = mdp_model
        self.params = params

        self.transition = torch.from_numpy(mdp_model.transition_matrix).to(
            self.params.device
        )
        self.reward = torch.from_numpy(mdp_model.reward_matrix).to(self.params.device)
        self.gamma = torch.tensor(mdp_model.gamma).to(self.params.device)

    def run(self) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """
        Run the algorithm.
        """
        value_functions = [
            torch.zeros(self.mdp_model.num_states).to(self.params.device)
        ]
        policies = []
        deltas = []

        for i in trange(self.params.num_iters):
            q_values = torch.einsum(
                "ijk, ijk -> kj",
                self.transition,
                (self.reward + self.gamma * value_functions[i][:, None, None]),
            )
            max_q_values, argmax_q_values = torch.max(q_values, dim=1)
            value_functions.append(max_q_values)
            policies.append(argmax_q_values.cpu().numpy())

            delta = torch.abs(value_functions[i] - value_functions[i + 1]).max().item()
            deltas.append(delta)

            if delta < self.params.theta:
                break

        value_functions = [v.cpu().numpy() for v in value_functions]

        return value_functions, policies, deltas
