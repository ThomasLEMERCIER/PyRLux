""" 
Module for Policy Iteration algorithm. 
"""

import numpy as np
import torch
from tqdm import trange

from pyrlux.utils.data_structures import MDP, PolicyIterationParams


class PolicyIteration:
    """
    Policy Iteration algorithm.
    """

    def __init__(
        self,
        mdp_model: MDP,
        params: PolicyIterationParams,
    ):
        """
        Args:
            mdp_model (MDP): MDP model.
            params (PolicyIterationParams): Parameters for the algorithm.
        """
        self.mdp_model = mdp_model
        self.params = params

        self.transition = torch.from_numpy(mdp_model.transition_matrix).to(
            self.params.device
        )
        self.reward = torch.from_numpy(mdp_model.reward_matrix).to(self.params.device)
        self.gamma = torch.tensor(mdp_model.gamma).to(self.params.device)

    def policy_evaluation(self, policy: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a policy.

        Args:
            policy (torch.Tensor): Policy to evaluate.

        Returns:
            torch.Tensor: Value function of the policy.
        """
        value_functions = [
            torch.zeros(self.mdp_model.num_states).to(self.params.device)
        ]

        for i in range(self.params.num_iters_eval):
            values = torch.einsum(
                "ij, ij -> j",
                self.transition[:, policy, range(self.mdp_model.num_states)],
                (
                    self.reward[:, policy, range(self.mdp_model.num_states)]
                    + self.gamma * value_functions[i][:, None]
                ),
            )
            value_functions.append(values)

            if (
                torch.abs(value_functions[i] - value_functions[i + 1]).max().item()
                < self.params.theta_eval
            ):
                break

        return value_functions[-1]

    def run(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Run the algorithm.
        """
        value_functions = []
        policies = [
            torch.randint(
                0, self.mdp_model.num_actions, (self.mdp_model.num_states,)
            ).to(self.params.device)
        ]

        for i in trange(self.params.num_iters):
            value_functions.append(self.policy_evaluation(policies[i]))
            q_values = torch.einsum(
                "ijk, ijk -> kj",
                self.transition,
                (self.reward + self.gamma * value_functions[i][:, None, None]),
            )
            _, argmax_q_values = torch.max(q_values, dim=1)
            policies.append(argmax_q_values)

            if torch.equal(policies[i], policies[i + 1]):
                break

        value_functions = [v.cpu().numpy() for v in value_functions]
        policies = [p.cpu().numpy() for p in policies]

        return value_functions, policies
