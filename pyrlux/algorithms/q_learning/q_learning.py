"""
Module for Q-learning algorithm.
"""

import numpy as np

from pyrlux.algorithms.exploration_policy import ExplorationPolicy
from pyrlux.utils.data_structures import Transition, QLearningParams


class QlearningAgent:
    """
    Q-learning agent.
    """

    def __init__(
        self,
        params: QLearningParams,
        exploration_policy: ExplorationPolicy,
    ) -> None:
        """
        Args:
            alpha (float): Learning rate.
            num_states (int): Number of states.
            num_actions (int): Number of actions.
            gamma (float): Discount factor.
            exploration_policy (ExplorationPolicy): Exploration policy.
        """
        self.params = params
        self.exploration_policy = exploration_policy

        self.q_values = np.zeros((self.params.num_states, self.params.num_actions))

    def update(self, transition: Transition) -> None:
        """
        Updates the Q-values.

        Args:
            transition (Transition): Transition.
        """
        delta = (
            transition.reward
            + self.params.gamma * np.max(self.q_values[transition.next_state])
            - self.q_values[transition.state, transition.action]
        )
        self.q_values[transition.state, transition.action] += self.params.alpha * delta

    def act(self, state: int) -> int:
        """
        Selects an action based on the exploration policy.

        Args:
            state (int): Current state.
        """
        return self.exploration_policy(
            state, np.arange(self.q_values.shape[1]), self.q_values
        )

    def get_best_action(self, state: int) -> np.int64:
        """
        Returns the best action for a given state.

        Args:
            state (int): Current state.

        Returns:
            np.int64: Best action.
        """
        return np.argmax(self.q_values[state])

    def get_policy(self) -> np.ndarray:
        """
        Returns the policy.
        """
        return np.argmax(self.q_values, axis=1)

    def get_value_function(self) -> np.ndarray:
        """
        Returns the value function.
        """
        return np.max(self.q_values, axis=1)
