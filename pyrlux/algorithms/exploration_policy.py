"""
Module implementing exploration policies for Reinforcement Learning algorithms.
"""

import random
import numpy as np


class ExplorationPolicy:
    """
    Base class for exploration policies.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, state, actions, q_values):
        raise NotImplementedError("ExplorationPolicy.__call__")


class EpsilonGreedy(ExplorationPolicy):
    """
    Epsilon-greedy exploration policy.
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        """
        Args:
            epsilon (float, optional): Probability of selecting a random action. Defaults to 0.1.
        """
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, state: int, actions: np.ndarray, q_values: np.ndarray):
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
            state (int): Current state.
            actions (np.ndarray): List of possible actions from the current state.
            q_values (np.ndarray): Q-values
        """
        if random.random() < self.epsilon:
            return random.choice(actions)
        return actions[np.argmax(q_values[state, actions])]


class EpsilonGreedyDecay(ExplorationPolicy):
    """
    Epsilon-greedy exploration policy with decay.
    """

    def __init__(self, epsilon: float = 0.1, decay: float = 0.999) -> None:
        """
        Args:
            epsilon (float, optional): Probability of selecting a random action. Defaults to 0.1.
            decay (float, optional): Decay rate. Defaults to 0.999.
        """
        super().__init__()
        self.epsilon = epsilon
        self.decay = decay

    def __call__(self, state: int, actions: np.ndarray, q_values: np.ndarray):
        """
        Selects an action based on the epsilon-greedy policy.

        Args:
            state (int): Current state.
            actions (np.ndarray): List of possible actions from the current state.
            q_values (np.ndarray): Q-values
        """
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(q_values[state, actions])]
        self.update()
        return action

    def update(self):
        """
        Updates epsilon.
        """
        self.epsilon *= self.decay


class Softmax(ExplorationPolicy):
    """
    Softmax exploration policy.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        """
        Args:
            temperature (float, optional): Temperature. Defaults to 0.1.
        """
        super().__init__()
        self.temperature = temperature

    def __call__(self, state: int, actions: np.ndarray, q_values: np.ndarray):
        """
        Selects an action based on the softmax policy.

        Args:
            state (int): Current state.
            actions (np.ndarray): List of possible actions from the current state.
            q_values (np.ndarray): Q-values
        """
        probabilities = np.exp(q_values[state, actions] / self.temperature)
        probabilities /= np.sum(probabilities)
        return np.random.choice(actions, p=probabilities)


class UCB1(ExplorationPolicy):
    """
    UCB1 exploration policy.
    """

    def __init__(self, nun_states: int, num_actions: int) -> None:
        """
        Args:
            c (float, optional): Exploration parameter. Defaults to 1.0.
        """
        super().__init__()
        self.nun_states = nun_states
        self.num_actions = num_actions

        self.time_selected = np.zeros((nun_states, num_actions))

    def __call__(self, state: int, actions: np.ndarray, q_values: np.ndarray):
        """
        Selects an action based on the UCB1 policy.

        Args:
            state (int): Current state.
            actions (np.ndarray): List of possible actions from the current state.
            q_values (np.ndarray): Q-values
        """
        total = np.sum(self.time_selected[state])
        if total == 0:
            return random.choice(actions)
        return actions[
            np.argmax(
                q_values[state, actions]
                + np.sqrt(2 * np.log(total) / self.time_selected[state, actions])
            )
        ]
