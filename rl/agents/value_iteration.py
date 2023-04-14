import numpy as np
import torch
from tqdm import tqdm
from rl.markov_process.mdp import MDP

def value_iteration(mdp_model: MDP, num_iter: int, device: torch.device, theta: float=1e-3) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Value iteration algorithm for MDPs.

    Args:
        mdp_model: MDP model
        num_iter: number of iterations
        device: torch device
        theta: convergence threshold
    
    Returns:
        Vs: list of value functions
        Pis: list of policies

    Notes:
        Pis[i] is the policy that maximizes Vs[i+1]
        Vs[0] is the initial value function
    """

    Vs = [torch.zeros(mdp_model.num_states).to(device)]
    Pis = []
    deltas = []

    t = torch.from_numpy(mdp_model.transition_matrix).to(device)
    r = torch.from_numpy(mdp_model.reward_matrix).to(device)
    gamma = torch.tensor(mdp_model.gamma).to(device)

    for i in tqdm(iterable=range(num_iter)):
        expected_rewards = torch.einsum('ijk, ijk -> kj', t, (r + gamma * Vs[i][:, None, None]))
        max_rewards, argmax_rewards = torch.max(expected_rewards, dim=1)
        Vs.append(max_rewards)
        Pis.append(argmax_rewards.cpu().numpy())

        delta = torch.abs(Vs[i] - Vs[i+1]).max().item()
        deltas.append(delta)

        if delta < theta:
            break
    
    Vs = [v.cpu().numpy() for v in Vs]

    return Vs, Pis, deltas