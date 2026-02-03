"""
Compound Poisson Random Walk Simulation

Exact path simulation for finite-activity jump processes.
"""

import numpy as np
from typing import Tuple
from src.jump_kernel import compute_total_rate, compute_jump_probabilities


def sample_compound_poisson(
    t: float,
    J: np.ndarray,
    rates: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample X_t from compound Poisson process.
    
    X_t = sum_{j=1}^{N_t} xi_j
    where N_t ~ Pois(q*t) and xi_j are i.i.d. with law pi.
    
    Args:
        t: Time parameter
        J: Jump vectors [M, m]
        rates: Jump rates [M]
        rng: Random number generator (seed=14)
        
    Returns:
        X_t: Final position in Z^m
    """
    q = compute_total_rate(rates)
    pi = compute_jump_probabilities(rates)
    
    N_t = rng.poisson(q * t)
    
    if N_t == 0:
        return np.zeros(J.shape[1], dtype=np.int64)
    
    jump_indices = rng.choice(len(J), size=N_t, p=pi)
    total_jump = np.sum(J[jump_indices], axis=0)
    
    return total_jump.astype(np.int64)


def sample_two_paths(
    t: float,
    J: np.ndarray,
    rates: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample two independent copies X_t and X'_t.
    
    Args:
        t: Time parameter
        J: Jump vectors
        rates: Jump rates
        rng: Random number generator
        
    Returns:
        X_t, X'_t: Two independent paths
    """
    X_t = sample_compound_poisson(t, J, rates, rng)
    X_prime_t = sample_compound_poisson(t, J, rates, rng)
    return X_t, X_prime_t
