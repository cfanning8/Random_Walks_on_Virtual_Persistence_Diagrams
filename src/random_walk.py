"""
Compound Poisson Simulation for Levy Process

Generalized to work with arbitrary LevyMeasure.
"""

import numpy as np
from src.jump_kernel import LevyMeasure


def simulate_levy_process(
    levy_measure: LevyMeasure,
    t: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate X_t on H for finite-activity Levy measure.
    
    For truncated nu_R (finite mass), the process has compound Poisson representation:
    X_t = sum_{k=1}^{N_t} xi_k
    
    where N_t ~ Pois(q*t) and xi_k are i.i.d. with distribution pi(kappa) = nu(kappa)/q.
    
    Args:
        levy_measure: LevyMeasure instance (must have finite total mass)
        t: Time parameter
        rng: NumPy random number generator
        
    Returns:
        coeffs: Integer np.ndarray shape (m,) representing X_t in basis
    """
    if levy_measure.K == 0:
        return np.zeros(levy_measure.m, dtype=np.int64)
    
    total_rate = levy_measure.total_mass()
    if total_rate == 0:
        return np.zeros(levy_measure.m, dtype=np.int64)
    
    # Number of jumps
    N_t = rng.poisson(total_rate * t)
    
    if N_t == 0:
        return np.zeros(levy_measure.m, dtype=np.int64)
    
    # Discrete distribution over jumps
    probs = levy_measure.jump_probabilities()
    
    # Sample jump indices
    jump_indices = rng.choice(levy_measure.K, size=N_t, p=probs)
    
    # Sum jumps
    jumps = levy_measure.jumps[jump_indices]  # shape [N_t, m]
    return jumps.sum(axis=0).astype(np.int64)


def sample_two_paths(
    levy_measure: LevyMeasure,
    t: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample two independent copies of the Levy process.
    
    Args:
        levy_measure: LevyMeasure instance
        t: Time parameter
        rng: NumPy random number generator
        
    Returns:
        X_t, X_t': Two independent copies
    """
    X_t = simulate_levy_process(levy_measure, t, rng)
    X_t_prime = simulate_levy_process(levy_measure, t, rng)
    return X_t, X_t_prime
