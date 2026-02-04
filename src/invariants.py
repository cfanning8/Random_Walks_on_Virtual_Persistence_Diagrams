"""
Heat Kernel Invariants

Computation of Z_H(t), E_H(t), C_H(t), R_{s,H}(0) via Monte Carlo.
"""

import numpy as np
from typing import Tuple
from src.random_walk import sample_compound_poisson, sample_two_paths
from src.jump_kernel import build_jump_kernel


def lambda_H(theta: np.ndarray, m: int, lambda_val: float = 1.0) -> float:
    """
    Compute Levy-Khintchine exponent for nearest-neighbor kernel.
    
    lambda_H(xi) = 2*lambda * sum_{i=1}^m (1 - cos(xi_i))
    
    Args:
        theta: Character in [-pi, pi)^m
        m: Dimension
        lambda_val: Jump rate parameter
        
    Returns:
        lambda_H(theta)
    """
    return 2.0 * lambda_val * np.sum(1.0 - np.cos(theta))


def estimate_Z_H(
    tau: float,
    m: int,
    lambda_val: float = 1.0,
    N: int = 100000,
    seed: int = 14
) -> float:
    """
    Estimate return profile Z_H(tau) = p_tau(0) via Monte Carlo.
    
    Args:
        tau: Dimensionless time parameter
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimated Z_H(tau)
    """
    rng = np.random.default_rng(seed)
    J, rates = build_jump_kernel(m, lambda_val)
    # With lambda=1, tau = t directly (no scaling needed)
    t = tau
    
    count = 0
    for _ in range(N):
        X_t = sample_compound_poisson(t, J, rates, rng)
        if np.array_equal(X_t, np.zeros(m, dtype=np.int64)):
            count += 1
    
    return count / N


def estimate_C_H(
    tau: float,
    m: int,
    lambda_val: float = 1.0,
    N: int = 100000,
    seed: int = 14
) -> float:
    """
    Estimate collision profile C_H(tau) = P(X_tau = X'_tau) via Monte Carlo.
    
    Args:
        tau: Dimensionless time parameter
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimated C_H(tau)
    """
    rng = np.random.default_rng(seed)
    J, rates = build_jump_kernel(m, lambda_val)
    # With lambda=1, tau = t directly (no scaling needed)
    t = tau
    
    count = 0
    for _ in range(N):
        X_t, X_prime_t = sample_two_paths(t, J, rates, rng)
        if np.array_equal(X_t, X_prime_t):
            count += 1
    
    return count / N


def estimate_E_H(
    tau: float,
    m: int,
    lambda_val: float = 1.0,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate energy profile E_H(tau) via Monte Carlo on theta-space.
    
    E_H(tau) = int lambda_H(theta) * exp(-tau*lambda_H(theta)) dmu
    
    Args:
        tau: Dimensionless time parameter
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimated E_H(tau)
    """
    rng = np.random.default_rng(seed)
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_H(theta, m, lambda_val)
        # Spectral side: exp(-tau * lambda_H(theta))
        acc += lam * np.exp(-tau * lam)
    
    return acc / N


def estimate_E_H_2(
    tau: float,
    m: int,
    lambda_val: float = 1.0,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate E_H^(2)(tau) = energy of kernel section k_t(·,0).
    
    E_H^(2)(tau) = int lambda_H(theta) * exp(-2*tau*lambda_H(theta)) dmu
    
    This is the Dirichlet energy of the kernel section, used for Sobolev bounds.
    
    Args:
        tau: Dimensionless time parameter
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimated E_H^(2)(tau)
    """
    rng = np.random.default_rng(seed)
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_H(theta, m, lambda_val)
        # Energy of kernel section: exp(-2*tau*lambda_H)
        acc += lam * np.exp(-2.0 * tau * lam)
    
    return acc / N


def estimate_Z_H_2(
    tau: float,
    m: int,
    lambda_val: float = 1.0,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate Z_H^(2)(tau) = ||k_t(·,0)||_2^2.
    
    Z_H^(2)(tau) = int exp(-2*tau*lambda_H(theta)) dmu
    
    This is the L^2 norm squared of the kernel section, used for Sobolev bounds.
    
    Args:
        tau: Dimensionless time parameter
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimated Z_H^(2)(tau)
    """
    rng = np.random.default_rng(seed)
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_H(theta, m, lambda_val)
        # L^2 norm squared: exp(-2*tau*lambda_H)
        acc += np.exp(-2.0 * tau * lam)
    
    return acc / N


def estimate_R_s_H(
    s: float,
    m: int,
    lambda_val: float = 1.0,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate resolvent R_{s,H}(0) via Monte Carlo on theta-space.
    
    R_{s,H}(0) = int 1/(s + lambda_H(theta)) dmu
    
    Args:
        s: Resolvent parameter (should be s = lambda_val)
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimated R_{s,H}(0)
    """
    rng = np.random.default_rng(seed)
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_H(theta, m, lambda_val)
        acc += 1.0 / (s + lam)
    
    return acc / N
