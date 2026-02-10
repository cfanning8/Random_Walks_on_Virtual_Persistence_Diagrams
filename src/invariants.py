"""
Heat Kernel Invariants

Computation of p_t(0), sum_h p_t(h)^2, -d/dt p_t(0), G_s(0,0) via Monte Carlo.
All functions use generic LevyMeasure, not hard-coded nearest-neighbor.
"""

import numpy as np
from typing import Callable
from src.jump_kernel import LevyMeasure, lambda_symbol


def estimate_return_probability(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 100000,
    seed: int = 14
) -> float:
    """
    Estimate return probability p_t(0) via spectral integration.
    
    p_t(0) = int_{hat(H)} exp(-t*lambda_H(theta)) dmu(theta)
    
    Args:
        t: Time parameter
        levy_measure: LevyMeasure instance (may be truncated)
        N: Number of Monte Carlo samples for theta-space integration
        seed: Random seed
        
    Returns:
        Estimate of p_t(0)
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += np.exp(-t * lam)
    
    return acc / N


def estimate_collision_probability(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 100000,
    seed: int = 14
) -> float:
    """
    Estimate collision probability sum_h p_t(h)^2 via spectral integration.
    
    sum_{h in H} p_t(h)^2 = int_{hat(H)} exp(-2*t*lambda_H(theta)) dmu(theta)
    
    This equals P(X_t = X_t') for two independent copies.
    
    Args:
        t: Time parameter
        levy_measure: LevyMeasure instance (may be truncated)
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimate of sum_h p_t(h)^2
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += np.exp(-2.0 * t * lam)
    
    return acc / N


def estimate_energy_derivative(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate energy -d/dt p_t(0) via spectral integration.
    
    -d/dt p_t(0) = int_{hat(H)} lambda_H(theta) * exp(-t*lambda_H(theta)) dmu(theta)
    
    Args:
        t: Time parameter
        levy_measure: LevyMeasure instance (may be truncated)
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimate of -d/dt p_t(0)
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += lam * np.exp(-t * lam)
    
    return acc / N


def estimate_resolvent_diagonal(
    s: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate resolvent diagonal G_s(0,0) via spectral integration.
    
    G_s(0,0) = int_{hat(H)} 1/(s + lambda_H(theta)) dmu(theta)
    
    Args:
        s: Resolvent parameter
        levy_measure: LevyMeasure instance (may be truncated)
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimate of G_s(0,0)
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += 1.0 / (s + lam)
    
    return acc / N


def estimate_kernel_section_energy(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate energy of kernel section k_t(·,0).
    
    E_H^(2)(t) = int lambda_H(theta) * exp(-2*t*lambda_H(theta)) dmu
    
    This is the Dirichlet energy of the kernel section, used for Sobolev bounds.
    
    Args:
        t: Time parameter
        levy_measure: LevyMeasure instance (may be truncated)
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimate of E_H^(2)(t)
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += lam * np.exp(-2.0 * t * lam)
    
    return acc / N


def estimate_kernel_section_l2_squared(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate L^2 norm squared of kernel section k_t(·,0).
    
    Z_H^(2)(t) = ||k_t(·,0)||_2^2 = int exp(-2*t*lambda_H(theta)) dmu
    
    This is used for Sobolev bounds.
    
    Args:
        t: Time parameter
        levy_measure: LevyMeasure instance (may be truncated)
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimate of ||k_t(·,0)||_2^2
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += np.exp(-2.0 * t * lam)
    
    return acc / N


def estimate_Ht_norm_squared(
    f_values: np.ndarray,
    basis_points: list,
    levy_measure: LevyMeasure,
    t: float,
    N: int = 50000,
    seed: int = 14
) -> float:
    """
    Estimate RKHS norm squared |f|_{H_t}^2 for finitely supported function f.
    
    |f|_{H_t}^2 = int |hat{f}(theta)|^2 * exp(t*lambda_H(theta)) dmu(theta)
    
    Args:
        f_values: Array of shape (M,) giving f(h_j) on finite set {h_j}
        basis_points: List of M group elements h_j, each as integer coefficients
        levy_measure: LevyMeasure instance
        t: Time parameter
        N: Number of Monte Carlo samples
        seed: Random seed
        
    Returns:
        Estimate of |f|_{H_t}^2
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    h_vectors = np.stack(basis_points, axis=0)  # shape [M, m]
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        
        # Fourier transform: hat{f}(theta) = sum_j f(h_j) * exp(-i <theta, h_j>)
        phases = h_vectors @ theta  # shape [M]
        f_hat = np.dot(f_values, np.exp(-1j * phases))
        
        lam = lambda_symbol(theta, levy_measure)
        acc += (np.abs(f_hat) ** 2) * np.exp(t * lam)
    
    return acc / N
