"""
Lipschitz and Regularity Bounds

Evaluation of theoretical bounds using correct spectral formulas.
All bounds use primitive notation: p_t(0), -d/dt p_t(0), G_s(0,0).
"""

import numpy as np
from typing import Dict, Callable
from src.jump_kernel import LevyMeasure, lambda_symbol


def compute_lipschitz_bound(
    f_Ht_norm: float,
    energy_derivative: float
) -> float:
    """
    Compute Lipschitz bound for heat-kernel features.
    
    Lip_rho(f) <= |f|_{H_t} * (-d/dt p_t(0))^{1/2}
    
    This uses the spectral formula:
    -d/dt p_t(0) = int lambda_H(theta) * exp(-t*lambda_H(theta)) dmu
    
    Args:
        f_Ht_norm: ||f||_{H_t} (RKHS norm)
        energy_derivative: -d/dt p_t(0) (energy/derivative)
        
    Returns:
        Upper bound on Lip_rho(f)
    """
    if energy_derivative < 0:
        return float('inf')
    
    return f_Ht_norm * np.sqrt(energy_derivative)


def compute_ultracontractive_bound(collision_probability: float) -> float:
    """
    Compute ultracontractive bound.
    
    ||P_t||_{2->infty}^2 = sum_h p_t(h)^2
    
    Args:
        collision_probability: sum_h p_t(h)^2
        
    Returns:
        Ultracontractive constant ||P_t||_{2->infty}
    """
    return np.sqrt(collision_probability)


def compute_sobolev_bound(
    f_l2_squared: float,
    dirichlet_energy: float,
    s: float,
    G_s_00: float
) -> float:
    """
    Compute Sobolev-Green bound.
    
    ||f||_infty^2 <= G_s(0,0) * (s * ||f||_2^2 + E_H(f,f))
    
    Args:
        f_l2_squared: ||f||_2^2
        dirichlet_energy: E_H(f,f) (Dirichlet energy)
        s: Resolvent parameter
        G_s_00: G_s(0,0) (resolvent diagonal)
        
    Returns:
        Upper bound on ||f||_infty
    """
    if G_s_00 <= 0:
        return float('inf')
    
    return np.sqrt(G_s_00 * (s * f_l2_squared + dirichlet_energy))


def compute_poincare_constant(
    levy_measure: LevyMeasure,
    mass_fn: Callable[[np.ndarray], float],
    N: int = 1000,
    seed: int = 14
) -> float:
    """
    Approximate Poincare constant Lambda.
    
    Lambda = inf_{theta != 0} lambda_H(theta) / Lip_rho(chi_theta)^2
    
    For general Levy measure, we approximate by sampling theta and
    computing the ratio for test jump directions.
    
    Args:
        levy_measure: LevyMeasure instance
        mass_fn: Function computing M(kappa) for jump vectors
        N: Number of samples
        seed: Random seed
        
    Returns:
        Approximate Lambda
    """
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    cur_min = float('inf')
    
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        
        lam = lambda_symbol(theta, levy_measure)
        
        max_lip_ratio = 0.0
        for jump in levy_measure.jumps:
            if np.array_equal(jump, np.zeros(m, dtype=np.int64)):
                continue
            
            phase_diff = np.dot(jump, theta)
            char_val = np.exp(1j * phase_diff)
            char_diff = abs(char_val - 1.0)
            
            rho_jump = mass_fn(jump)
            if char_diff > 0 and rho_jump > 0:
                lip_ratio = char_diff / rho_jump
                max_lip_ratio = max(max_lip_ratio, lip_ratio)
        
        if max_lip_ratio > 0:
            ratio = lam / (max_lip_ratio ** 2)
            if ratio < cur_min:
                cur_min = ratio
    
    return cur_min if cur_min != float('inf') else 1.0
