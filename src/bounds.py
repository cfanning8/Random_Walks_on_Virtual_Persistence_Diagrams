"""
Lipschitz and Regularity Bounds

Evaluation of theoretical bounds from the Heat Flow Random Walks paper.
"""

import numpy as np
from typing import Dict


def compute_poincare_constant(
    m: int,
    lambda_val: float = 1.0,
    N: int = 1000,
    seed: int = 14
) -> float:
    """
    Approximate Poincare constant Lambda.
    
    Lambda = inf_{theta != 0} lambda_H(theta) / Lip_rho(chi_theta)^2
    
    For nearest-neighbor kernel, we approximate by sampling theta and
    computing the ratio for test jump directions.
    
    Args:
        m: Dimension of H
        lambda_val: Jump rate parameter
        N: Number of samples
        seed: Random seed
        
    Returns:
        Approximate Lambda
    """
    from src.jump_kernel import build_jump_kernel
    from src.invariants import lambda_H
    
    rng = np.random.default_rng(seed)
    J, rates = build_jump_kernel(m, lambda_val)
    
    cur_min = float('inf')
    
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        
        lam = lambda_H(theta, m, lambda_val)
        
        max_lip_ratio = 0.0
        for jump in J:
            if np.array_equal(jump, np.zeros(m, dtype=np.int64)):
                continue
            
            phase_diff = np.dot(jump, theta)
            char_val = np.exp(1j * phase_diff)
            char_diff = abs(char_val - 1.0)
            
            rho_jump = 1.0
            if char_diff > 0 and rho_jump > 0:
                lip_ratio = char_diff / rho_jump
                max_lip_ratio = max(max_lip_ratio, lip_ratio)
        
        if max_lip_ratio > 0:
            ratio = lam / (max_lip_ratio ** 2)
            if ratio < cur_min:
                cur_min = ratio
    
    return cur_min if cur_min != float('inf') else 1.0


def compute_lipschitz_bound(
    Z_H: float,
    E_H: float,
    Lambda: float
) -> float:
    """
    Compute Lipschitz bound for heat-kernel features.
    
    Lip_rho(f) <= Lambda^{-1/2} * ||f||_{H_t} * E_H(t)^{1/2}
    
    For f = k_t(·, gamma_0), we have ||f||_{H_t}^2 = Z_H(t).
    
    Args:
        Z_H: Return profile
        E_H: Energy profile
        Lambda: Poincare constant
        
    Returns:
        Upper bound on Lipschitz constant
    """
    if Lambda <= 0:
        return float('inf')
    
    return (Lambda ** (-0.5)) * np.sqrt(Z_H) * np.sqrt(E_H)


def compute_ultracontractive_bound(C_H: float) -> float:
    """
    Compute ultracontractive bound.
    
    ||P_t||_{2->infty}^2 = C_H(t)
    
    Args:
        C_H: Collision profile
        
    Returns:
        Ultracontractive constant
    """
    return np.sqrt(C_H)


def evaluate_all_bounds(
    invariants: Dict,
    m: int,
    lambda_val: float = 1.0
) -> Dict:
    """
    Evaluate all theoretical bounds.
    
    Args:
        invariants: Dictionary with Z_H, C_H, E_H, R_s_H
        m: Dimension of H
        lambda_val: Jump rate parameter
        
    Returns:
        Dictionary of computed bounds
    """
    Lambda = compute_poincare_constant(m, lambda_val)
    
    bounds = {
        "Lambda": float(Lambda),
        "Lipschitz_bounds": [],
        "Ultracontractive_bounds": []
    }
    
    for i, tau in enumerate(invariants["tau_values"]):
        Z_H = invariants["Z_H"][i]
        E_H = invariants["E_H"][i]
        C_H = invariants["C_H"][i]
        
        lip_bound = compute_lipschitz_bound(Z_H, E_H, Lambda)
        ultra_bound = compute_ultracontractive_bound(C_H)
        
        bounds["Lipschitz_bounds"].append({
            "tau": float(tau),
            "bound": float(lip_bound)
        })
        bounds["Ultracontractive_bounds"].append({
            "tau": float(tau),
            "bound": float(ultra_bound)
        })
    
    bounds["Resolvent_bound"] = {
        "s": float(lambda_val),
        "R_s_H": float(invariants["R_s_H"])
    }
    
    return bounds
