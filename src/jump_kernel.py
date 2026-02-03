"""
Translation-Invariant Jump Kernel on VPD Group

Nearest-neighbor jump kernel: j(±e_i) = lambda, j(0) = 0.
"""

import numpy as np
from typing import Tuple


def build_jump_kernel(m: int, lambda_val: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build nearest-neighbor jump kernel on Z^m.
    
    Args:
        m: Dimension of the group H (number of basis elements)
        lambda_val: Jump rate parameter (default 1.0)
        
    Returns:
        J: Array of shape [2*m, m] containing jump vectors
        rates: Array of shape [2*m] containing jump rates
    """
    J = []
    rates = []
    
    for i in range(m):
        e_i = np.zeros(m, dtype=np.int64)
        e_i[i] = 1
        J.append(e_i)
        rates.append(lambda_val)
        
        e_i_neg = np.zeros(m, dtype=np.int64)
        e_i_neg[i] = -1
        J.append(e_i_neg)
        rates.append(lambda_val)
    
    J = np.array(J, dtype=np.int64)
    rates = np.array(rates, dtype=np.float64)
    
    return J, rates


def compute_total_rate(rates: np.ndarray) -> float:
    """Compute total jump rate q = sum(rates)."""
    return float(np.sum(rates))


def compute_jump_probabilities(rates: np.ndarray) -> np.ndarray:
    """
    Compute probability distribution for jumps.
    
    Args:
        rates: Array of jump rates
        
    Returns:
        pi: Probability distribution (normalized rates)
    """
    q = compute_total_rate(rates)
    return rates / q
