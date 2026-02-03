"""
Virtual Persistence Diagram Group Construction

Core theoretical components for working with VPD groups in the uniformly discrete regime.
"""

import numpy as np
from typing import List, Tuple, Dict


def index_birth_death_pairs(
    alpha_A: List[Tuple[int, int]], 
    alpha_B: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Index all distinct birth-death pairs from two diagrams.
    
    Args:
        alpha_A: List of (birth, death) pairs from first diagram
        alpha_B: List of (birth, death) pairs from second diagram
        
    Returns:
        S_points: List of distinct BD points (basis for H)
        index_of: Dictionary mapping BD point to its index
    """
    all_pairs = set(alpha_A) | set(alpha_B)
    S_points = sorted(list(all_pairs))
    index_of = {bd: i for i, bd in enumerate(S_points)}
    return S_points, index_of


def form_virtual_difference(
    alpha_A: List[Tuple[int, int]],
    alpha_B: List[Tuple[int, int]],
    index_of: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """
    Form the virtual difference beta = alpha^A - alpha^B in K(X,A).
    
    Args:
        alpha_A: List of (birth, death) pairs from first diagram
        alpha_B: List of (birth, death) pairs from second diagram
        index_of: Dictionary mapping BD point to its index
        
    Returns:
        beta: Integer vector in Z^m representing the virtual diagram
    """
    m = len(index_of)
    beta = np.zeros(m, dtype=np.int64)
    
    for bd in alpha_A:
        beta[index_of[bd]] += 1
    
    for bd in alpha_B:
        beta[index_of[bd]] -= 1
    
    return beta


def compute_effective_subgroup(
    alpha_A: List[Tuple[int, int]],
    alpha_B: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int], np.ndarray]:
    """
    Construct the effective subgroup H from two diagrams.
    
    Args:
        alpha_A: List of (birth, death) pairs from first diagram
        alpha_B: List of (birth, death) pairs from second diagram
        
    Returns:
        S_points: Basis points for H
        index_of: Dictionary mapping BD point to index
        beta: Virtual difference vector
    """
    S_points, index_of = index_birth_death_pairs(alpha_A, alpha_B)
    beta = form_virtual_difference(alpha_A, alpha_B, index_of)
    return S_points, index_of, beta
