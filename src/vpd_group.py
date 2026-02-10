"""
Virtual Persistence Diagram Group Construction

Core theoretical components for working with VPD groups in the uniformly discrete regime.
Includes mass functional M and metric rho.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


def compute_diagonal_distance(bd_point: Tuple[int, int]) -> float:
    """
    Compute distance from BD point to diagonal.
    
    d(x, A) = min_{a in A} d(x, a) where d is L1 metric.
    For point (b, d), distance to diagonal is |b - d| / 2.
    
    Args:
        bd_point: (birth, death) tuple
        
    Returns:
        Distance to diagonal
    """
    b, d = bd_point
    return abs(b - d) / 2.0


def compute_mass(
    coeffs: np.ndarray,
    basis_bd_points: List[Tuple[int, int]],
    diag_distance_fn: Callable[[Tuple[int, int]], float] = None
) -> float:
    """
    Compute mass functional M(g) for group element g.
    
    M(g) = sum_{u} |n_u| * d_1(u, [A])
    
    where g = sum_u n_u * e_u and d_1 is the 1-strengthened metric.
    
    Args:
        coeffs: Integer vector of shape (m,) representing g in basis
        basis_bd_points: List of BD points (u in X/A) forming basis
        diag_distance_fn: Function computing d_1(u, [A]) (default: uses compute_diagonal_distance)
        
    Returns:
        M(g)
    """
    if diag_distance_fn is None:
        diag_distance_fn = compute_diagonal_distance
    
    if len(coeffs) != len(basis_bd_points):
        raise ValueError(f"coeffs length {len(coeffs)} != basis length {len(basis_bd_points)}")
    
    mass = 0.0
    for i, (n_u, bd_point) in enumerate(zip(coeffs, basis_bd_points)):
        d_to_diag = diag_distance_fn(bd_point)
        mass += abs(n_u) * d_to_diag
    
    return mass


def rho_upper_bound(
    coeffs: np.ndarray,
    basis_bd_points: List[Tuple[int, int]],
    diag_distance_fn: Callable[[Tuple[int, int]], float] = None
) -> float:
    """
    Upper bound on rho(g, 0) using mass functional.
    
    By Lemma rho-mass: rho(g, 0) <= M(g)
    
    Args:
        coeffs: Integer vector representing g
        basis_bd_points: List of BD points forming basis
        diag_distance_fn: Function computing d_1(u, [A])
        
    Returns:
        Upper bound on rho(g, 0)
    """
    return compute_mass(coeffs, basis_bd_points, diag_distance_fn)


def build_geometry_window(
    all_bd_points: List[Tuple[int, int]],
    max_birth: Optional[int] = None,
    max_death: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Build geometry window from collection of BD points.
    
    This defines the finite "geometry window" independent of specific diagrams.
    For experiments, this can be all BD points appearing in the dataset,
    or a truncation in the underlying BD lattice.
    
    Args:
        all_bd_points: Collection of BD points from all diagrams
        max_birth: Optional maximum birth time (truncation)
        max_death: Optional maximum death time (truncation)
        
    Returns:
        basis_points: Ordered list of BD points in geometry window
        index_of: Dictionary mapping BD point to index
    """
    unique_points = sorted(set(all_bd_points))
    
    if max_birth is not None or max_death is not None:
        filtered = []
        for b, d in unique_points:
            if max_birth is not None and b > max_birth:
                continue
            if max_death is not None and d > max_death:
                continue
            filtered.append((b, d))
        unique_points = filtered
    
    index_of = {bd: i for i, bd in enumerate(unique_points)}
    return unique_points, index_of


def index_birth_death_pairs(
    alpha_A: List[Tuple[int, int]], 
    alpha_B: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Index all distinct birth-death pairs from two diagrams.
    
    DEPRECATED: Use build_geometry_window instead for geometry-first approach.
    Kept for backward compatibility.
    
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
        if bd in index_of:
            beta[index_of[bd]] += 1
    
    for bd in alpha_B:
        if bd in index_of:
            beta[index_of[bd]] -= 1
    
    return beta


def compute_effective_subgroup(
    alpha_A: List[Tuple[int, int]],
    alpha_B: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int], np.ndarray]:
    """
    Construct the effective subgroup H from two diagrams.
    
    DEPRECATED: Use build_geometry_window + form_virtual_difference instead.
    Kept for backward compatibility.
    
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
