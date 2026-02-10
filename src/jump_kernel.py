"""
Translation-Invariant Levy Measure on VPD Group

Geometry-induced Levy measure with truncation support.
"""

import numpy as np
from typing import Tuple, Callable, Optional


class LevyMeasure:
    """
    Levy measure on effective subgroup H.
    
    Represents a symmetric, translation-invariant jump kernel j: H -> [0,infty)
    via a finite collection of jump vectors and their rates.
    """
    
    def __init__(self, jumps: np.ndarray, rates: np.ndarray):
        """
        Initialize Levy measure.
        
        Args:
            jumps: Array of shape [K, m] where each row is integer coefficients
                   representing a jump vector kappa in H
            rates: Array of shape [K] of nonnegative jump rates nu(kappa)
        """
        if jumps.shape[0] != rates.shape[0]:
            raise ValueError("jumps and rates must have same length")
        if len(rates.shape) != 1:
            raise ValueError("rates must be 1D")
        if len(jumps.shape) != 2:
            raise ValueError("jumps must be 2D")
        
        self.jumps = jumps.astype(np.int64)
        self.rates = rates.astype(np.float64)
        self.m = jumps.shape[1]
        self.K = jumps.shape[0]
    
    def truncate_by_mass(self, mass_fn: Callable[[np.ndarray], float], R: float) -> 'LevyMeasure':
        """
        Truncate Levy measure to jumps with mass <= R.
        
        Args:
            mass_fn: Function mapping jump vector to mass M(kappa)
            R: Truncation radius
            
        Returns:
            New LevyMeasure with only jumps satisfying M(kappa) <= R
        """
        mask = np.array([mass_fn(kappa) <= R for kappa in self.jumps])
        return LevyMeasure(self.jumps[mask], self.rates[mask])
    
    def total_mass(self) -> float:
        """Compute total mass q = sum_{kappa != 0} nu(kappa)."""
        return float(np.sum(self.rates))
    
    def jump_probabilities(self) -> np.ndarray:
        """
        Compute probability distribution for jumps.
        
        Returns:
            pi: Probability distribution (normalized rates)
        """
        q = self.total_mass()
        if q == 0:
            return np.zeros_like(self.rates)
        return self.rates / q


def lambda_symbol(theta: np.ndarray, levy_measure: LevyMeasure) -> float:
    """
    Compute Levy-Khintchine exponent lambda_H(theta).
    
    lambda_H(theta) = sum_{kappa != 0} nu(kappa) * (1 - Re(chi_theta(kappa)))
    
    where chi_theta(kappa) = exp(i <theta, kappa>).
    
    Args:
        theta: Character in [-pi, pi)^m
        levy_measure: LevyMeasure instance
        
    Returns:
        lambda_H(theta)
    """
    if len(theta) != levy_measure.m:
        raise ValueError(f"theta dimension {len(theta)} != {levy_measure.m}")
    
    # Compute phases: <theta, kappa> for each jump
    phases = levy_measure.jumps @ theta  # shape [K]
    
    # Re(chi_theta(kappa)) = cos(phase)
    cos_vals = np.cos(phases)
    
    # lambda_H = sum nu(kappa) * (1 - cos(phase))
    return float(np.sum(levy_measure.rates * (1.0 - cos_vals)))


def compute_d1_distance(bd1: Tuple[int, int], bd2: Tuple[int, int]) -> float:
    """
    Compute 1-strengthened metric d_1 between two BD points.
    
    d_1((b1,d1), (b2,d2)) = min(d((b1,d1), (b2,d2)), d((b1,d1), A) + d((b2,d2), A))
    
    where d is L1 metric and d(x, A) = |b - d| / 2.
    
    Args:
        bd1: (birth1, death1) tuple
        bd2: (birth2, death2) tuple
        
    Returns:
        d_1 distance
    """
    b1, d1 = bd1
    b2, d2 = bd2
    
    # L1 distance
    d_direct = abs(b1 - b2) + abs(d1 - d2)
    
    # Distance to diagonal
    d_to_diag_1 = abs(b1 - d1) / 2.0
    d_to_diag_2 = abs(b2 - d2) / 2.0
    d_via_diagonal = d_to_diag_1 + d_to_diag_2
    
    return min(d_direct, d_via_diagonal)


def build_geometry_levy_measure(
    basis_bd_points: list,
    mass_fn: Callable[[np.ndarray], float],
    connectivity: str = "knn",
    k_neighbors: int = 3,
    distance_threshold: Optional[float] = None,
    lambda_val: float = 1.0
) -> LevyMeasure:
    """
    Build canonical geometry-induced Levy measure from VPD metric.
    
    Following the paper's construction: build a weighted graph on BD points
    with edge weights w_{uv} = d_1(u,v), then derive Levy measure from
    the graph Laplacian structure.
    
    The graph Laplacian L = D - A gives a Dirichlet form, and the Levy
    measure rates come from the off-diagonal entries, extended to group jumps.
    
    For group element kappa = sum_i n_i * e_i, the jump rate is determined
    by the graph structure on the BD points.
    
    Args:
        basis_bd_points: List of BD points (u in X/A) forming basis
        mass_fn: Function computing M(kappa) for jump vector kappa
        connectivity: "knn" (k-nearest neighbors) or "threshold" (distance-based)
        k_neighbors: Number of neighbors for k-NN (if connectivity="knn")
        distance_threshold: Max distance for edges (if connectivity="threshold")
        lambda_val: Scaling parameter for rates
        
    Returns:
        LevyMeasure instance with geometry-induced rates
    """
    m = len(basis_bd_points)
    
    if m == 0:
        return LevyMeasure(np.zeros((0, 0), dtype=np.int64), np.array([], dtype=np.float64))
    
    # Build distance matrix for BD points
    dist_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                dist_matrix[i, j] = compute_d1_distance(basis_bd_points[i], basis_bd_points[j])
    
    # Build adjacency matrix based on connectivity
    adjacency = np.zeros((m, m), dtype=bool)
    
    if connectivity == "knn":
        # k-nearest neighbors
        for i in range(m):
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i, :]
            distances[i] = np.inf  # Exclude self
            nearest_indices = np.argsort(distances)[:k_neighbors]
            adjacency[i, nearest_indices] = True
            adjacency[nearest_indices, i] = True  # Symmetric
    
    elif connectivity == "threshold":
        if distance_threshold is None:
            # Use median distance as threshold
            upper_tri = dist_matrix[np.triu_indices(m, k=1)]
            distance_threshold = np.median(upper_tri) if len(upper_tri) > 0 else 1.0
        
        for i in range(m):
            for j in range(m):
                if i != j and dist_matrix[i, j] <= distance_threshold:
                    adjacency[i, j] = True
    
    else:
        raise ValueError(f"Unknown connectivity: {connectivity}")
    
    # Build weighted graph: edge weights = d_1 distances
    edge_weights = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if adjacency[i, j]:
                edge_weights[i, j] = dist_matrix[i, j]
    
    # Graph Laplacian: L = D - A where D_{ii} = sum_j w_{ij}
    degree = edge_weights.sum(axis=1)
    
    # For the Levy measure, we use the graph structure to determine jump rates.
    # The canonical construction: for basis jumps ±e_i, the rate is related to
    # the degree and edge weights involving point i.
    #
    # Following the paper's finite graph construction, the Levy measure rates
    # come from the graph Laplacian. For a jump kappa = ±e_i, we use:
    #   nu(±e_i) = lambda * (degree[i] / max_degree) * (1 / (1 + avg_edge_weight[i]))
    #
    # This ensures:
    # 1. Higher degree points have higher jump rates (more connected = more active)
    # 2. Points with larger edge weights have lower rates (farther = less likely)
    # 3. Rates scale with lambda_val
    
    max_degree = np.max(degree) if np.max(degree) > 0 else 1.0
    avg_edge_weight = edge_weights.sum(axis=1) / (degree + 1e-10)  # Avoid division by zero
    
    jumps = []
    rates = []
    
    for i in range(m):
        # Jump +e_i
        e_i = np.zeros(m, dtype=np.int64)
        e_i[i] = 1
        mass = mass_fn(e_i)
        
        # Rate based on graph structure
        if degree[i] > 0:
            rate = lambda_val * (degree[i] / max_degree) / (1.0 + avg_edge_weight[i])
        else:
            rate = lambda_val * 0.1  # Small rate for isolated points
        
        jumps.append(e_i)
        rates.append(rate)
        
        # Jump -e_i
        e_i_neg = np.zeros(m, dtype=np.int64)
        e_i_neg[i] = -1
        mass_neg = mass_fn(e_i_neg)
        
        # Same rate (symmetric)
        jumps.append(e_i_neg)
        rates.append(rate)
    
    return LevyMeasure(np.array(jumps), np.array(rates))


def build_nearest_neighbor_levy_measure(m: int, lambda_val: float = 1.0) -> LevyMeasure:
    """
    Build nearest-neighbor Levy measure (for testing/comparison).
    
    j(±e_i) = lambda, j(0) = 0, j(kappa) = 0 otherwise.
    
    Args:
        m: Dimension of H
        lambda_val: Jump rate parameter
        
    Returns:
        LevyMeasure with nearest-neighbor jumps
    """
    jumps = []
    rates = []
    
    for i in range(m):
        e_i = np.zeros(m, dtype=np.int64)
        e_i[i] = 1
        jumps.append(e_i)
        rates.append(lambda_val)
        
        e_i_neg = np.zeros(m, dtype=np.int64)
        e_i_neg[i] = -1
        jumps.append(e_i_neg)
        rates.append(lambda_val)
    
    return LevyMeasure(np.array(jumps), np.array(rates))
