# Heat Flow Random Walk on Virtual Persistence Diagram Groups

## Current Status

Implementing the full MUTAG pipeline for heat flow random walks on virtual persistence diagrams:
- MUTAG dataset: two graphs (mutagenic vs non-mutagenic)
- Lower-star clique filtration with integer vertex labels
- Persistent homology computation (H1)
- Virtual persistence diagram (VPD) group construction
- Translation-invariant random walk on effective subgroup H
- Heat kernel invariants (Z_H, E_H, C_H, R_{s,H})
- Lipschitz/ultracontractive/resolvent bounds evaluation

## Mathematical Framework

### Uniformly Discrete Regime (Heat Flow Random Walks Paper)
- Metric pair (X,d,A) with uniformly discrete (X/A, d_1, [A])
- K(X,A) is discrete locally compact abelian group
- Translation-invariant convolution semigroups (p_t) on K(X,A)
- Effective subgroup H = <supp(p_t)> is countable
- Levy-Khintchine exponent lambda_H: hat(H) -> [0,infty)
- Heat kernel invariants:
  - Z_H(t) = p_t(0) (return profile)
  - E_H(t) = int lambda_H(theta) e^{-t lambda_H(theta)} dmu (energy profile)
  - C_H(t) = sum_h p_t(h)^2 (collision profile)
  - R_{s,H}(0) = int 1/(s+lambda_H(theta)) dmu (resolvent)
- Compound Poisson representation for finite activity
- Spectral Lipschitz bound: Lip_rho(f) <= Lambda^{-1/2} ||f||_{H_t} E_H(t)^{1/2}

### MUTAG Pipeline
- Two graphs G^A (non-mutagenic) and G^B (mutagenic)
- Vertex filtration: phi(v) = ell_V(v) in {0,...,6}
- Lower-star clique filtration: Phi(sigma) = max_{v in sigma} phi(v)
- Birth-death space: X = {(b,d) in Z^2 : b < d} with l1 metric
- Diagrams: alpha^A, alpha^B in D(X,A) (H0 persistence)
- Virtual difference: beta = alpha^A - alpha^B in K(X,A)
- Effective subgroup: H = <supp(alpha^A) cup supp(alpha^B)> cong Z^m
- Jump kernel: j(±e_i) = lambda, j(0) = 0 (nearest-neighbor)
- Exponent: lambda_H(xi) = 2*lambda * sum_{i=1}^m (1 - cos(xi_i))

## Project Structure (Math Mode)

- `src/` - Source code modules
- `scripts/` - Executable scripts and entry points
  - `scripts/figures/` - Figure outputs from scripts
  - `scripts/example/` - Example-specific outputs
- `results/` - Computed results and outputs
  - `results/figures/` - Figure outputs
  - `results/example/` - Example-specific results
- `data/` - Input datasets (OHSU graph dataset will be downloaded here)

## Setup

1. Activate virtual environment:
   ```
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Next Actions

- Download MUTAG dataset to `data/`
- Select two graphs (one mutagenic, one non-mutagenic)
- Implement lower-star clique filtration with vertex labels
- Compute persistent homology (H1) using GUDHI
- Construct VPD group K(X,A) and effective subgroup H
- Implement jump kernel j and compound Poisson simulation
- Compute heat kernel invariants via Monte Carlo
- Evaluate Lipschitz bounds and other regularity estimates

## Known Issues

None yet.
