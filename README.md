# Heat Flow Random Walk on Virtual Persistence Diagram Groups

## Current Status

Full MUTAG pipeline implemented and working:
- MUTAG dataset: two largest graphs (mutagenic vs non-mutagenic)
- Lower-star clique filtration with integer vertex labels
- Persistent homology computation (H0)
- Virtual persistence diagram (VPD) group construction
- Translation-invariant random walk on effective subgroup H
- Heat kernel invariants computed: Z_H(tau), E_H(tau), C_H(tau), R_{s,H}(0)
- Lipschitz and ultracontractive bounds evaluated
- Comprehensive figures generated

Results saved in `results/example/bounds/` and `results/example/figures/`

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

## Usage

1. Run the full pipeline:
   ```
   python scripts/example/mutag_pipeline.py
   ```

2. Generate figures:
   ```
   python scripts/example/generate_figures.py
   ```

## Results

- Invariants: `results/example/bounds/invariants.json`
- Bounds: `results/example/bounds/bounds.json`
- Figures: `results/example/figures/*.png`

## Known Issues

None yet.
