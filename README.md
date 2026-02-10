# Heat Flow Random Walks for Uniformly Discrete Virtual Persistence Diagrams

This repository provides a Python implementation of translation-invariant heat flow random walks on virtual persistence diagram groups, as described in:

**Paper Title**: *Heat Flow Random Walks for Uniformly Discrete Virtual Persistence Diagrams*

## Mathematical Background

### Virtual Persistence Diagram Groups

Persistent homology associates to a filtered simplicial complex a persistence diagram: a finite multiset of points in $\mathbb{R}^2$ encoding birth and death parameters of homological features. However, finite persistence diagrams form a commutative monoid without additive inverses.

**Virtual persistence diagrams** extend this framework to arbitrary pointed metric spaces. They are constructed by passing to the Grothendieck group $K(X,A)$ of diagrams relative to a metric pair $(X,d,A)$, equipped with the canonical translation-invariant Grothendieck metric $\rho$ extending the Wasserstein-1 distance.

### Key Results

1. **Uniformly Discrete Metric Pairs**: For integer-valued filtrations, $(X/A, d_1, [A])$ is uniformly discrete, where $d_1$ is the 1-strengthened metric.

2. **Canonical Levy Measure**: Geometry-induced Levy measure constructed from VPD metric via graph Laplacian on birth-death points.

3. **Heat Kernel Invariants**: Spectral integration over dual group $\widehat{H}$ to compute:
   - Return probability $p_t(0)$
   - Collision probability $\sum_h p_t(h)^2$
   - Energy derivative $-\frac{d}{dt}p_t(0)$
   - Resolvent diagonal $G_s(0,0)$

4. **Theoretical Bounds**: Lipschitz, ultracontractive, and Sobolev-Green bounds using spectral formulas.

## Installation

### Requirements

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- NetworkX (for graph-based examples)
- gudhi (for persistence computation)
- pyvista (for 3D visualizations)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Pipeline

```bash
python scripts/example/synthetic_pipeline.py
```

This runs the complete geometry-first pipeline:
1. Load synthetic graphs A and B
2. Compute H1 persistence diagrams
3. Build geometry window from all BD points
4. Construct canonical Levy measure
5. Compute invariants via spectral integration
6. Evaluate bounds

### Generate Figures

```bash
python scripts/example/generate_figures.py all
```

## Project Structure

```
.
├── src/                    # Core implementation
│   ├── vpd_group.py        # Geometry window, mass functional M, d_1 metric
│   ├── jump_kernel.py      # LevyMeasure class, canonical construction
│   ├── invariants.py       # Heat kernel invariants (p_t(0), sum p_t(h)^2, etc.)
│   ├── bounds.py           # Theoretical bounds (Lipschitz, ultracontractive, Sobolev)
│   ├── random_walk.py      # Compound Poisson simulation
│   └── pyvista_helpers.py  # 3D graph visualization
│
├── scripts/example/        # Example pipeline scripts
└── results/example/        # Computed results and figures
```

## Core Modules

### `src/vpd_group.py`
- Geometry window construction from BD points
- Mass functional $M(g) = \sum_u |n_u| \cdot d_1(u, [A])$
- 1-strengthened metric $d_1$ computation

### `src/jump_kernel.py`
- `LevyMeasure` class with jumps and rates
- Canonical geometry-induced Levy measure construction
- Levy-Khintchine exponent $\lambda_H(\theta)$

### `src/invariants.py`
- Return probability $p_t(0)$ via Monte Carlo integration
- Collision probability $\sum_h p_t(h)^2$
- Energy derivative $-\frac{d}{dt}p_t(0)$
- Resolvent diagonal $G_s(0,0)$

### `src/bounds.py`
- Lipschitz bounds: $\mathrm{Lip}_\rho(f) \leq |f|_{H_t} \cdot (-\frac{d}{dt}p_t(0))^{1/2}$
- Ultracontractive bounds: $\|P_t\|_{2\to\infty} = \sqrt{\sum_h p_t(h)^2}$
- Sobolev-Green bounds using $G_s(0,0)$

### `src/random_walk.py`
- Compound Poisson simulation (generic over LevyMeasure)
- Translation-invariant heat flow random walks

## References

**Core Theory**:
- Liggett, T. M. (2010). *Continuous Time Markov Processes: An Introduction*. Graduate Studies in Mathematics, 113. American Mathematical Society.
- Fukushima, M., Oshima, Y., & Takeda, M. (1994). *Dirichlet Forms and Symmetric Markov Processes*. De Gruyter Studies in Mathematics, 19. Walter de Gruyter.
- Applebaum, D. (2009). *Lévy Processes and Stochastic Calculus* (2nd ed.). Cambridge Studies in Advanced Mathematics, 116. Cambridge University Press.
- Berg, C., & Forst, G. (1975). *Potential Theory on Locally Compact Abelian Groups*. Ergebnisse der Mathematik und ihrer Grenzgebiete, 87. Springer-Verlag.

**Virtual Persistence Diagrams**:
- Bubenik, P., & Elchesen, A. (2022). Virtual persistence diagrams, signed measures, Wasserstein distances, and Banach spaces. *Journal of Applied and Computational Topology*, 6, 429--474.
- Fanning, C., & Aktas, M. (2025). Reproducing Kernel Hilbert Spaces for Virtual Persistence Diagrams. Preprint: https://arxiv.org/abs/2512.07282

**Persistent Homology**:
- Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2000). Topological persistence and simplification. *Proceedings 41st Annual Symposium on Foundations of Computer Science*, 454--463.
- Zomorodian, A., & Carlsson, G. (2005). Computing Persistent Homology. *Discrete & Computational Geometry*, 33, 249--274.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of Persistence Diagrams. *Discrete & Computational Geometry*, 37(1), 103--120.
- Oudot, S. Y. (2015). *Persistence Theory: From Quiver Representations to Data Analysis*. Mathematical Surveys and Monographs, 209. American Mathematical Society.

**Graph Models**:
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of "small-world" networks. *Nature*, 393(6684), 440--442.
