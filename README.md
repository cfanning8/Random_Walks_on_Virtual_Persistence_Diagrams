# Heat Flow Random Walk on Virtual Persistence Diagram Groups

## Overview

This project implements a complete computational pipeline for analyzing graph data through **Virtual Persistence Diagrams (VPDs)** and **translation-invariant heat flow random walks**. The implementation rigorously follows the mathematical framework from "Heat Flow Random Walks for Uniformly Discrete Virtual Persistence Diagrams" and provides a concrete example using **synthetic small-world graphs**.

**Architecture**: Geometry-first approach - build geometry window and canonical Levy measure from VPD metric, compute invariants from Levy measure (geometry-only), then apply to specific diagrams.

**Notation**: We use **only primitive notation** from the mathematics: `p_t(0)`, `sum_h p_t(h)^2`, `-d/dt p_t(0)`, `G_s(0,0)`. No shorthand invariants.

---

## Mathematical Framework

### Metric Pair and Uniformly Discrete Spaces

**Metric Pair** `(X, d, A)`:
- `X = {(b,d) in Z^2 : b < d}` (birth-death space, off-diagonal points)
- `d((b1,d1), (b2,d2)) = |b1-b2| + |d1-d2|` (L1 metric)
- `A = {(b,b) : b in Z}` (diagonal subset)

**1-Strengthened Metric**:
```
d_1(x,y) = min(d(x,y), d(x,A) + d(y,A))
```

This induces quotient metric `d_1` on `X/A`. Under Hypothesis (H), `(X/A, d_1, [A])` is uniformly discrete. In our implementation, integer-valued filtrations guarantee this.

### Virtual Persistence Diagram Groups

**Grothendieck Group**: `K(X,A)` is the group completion of the diagram monoid `D(X,A)`:
```
K(X,A) = {alpha - beta : alpha, beta in D(X,A)} / ~
```

In the uniformly discrete case: `K(X,A) cong direct_sum_{x in X\A} Z e_x cong Z^m`.

### Mass Functional

For group element `g = sum_u n_u * e_u`:
```
M(g) = sum_u |n_u| * d_1(u, [A])
```

By Lemma rho-mass: `rho(g, 0) <= M(g)`. Used for metric truncation.

### Canonical Geometry-Induced Levy Measure

Following the paper's construction (Section 4, Theorem 4.1):

1. **Graph Construction**: Build weighted graph `H_F` on BD points `X_F/A` with:
   - Edge weights: `w_{uv} = d_1(u,v)` for edges `{u,v} in E`
   - Connectivity: k-nearest neighbors based on `d_1` distances

2. **Graph Laplacian**: `L = D - A` where:
   - `D_{uu} = sum_v w_{uv}` (degree matrix)
   - `A_{uv} = w_{uv}` (weighted adjacency)

3. **Levy Measure Rates**: For basis jump `kappa = ±e_i`:
   ```
   nu(±e_i) = lambda * (degree[i] / max_degree) / (1 + avg_edge_weight[i])
   ```
   
   This ensures:
   - Higher degree points (more connected) have higher rates
   - Points with larger edge weights (farther) have lower rates
   - Rates scale with parameter `lambda`

4. **Levy-Khintchine Exponent**:
   ```
   lambda_H(theta) = sum_{kappa != 0} nu(kappa) * (1 - Re(chi_theta(kappa)))
   ```
   
   where `chi_theta(kappa) = exp(i <theta, kappa>)` and `theta in [-pi, pi)^m`.

5. **Metric Truncation**: For truncation radius `R > 0`:
   ```
   nu_R(kappa) = nu(kappa) * 1_{rho(kappa,0) <= R}
   ```

### Heat Kernel Invariants (Primitive Notation)

**Return Probability**:
```
p_t(0) = int_{hat(H)} exp(-t*lambda_H(theta)) dmu(theta)
```

**Collision Probability**:
```
sum_{h in H} p_t(h)^2 = int_{hat(H)} exp(-2*t*lambda_H(theta)) dmu(theta)
```

This equals `P(X_t = X_t')` for two independent copies.

**Energy / Derivative**:
```
-d/dt p_t(0) = int_{hat(H)} lambda_H(theta) * exp(-t*lambda_H(theta)) dmu(theta)
```

**Resolvent Diagonal**:
```
G_s(0,0) = int_{hat(H)} 1/(s + lambda_H(theta)) dmu(theta)
```

**Normalized Spectral Energy Scale** (ratio, not named):
```
(-d/dt p_t(0)) / p_t(0) = int lambda_H(theta) * exp(-t*lambda_H(theta)) dmu / int exp(-t*lambda_H(theta)) dmu
```

### Theoretical Bounds

**Lipschitz Bound**:
```
Lip_rho(f) <= |f|_{H_t} * (-d/dt p_t(0))^{1/2}
```

For `f = k_t(·,0)`, we have `|f|_{H_t}^2 = p_t(0)`.

**Ultracontractivity**:
```
||P_t||_{2->infty}^2 = sum_h p_t(h)^2
```

**Sobolev-Green Inequality**:
```
||f||_infty^2 <= G_s(0,0) * (s*||f||_2^2 + E_H(f,f))
```

For kernel section `f = k_t(·,0)`:
- `||f||_2^2 = int exp(-2*t*lambda_H(theta)) dmu`
- `E_H(f,f) = int lambda_H(theta)*exp(-2*t*lambda_H(theta)) dmu`

---

## Implementation

### Architecture: Geometry-First Pipeline

1. **Build Geometry Window**: Collect all BD points from dataset, define finite geometry window `basis_bd_points` independent of specific diagrams
2. **Construct Canonical Levy Measure**: Build weighted graph on BD points with edge weights `w_{uv} = d_1(u,v)`, derive Levy measure rates from graph Laplacian structure
3. **Compute Invariants**: Evaluate `p_t(0)`, `sum_h p_t(h)^2`, `-d/dt p_t(0)`, `G_s(0,0)` via spectral integration over `hat(H)`
4. **Apply to Diagrams**: Form virtual difference `beta = alpha^A - alpha^B` and evaluate functionals

### Data Structures

**LevyMeasure Class**:
- `jumps`: Array `[K, m]` of jump vectors `kappa in H`
- `rates`: Array `[K]` of jump rates `nu(kappa)`
- `truncate_by_mass(mass_fn, R)`: Truncate to `M(kappa) <= R`

**Geometry Window**:
- `basis_bd_points`: List of BD points `(b,d)` in geometry window
- `index_of`: Dictionary mapping BD point to basis index
- `mass_fn(kappa)`: Computes `M(kappa) = sum |n_u| * d_1(u, [A])`

### Algorithms

**Canonical Levy Measure Construction**:
```python
def build_geometry_levy_measure(basis_bd_points, mass_fn, connectivity="knn", k_neighbors=3, lambda_val=1.0):
    """
    1. Build distance matrix: d_1(u,v) for all BD points
    2. Build k-NN graph: connect each point to k nearest neighbors
    3. Edge weights: w_{uv} = d_1(u,v)
    4. Graph Laplacian: L = D - A
    5. Levy rates: nu(±e_i) = lambda * (degree[i]/max_degree) / (1 + avg_edge_weight[i])
    """
```

**Spectral Integration** (generic over LevyMeasure):
```python
def lambda_symbol(theta, levy_measure):
    """lambda_H(theta) = sum nu(kappa) * (1 - cos(<theta, kappa>))"""
    phases = levy_measure.jumps @ theta
    return sum(levy_measure.rates * (1.0 - cos(phases)))

def estimate_return_probability(t, levy_measure, N, seed):
    """p_t(0) = int exp(-t*lambda_H(theta)) dmu via Monte Carlo"""
    # Sample theta ~ Uniform([-pi, pi)^m)
    # Evaluate lambda_H(theta) using lambda_symbol
    # Average exp(-t*lambda_H(theta))
```

**Time Complexity**: `O(N * K * m)` where `K` is number of jumps, `N` is Monte Carlo samples.

### Synthetic Small-World Graphs

**Dataset**: Watts-Strogatz small-world model
- Graph A: `n=50`, `k=6`, `p=0.3`, edge weights `[1, 8]`
- Graph B: `n=60`, `k=8`, `p=0.4`, edge weights `[1, 10]`

**Edge-Based Filtration for H1**:
- Vertices: born at `min_edge_weight - 1`
- Edges: born at their weight
- Triangles: born at maximum edge weight

This guarantees integer-valued birth-death times, ensuring uniformly discrete metric pair.

---

## Project Structure

```
.
├── src/                    # Source code modules
│   ├── vpd_group.py        # Geometry window, mass functional M, d_1 metric
│   ├── jump_kernel.py      # LevyMeasure class, canonical construction, lambda_symbol
│   ├── invariants.py       # Primitive invariants (p_t(0), sum p_t(h)^2, -d/dt p_t(0), G_s(0,0))
│   ├── bounds.py           # Bounds using correct spectral formulas
│   ├── random_walk.py      # Compound Poisson simulation (generic over LevyMeasure)
│   └── pyvista_helpers.py  # 3D graph visualization (300 DPI)
│
├── scripts/example/        # Pipeline scripts
│   ├── generate_synthetic_graphs.py
│   ├── synthetic_pipeline.py  # Geometry-first pipeline
│   └── generate_figures.py    # Figure generation (300 DPI, primitive notation)
│
├── results/example/
│   ├── bounds/             # JSON files with primitive notation
│   │   ├── invariants.json  # return_probability, collision_probability, energy_derivative, resolvent_diagonal
│   │   └── bounds.json
│   ├── figures/            # All figures at 300 DPI
│   └── pipeline_data.pkl
│
└── data/synthetic/         # Synthetic graph data
```

---

## Setup

### Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Key Dependencies**: `numpy`, `networkx`, `gudhi`, `matplotlib`, `pyvista`, `scipy`

### Generate Synthetic Graphs

```bash
python scripts/example/generate_synthetic_graphs.py
```

---

## Usage

### Full Pipeline

```bash
python scripts/example/synthetic_pipeline.py
```

**Pipeline Steps**:
1. Load synthetic graphs A and B
2. Compute H1 persistence diagrams `alpha_A`, `alpha_B`
3. **Build geometry window** from all BD points (independent of specific diagrams)
4. **Construct canonical Levy measure** from VPD metric (graph Laplacian on BD points)
5. **Compute invariants** from Levy measure via spectral integration:
   - `p_t(0)` (return probability)
   - `sum_h p_t(h)^2` (collision probability)
   - `-d/dt p_t(0)` (energy derivative)
   - `G_s(0,0)` (resolvent diagonal)
6. Form virtual difference `beta = alpha^A - alpha^B`
7. Evaluate bounds using correct spectral formulas
8. Save results with primitive notation

### Generate Figures

```bash
python scripts/example/generate_figures.py all
```

**Figure Types**:
- `graphs`: 3D PyVista visualizations (300 DPI)
- `diagrams`: Persistence diagrams (300 DPI)
- `virtual`: Virtual diagram `beta` (300 DPI)
- `invariants`: Invariant profiles using primitive notation (300 DPI)
- `bounds`: Bounds visualization using correct spectral formulas (300 DPI)

---

## Results

### Numerical Results

**Time Range**: `tau in [0.0, 1.0]` with 15 evenly spaced points (linear spacing)

**Invariants** (`results/example/bounds/invariants.json`):
- `return_probability`: `p_t(0)` values (bootstrap samples for each `tau`)
  - Range: `[0.0011, 1.0]` (decreasing from 1.0 at tau=0)
  - Bootstrap: 10 independent estimates per time point
  - Monte Carlo samples: `N = 100,000` per estimate
- `collision_probability`: `sum_h p_t(h)^2` values
  - Range: `[8.69e-06, 1.0]` (decreasing from 1.0 at tau=0)
  - Bootstrap: 10 independent estimates per time point
  - Monte Carlo samples: `N = 100,000` per estimate
- `energy_derivative`: `-d/dt p_t(0)` values
  - Bootstrap: 10 independent estimates per time point
  - Monte Carlo samples: `N = 50,000` per estimate
- `resolvent_diagonal`: `G_s(0,0)` value
  - Fixed parameter: `s = lambda = 1.0`
  - Monte Carlo samples: `N = 50,000`

**Bounds** (`results/example/bounds/bounds.json`):
- `lipschitz_bounds`: Using `|f|_{H_t} * (-d/dt p_t(0))^{1/2}`
  - Range: `[0.0026, 2.83]` (decreasing over time)
  - Computed from bootstrap samples of return_prob and energy_derivative
  - Formula: For each bootstrap sample `j`, `bound_j = sqrt(rp_samples[j]) * sqrt(ed_samples[j])`
  - Final bound: Mean of bootstrap samples, CI from 2.5th and 97.5th percentiles
- `ultracontractive_bounds`: Using `sqrt(sum_h p_t(h)^2)`
  - Range: `[0.0031, 1.0]` (decreasing from 1.0 at tau=0)
  - Computed as `sqrt(collision_probability)` for each time point
  - CI: `sqrt(percentile(collision_prob_samples, 2.5))` to `sqrt(percentile(collision_prob_samples, 97.5))`
- `sobolev_bounds`: Using `G_s(0,0) * (s*||f||_2^2 + E_H(f,f))`
  - Range: `[0.0024, 1.02]` (very close to ultracontractive bounds)
  - Computed from kernel section L2 norm and Dirichlet energy
  - Formula: `sqrt(G_s(0,0) * (s * f_l2_sq + dirichlet_energy))`
  - Uses `estimate_kernel_section_l2_squared` and `estimate_kernel_section_energy`
- `Lambda`: Poincare constant = 0.157579 (computed from Levy measure structure)

**Geometry Window**:
- Basis size: `m = 15` BD points
- Levy measure: 30 jumps (15 basis elements × 2 directions), total mass = 8.0
- Connectivity: k-NN with `k = 3` neighbors

**Virtual Difference**:
- Graph A: 7 H1 persistence pairs
- Graph B: 19 H1 persistence pairs
- Virtual difference: 13 nonzero entries

### Figures

All figures saved in `results/example/figures/` at **300 DPI** with publication-quality styling.

#### 1. Graph Visualizations (graph_A.png, graph_B.png)

**Technology**: PyVista 3D rendering (off-screen mode)

**Specifications**:
- **Resolution**: 300 DPI, 2000×2000 pixels
- **Layout**: 3D positions computed from 2D Kamada-Kawai layout with z-variation
- **Nodes**: 
  - Color: `#333333` (dark gray)
  - Size: 0.1 radius
  - Opacity: 0.85
  - Smooth shading enabled
- **Edges**:
  - Color: `#333333` (dark gray, uniform - no red highlighting)
  - Radius: 0.015
  - Opacity: 0.85
  - **No numerical labels** displayed
- **Background**: Transparent
- **Lighting**: Two directional lights (white, intensity 1.5 and 0.8)
- **Camera**: Position (5, 5, 5), focal point (0, 0, 0), up vector (0, 0, 1)
- **Random seed**: 14 (for layout reproducibility)

#### 2. Persistence Diagrams (persistence_diagram_A.png, persistence_diagram_B.png)

**Specifications**:
- **Resolution**: 300 DPI
- **Size**: 8×8 inches
- **Background**: `#f5f5f5` (light gray)
- **Gridlines**: 
  - Only visible **above the diagonal** (integer lattice effect)
  - Color: `lightgray`, dashed style, linewidth 0.5, alpha 0.5
  - Hidden below diagonal using `Polygon` patch
- **Points**:
  - Color: `#2ca02c` (green)
  - Size: 80
  - Edge color: black, linewidth 1.5
  - Multiplicity labels: White text, bold, fontsize 9 (if mult > 1)
- **Diagonal**: Red dashed line (`r--`), linewidth 1.5, alpha 0.7
- **Axis limits**: **Consistent across all three diagrams** (A, B, virtual)
  - Computed from global min/max of all BD points: `[global_min - 0.5, global_max + 0.5]`
- **Ticks**: Integer ticks from `max(0, int(min_val))` to `int(max_val)`
  - If range ≤ 50: all integers shown
  - If range > 50: step size = `max(1, (max - min) // 20)`
- **Aspect ratio**: Equal (`adjustable='box'`)
- **Labels**: "Birth Time" (x-axis), "Death Time" (y-axis), fontsize 14

#### 3. Virtual Persistence Diagram (virtual_diagram.png)

**Specifications**:
- **Same visual style** as persistence diagrams (consistent axis limits, gridlines, etc.)
- **Points**:
  - **Positive coefficients**: Green circles (`#2ca02c`), size 80 - **matches persistence diagram color**
  - **Negative coefficients**: Red squares (`#d62728`), size 80
  - Edge color: black, linewidth 1.5
  - Multiplicity labels: White text, bold, fontsize 9 (if |coeff| > 1)
- **All other specifications**: Same as persistence diagrams

#### 4. Invariant Profiles (invariant_profiles.png)

**Specifications**:
- **Resolution**: 300 DPI
- **Size**: 8×6 inches
- **Background**: White
- **X-axis**: `tau in [0.0, 1.0]` (linear scale)
- **Y-axis**: `[0.0, 10.0]` (linear scale)
- **Scale**: **Linear** (not logarithmic)
- **Color Scheme** (4 maximally contrastive colors for white background):
  - **Deep Blue** `#0066CC`: `p_t(0)` (return probability)
  - **Deep Orange** `#CC6600`: `sum_h p_t(h)^2` (collision probability)
  - **Deep Green** `#009900`: `-d/dt p_t(0)` (energy derivative)
  - **Deep Purple** `#9900CC`: `(-d/dt p_t(0))/p_t(0)` (normalized energy scale)
- **Line styles**: All solid (`-`)
- **Markers**:
  - `p_t(0)`: Circles (`o`), every 10th point
  - `sum_h p_t(h)^2`: Squares (`s`), every 10th point
  - `-d/dt p_t(0)`: Triangles (`^`), every 10th point
  - Normalized: Diamonds (`d`), every 10th point
- **Line width**: 2.5
- **Confidence intervals**: 
  - Bootstrap 95% CI (2.5th and 97.5th percentiles)
  - `fill_between` with alpha 0.3
  - 10 bootstrap samples per time point
- **Labels**: 
  - X-axis: `$\tau$` (LaTeX, bold, fontsize 14)
  - Y-axis: "Invariant Value" (fontsize 14, bold)
- **Legend**: Best location, frame on, fancybox, shadow, 2 columns

#### 5. Bounds Over Time (bounds_over_time.png)

**Specifications**:
- **Resolution**: 300 DPI
- **Size**: 8×6 inches
- **Background**: White
- **X-axis**: `tau in [0.0, 1.0]` (linear scale)
- **Y-axis**: `[0.0, 10.0]` (linear scale)
- **Scale**: **Linear** (not logarithmic)
- **Color Scheme** (matched to invariant profiles):
  - **Deep Blue** `#0066CC`: `Lip_rho(f)` (Lipschitz bound) - matches `p_t(0)` color
  - **Deep Orange** `#CC6600`: `||P_t||_{2->infty}` (ultracontractive bound) - matches `sum_h p_t(h)^2` color
  - **Deep Green** `#009900`: `||f||_infty` (Sobolev bound, **dashed line**) - matches `-d/dt p_t(0)` color
  - **Deep Purple** `#9900CC`: `(-d/dt p_t(0))/p_t(0)` (normalized energy scale) - same in both plots
- **Line styles**:
  - Lipschitz: Solid (`-`)
  - Ultracontractive: Solid (`-`)
  - Sobolev: **Dashed** (`--`) for visual distinction
  - Normalized: Solid (`-`)
- **Markers**:
  - Lipschitz: Circles (`o`), every 10th point
  - Ultracontractive: Squares (`s`), every 10th point
  - Sobolev: Triangles (`^`), every 10th point
  - Normalized: Diamonds (`d`), every 10th point
- **Line width**: 2.5
- **Confidence intervals**: 
  - Bootstrap 95% CI (2.5th and 97.5th percentiles)
  - `fill_between` with alpha 0.3
- **Labels**: 
  - X-axis: `$\tau$` (LaTeX, bold, fontsize 14)
  - Y-axis: "Value" (fontsize 14, bold)
- **Legend**: Best location, frame on, fancybox, shadow, 2 columns

**Color Matching Logic**:
- Colors match between invariant profiles and bounds plots based on mathematical relationship:
  - `Lip_rho(f)` uses `p_t(0)` → same blue color
  - `||P_t||_{2->infty}` uses `sum_h p_t(h)^2` → same orange color
  - `||f||_infty` uses energy-related quantities → same green color
  - Normalized energy scale appears in both → same purple color

---

## Implementation Details

### Canonical Levy Measure Construction

The Levy measure is constructed from the VPD metric via:

1. **Distance Matrix**: Compute `d_1(u,v)` for all BD points `u,v in basis_bd_points`
2. **k-NN Graph**: Connect each BD point to its `k` nearest neighbors (default `k=3`)
3. **Edge Weights**: `w_{uv} = d_1(u,v)` for edges in graph
4. **Graph Laplacian**: `L = D - A` where `D` is degree matrix, `A` is weighted adjacency
5. **Levy Rates**: For basis jump `kappa = ±e_i`:
   ```
   nu(±e_i) = lambda * (degree[i] / max_degree) / (1 + avg_edge_weight[i])
   ```

This construction ensures the Levy measure reflects the VPD geometry: well-connected BD points (high degree) have higher jump rates, while isolated points have lower rates.

### Spectral Integration

All invariants computed via Monte Carlo integration over `hat(H) cong [-pi, pi)^m`:

- **Sample**: `theta ~ Uniform([-pi, pi)^m)`
- **Evaluate**: `lambda_H(theta)` using `lambda_symbol(theta, levy_measure)`
- **Integrand**: e.g., `exp(-t*lambda_H(theta))` for `p_t(0)`
- **Average**: Over `N` samples

**Sample Sizes**:
- Return and collision probability: `N = 100,000`
- Energy derivative and resolvent: `N = 50,000`

**Bootstrap Confidence Intervals**: `n_bootstrap = 10` independent estimates per time point, compute 2.5th and 97.5th percentiles.

### Notation Alignment

**Code and JSON use primitive notation**:
- `return_probability` -> `p_t(0)`
- `collision_probability` -> `sum_h p_t(h)^2`
- `energy_derivative` -> `-d/dt p_t(0)`
- `resolvent_diagonal` -> `G_s(0,0)`

**No shorthand invariants** like `Z_H`, `C_H`, `E_H`, `m_H`, `R_{s,H}` appear anywhere.

### Random Seed Policy

**All random operations use `seed=14`** (per `.cursorrules`).

### Figure Generation Implementation

**Script**: `scripts/example/generate_figures.py`

**Modular Design**: Each figure type can be generated independently:
```bash
python scripts/example/generate_figures.py graphs      # 3D graph visualizations
python scripts/example/generate_figures.py diagrams    # Persistence diagrams
python scripts/example/generate_figures.py virtual     # Virtual diagram
python scripts/example/generate_figures.py invariants  # Invariant profiles
python scripts/example/generate_figures.py bounds      # Bounds plot
python scripts/example/generate_figures.py all         # All figures
```

**Publication Style Configuration**:
- Font family: Serif (Times New Roman, Times, DejaVu Serif)
- Font sizes: 12 (base), 14 (axes labels), 11 (legend)
- Figure DPI: 300 (hardcoded in `FIGURE_DPI = 300`)
- Background: White (`figure.facecolor = 'white'`)
- Grid: Enabled with alpha 0.3, linewidth 0.5
- Spines: Top and right removed

**Data Loading**:
- Pipeline data loaded from `results/example/pipeline_data.pkl`
- Invariants loaded from `results/example/bounds/invariants.json`
- Bounds loaded from `results/example/bounds/bounds.json`
- All data uses primitive notation (no shorthand)

**3D Graph Visualization**:
- Uses `src/pyvista_helpers.py` for PyVista rendering
- Layout: 2D Kamada-Kawai layout with z-variation (seed=14)
- No red edge highlighting (all edges uniform gray)
- No numerical edge labels displayed
- Off-screen rendering for headless environments

**Persistence Diagram Gridlines**:
- Implementation: `Polygon` patch covering triangle below diagonal
- Polygon vertices: `(min_val-0.5, min_val-0.5)`, `(max_val+0.5, min_val-0.5)`, `(max_val+0.5, max_val+0.5)`
- Polygon color: `#f5f5f5` (matches background), zorder=2
- Gridlines drawn at zorder=0, so hidden below diagonal
- Effect: Integer lattice visible only above diagonal, showing discrete space structure

**Consistent Axis Limits**:
- All three diagrams (A, B, virtual) use same axis limits
- Computed from global min/max: `global_min = min(all_births + all_deaths)`, `global_max = max(all_births + all_deaths)`
- Limits: `[global_min - 0.5, global_max + 0.5]` for both axes
- Ensures visual consistency and proper comparison

**Color Scheme Selection**:
- 4 colors chosen for maximum contrast against white background
- Colors well-separated in RGB space: Blue, Orange, Green, Purple
- Deep, saturated colors for visibility: `#0066CC`, `#CC6600`, `#009900`, `#9900CC`
- Colors matched between plots based on mathematical relationships
- All colors defined as constants at function start for consistency

**Confidence Intervals**:
- Bootstrap method: 10 independent estimates per time point
- Percentiles: 2.5th and 97.5th (95% CI)
- Visualization: `fill_between` with alpha 0.3, no linewidth
- Computed from bootstrap samples stored in JSON files

**Linear vs Logarithmic Scales**:
- **Invariant profiles**: Linear scale (y-axis 0-10)
- **Bounds plot**: Linear scale (y-axis 0-10)
- Both use `ax.plot()` not `ax.semilogy()`
- X-axis: Linear from 0.0 to 1.0 for both plots

---

## Mathematical Rigor

This implementation rigorously follows the paper's construction:

1. **Geometry-First**: Geometry window and Levy measure constructed independently of specific diagrams
2. **Canonical Levy Measure**: Built from VPD metric via graph Laplacian on BD points
3. **Spectral Integration**: All invariants computed via Levy-Khintchine exponent `lambda_H(theta)`
4. **Primitive Notation**: Only mathematical primitives used, no shorthand
5. **Correct Bounds**: All bounds use exact spectral formulas from theorems
6. **Metric Truncation**: Levy measure supports truncation by mass functional `M(kappa)`

The code structure directly reflects the mathematical structure: geometry → Levy measure → invariants → bounds.

---

## Complete Implementation Specifications

### Monte Carlo Integration Details

**Spectral Integration Over Dual Group**:
- Dual group: `hat(H) cong [-pi, pi)^m` (m-torus)
- Sampling: `theta ~ Uniform([-pi, pi)^m)` using `np.random.uniform(-np.pi, np.pi, size=m)`
- Random seed: 14 (consistent across all computations)
- Evaluation: `lambda_H(theta) = sum_{kappa} nu(kappa) * (1 - cos(<theta, kappa>))`
- Integration: Average integrand over N samples

**Sample Sizes**:
- Return probability `p_t(0)`: `N = 100,000` samples
- Collision probability `sum_h p_t(h)^2`: `N = 100,000` samples
- Energy derivative `-d/dt p_t(0)`: `N = 50,000` samples
- Resolvent diagonal `G_s(0,0)`: `N = 50,000` samples
- Kernel section L2 norm: `N = 50,000` samples
- Kernel section Dirichlet energy: `N = 50,000` samples

**Bootstrap Procedure**:
- For each time point `tau`, compute `n_bootstrap = 10` independent estimates
- Each estimate uses different random seed: `seed = 14 + j` for `j in [0, n_bootstrap)`
- Confidence intervals: 2.5th and 97.5th percentiles of bootstrap samples
- All bootstrap samples saved in JSON for reproducibility

### Levy Measure Construction Details

**Graph Construction**:
- Distance matrix: `dist_matrix[i,j] = d_1(basis_bd_points[i], basis_bd_points[j])`
- Connectivity: k-nearest neighbors (`k = 3` by default)
- For each point `i`, find `k` nearest neighbors (excluding self)
- Symmetric adjacency: if `i` is neighbor of `j`, then `j` is neighbor of `i`

**Edge Weights**:
- `edge_weights[i,j] = dist_matrix[i,j]` if `adjacency[i,j] == True`, else 0
- Degree: `degree[i] = sum_j edge_weights[i,j]`
- Average edge weight: `avg_edge_weight[i] = degree[i] / (degree[i] + 1e-10)` (avoid division by zero)

**Levy Rates**:
- For basis jump `+e_i`: `rate = lambda_val * (degree[i] / max_degree) / (1.0 + avg_edge_weight[i])`
- For basis jump `-e_i`: Same rate (symmetric)
- Isolated points (degree=0): `rate = lambda_val * 0.1` (small default rate)
- Total mass: `sum_{kappa} nu(kappa) = 2 * sum_i rate_i`

**Implementation**:
- `LevyMeasure` class: `jumps` array `[K, m]`, `rates` array `[K]`
- `K = 2 * m` (each basis element has ± directions)
- `truncate_by_mass(mass_fn, R)`: Returns new `LevyMeasure` with `M(kappa) <= R`

### Figure Generation Code Patterns

**Color Constants** (defined at function start):
```python
color_return = '#0066CC'      # Deep blue
color_collision = '#CC6600'   # Deep orange
color_energy = '#009900'      # Deep green
color_normalized = '#9900CC'  # Deep purple
```

**Confidence Interval Visualization**:
```python
ax.fill_between(tau_values, lower, upper, alpha=0.3, color=color, linewidth=0)
ax.plot(tau_values, mean_values, '-', color=color, linewidth=2.5, ...)
```

**Persistence Diagram Gridline Hiding**:
```python
triangle = mpatches.Polygon([(min_val-0.5, min_val-0.5), 
                            (max_val+0.5, min_val-0.5),
                            (max_val+0.5, max_val+0.5)],
                           facecolor='#f5f5f5', edgecolor='none', zorder=2)
ax.add_patch(triangle)
# Gridlines drawn at zorder=0, so hidden below diagonal
```

**3D Graph Layout**:
```python
pos_2d = nx.kamada_kawai_layout(G, weight=None, dim=2)  # 2D layout
pos_3d = {node: (x, y, 0.1*(i%10)-0.5) for i, node in enumerate(G.nodes())}
```

### File Formats and Data Structures

**JSON Structure** (`invariants.json`):
```json
{
  "tau_values": [0.0, 0.0714, ..., 1.0],
  "return_probability": [[sample1, sample2, ...], ...],  // 15 lists of 10 samples
  "collision_probability": [[sample1, sample2, ...], ...],
  "energy_derivative": [[sample1, sample2, ...], ...],
  "resolvent_diagonal": float,
  "resolvent_diagonal_samples": [sample1, sample2, ...]
}
```

**JSON Structure** (`bounds.json`):
```json
{
  "Lambda": float,
  "lipschitz_bounds": [bound1, bound2, ...],  // 15 values
  "ultracontractive_bounds": [bound1, bound2, ...],
  "sobolev_bounds": [bound1, bound2, ...]
}
```

**Pickle Structure** (`pipeline_data.pkl`):
- `G_A`, `G_B`: NetworkX graph objects
- `alpha_A`, `alpha_B`: Lists of (birth, death) tuples
- `beta`: NumPy array of coefficients
- `S_points`: List of BD points (basis)
- `geometry_basis_points`: List of BD points in geometry window
- `canonical_levy_measure_jumps`: List of jump vectors
- `canonical_levy_measure_rates`: List of jump rates

### Reproducibility

**Random Seed Policy**:
- All random operations use `seed=14` (per `.cursorrules`)
- Monte Carlo sampling: `rng = np.random.default_rng(14)`
- Bootstrap: `seed=14+j` for j-th bootstrap sample
- Graph layout: `np.random.seed(14)` before layout computation
- NetworkX layouts: Seed set before `kamada_kawai_layout` and `spring_layout`

**Deterministic Computations**:
- All numerical operations use NumPy with fixed precision
- JSON files preserve exact numerical values (float conversion)
- Pickle files preserve exact graph structures and arrays
- Figure generation is deterministic given same input data

### Performance Characteristics

**Time Complexity**:
- Levy measure construction: `O(m^2)` for distance matrix, `O(m * k)` for k-NN graph
- Spectral integration: `O(N * K * m)` where `K` is number of jumps, `N` is MC samples
- Bootstrap: `O(n_bootstrap * N * K * m)` per time point
- Total pipeline: `O(n_tau * n_bootstrap * N * K * m)` where `n_tau = 15`

**Memory Usage**:
- Distance matrix: `O(m^2)` floats
- Levy measure: `O(K * m)` integers (jumps) + `O(K)` floats (rates)
- Bootstrap samples: `O(n_tau * n_bootstrap)` floats per invariant
- Total: Approximately 10-50 MB for typical `m=15` case

**Actual Runtime**:
- Pipeline execution: ~30-60 seconds (depending on MC sample sizes)
- Figure generation: ~5-10 seconds (PyVista rendering is slowest)
- Total end-to-end: ~1-2 minutes
