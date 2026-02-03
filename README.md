# Heat Flow Random Walk on Virtual Persistence Diagram Groups

## Current Status

Setting up project structure and environment for implementing the full pipeline:
- OHSU graph selection and filtration
- Persistent homology computation
- Virtual persistence diagram (VPD) group construction
- Translation-invariant random walk on VPD group
- Heat kernel and invariant approximation
- Lipschitz/ultracontractive/resolvent bounds evaluation

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

- Download OHSU dataset to `data/`
- Implement graph loading and Laplacian computation
- Implement edge labeling via Laplacian pseudoinverse
- Implement clique filtration and persistent homology
- Implement VPD group and metric
- Implement random walk and heat kernel approximation
- Implement bounds evaluation

## Known Issues

None yet.
