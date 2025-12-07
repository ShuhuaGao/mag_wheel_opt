# Surrogate-Assisted Optimal Design of Halbach Magnetic Wheels for Tubular Steel Structure Inspection Robots

This repository implements a surrogate-based optimization workflow for a magnetic-adhesion climbing robot wheel.  
High-fidelity samples are generated with COMSOL, a deep learning surrogate (FT-Transformer) is trained on these samples, and a hybrid GA–DE optimizer searches for designs that maximize the adhesion-to-weight ratio \(F_y / G\) under safety constraints. Final candidate designs are then validated again in COMSOL.

---

## 1. Directory Structure

- `COMSOL_FEM/`
  - `lhs_generate_and_save.m`  
    Generate the FEM dataset using Latin Hypercube Sampling (LHS) in COMSOL/MATLAB.
  - `filter.m`  
    Filter and clean the raw dataset (e.g., remove invalid points, enforce basic bounds/consistency).
  - `testindividual.m`  
    Validate a specific design (set of design variables) in COMSOL to check the surrogate/optimizer results.
  - `model/wheel.mph`
    COMSOL model for the magnetic wheel.

- `Surrogate_optimization/`
  - `ft-transformer.py`  
    Train the FT-Transformer surrogate model (regressing from design variables to \(F_y\), \(G\), and \(F_y/G\)).
    The script saves the trained weights and scalers for later use.
  - `GA-DE.py`  
    Run the hybrid GA–DE algorithm using the trained surrogate model to maximize \(F_y/G\) under constraints.
    Produces CSV logs and convergence/diversity figures.

---

## 2. Prerequisites

### COMSOL / MATLAB side

- MATLAB (with access to COMSOL via LiveLink or equivalent API)
- COMSOL Multiphysics model configured for the magnetic wheel
- Ability to run `.m` scripts that drive COMSOL simulations:
  - `lhs_generate_and_save.m`
  - `filter.m`
  - `testindividual.m`

### Python side

- Python 3.8+  
- Recommended packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `torch` (PyTorch)
  - `joblib`

