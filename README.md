# dcFEM Preliminary Work

## Features
- Differentiable FEM-like assembly
- Conjugate Gradient solver
- Gradient flow through solver
- Inverse problem (recover diffusion coefficient)

## Result
D_true = 2.0  
D_learned ≈ 2.0  

## Why this matters
Demonstrates feasibility of differentiable FEM for DeepChem.

## What this demonstrates

This experiment shows:

1. End-to-end differentiability of PDE solver
2. Gradient flow through CG solver
3. Successful recovery of unknown diffusion coefficient D

This directly validates the feasibility of differentiable FEM pipelines
proposed in my GSoC project (dcFEM).