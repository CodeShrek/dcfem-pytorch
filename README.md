# dcFEM-PyTorch

A minimal differentiable Finite Element Method (FEM)-style solver implemented in PyTorch.

This repository demonstrates the core idea behind my GSoC 2026 proposal for DeepChem:  
**dcFEM (Differentiable FEM Framework)** — enabling end-to-end differentiable PDE solvers for scientific machine learning.

---

##  Features

- Differentiable FEM-like assembly  
- Conjugate Gradient (CG) solver (autograd-compatible)  
- End-to-end gradient flow through solver  
- Inverse problem: recovery of diffusion coefficient \( D \)  
- Visualization of true vs predicted solution  

---

##  Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux

# Windows:
# venv\Scripts\activate
```
##Install dependencies:
```bash
pip install torch matplotlib
```
###RUN:
```bash
python -m problems.inverse_diffusion
```
