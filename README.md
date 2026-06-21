# Emergent Geometry from Entanglement

### A Numerical Demonstration Using Heisenberg Spin Chains

This repository contains the full numerical pipeline used in the paper:
**“Emergent Geometry from Entanglement: A Numerical Demonstration Using Heisenberg Spin Chains” (2026)**

## 🔍 Overview
We study the isotropic antiferromagnetic Heisenberg spin chain (XXZ, $\Delta = 1$) using exact diagonalization for systems **N = 6–18**. By constructing a graph Laplacian from the ground-state **mutual-information matrix**, we demonstrate that smooth spatial geometry emerges from pure entanglement, featuring:
*   A constant zero mode corresponding to a Laplace–Beltrami operator.
*   Emergent Neumann boundary conditions under OBC.
*   A robust, finite-size mass gap that distinguishes between OBC ($b_{\text{OBC}} \approx 0.145$) and PBC ($b_{\text{PBC}} \approx 1.170$).
*   Criticality-dependent structure ($|\Delta| \le 1$).

## 📁 Repository Structure
*   `main.pdf/tex` : The final paper.
*   `code/` : Python implementation and validation (`testing.py`).
*   `output/` : Generated plots.

## ▶ Running the Pipeline
Run `python main.py` to reproduce the figures.
