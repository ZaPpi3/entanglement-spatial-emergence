# Emergent Geometry from Entanglement: A Numerical Demonstration Using Heisenberg Spin Chains

[![Paper Link](https://shields.io)](https://arxiv.org)
[![License: MIT](https://shields.io)](https://opensource.org)

Official repository for the paper **“Emergent Geometry from Entanglement: A Numerical Demonstration Using Heisenberg Spin Chains” (2026)** by Paul Jarvis. This work presents a numerical pipeline demonstrating how spatial geometry emerges from quantum entanglement in Heisenberg spin chains, without prior assumptions.

---

## 🔍 Conceptual Overview

The framework treats the ground state of an isotropic antiferromagnetic Heisenberg spin chain as a pure correlation metric. It constructs an information-derived graph Laplacian to analyze its spectral and structural properties across various system sizes ($N = 6$ to $18$), topologies (Open vs. Periodic), and phase boundaries ($\Delta$).

## 🧠 Key Scientific Findings

*   **Emergent Boundaries:** The pipeline reproduces Neumann boundary conditions natively from entanglement data (OBC) and identifies the expected constant zero mode.
*   **Bulk Mass Scale:** Finite-size scaling reveals a non-zero intercept in the first non-zero mode,, indicating a genuine bulk mass scale. This effect is significantly magnified under Periodic Boundary Conditions (PBC), pointing to long-range entanglement monogamy constraints.
*   **Criticality Dependence:** Smooth, CFT-matching geometries arise in the critical phase ($\Delta \le 1$), while geometric structures collapse in the gapped phase ($\Delta > 1$).

## ⚙️ The Computational Pipeline

The codebase implements a 5-step process:
1.  **Exact Diagonalization (ED):** Generates $|\psi_0\rangle$ (up to $N=18$).
2.  **Quantum Information Processing:** Calculates $I(i, j)$ via reduced density matrices.
3.  **Matrix Normalization:** Normalizes $I(i, j)$ to build the adjacency matrix.
4.  **Spectral Decomposition:** Solves $L = D - A$ for eigenvalues and modes.
5.  **Statistical Validation:** Employs AICc and bootstrap resampling.

## 📁 Repository Structure

*   `main.pdf` / `main.tex` : Manuscript and LaTeX source.
*   `code/testing.py` : Production code (ED, solvers, stats).
*   `output/` : Figures 1-12 in high-res (.png/.pdf).


