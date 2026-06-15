# emergent-geometry-heisenberg

Numerical exact diagonalisation protocols and spectral graph theory pipelines for extracting smooth spatial differential operators directly from the entanglement structure of pure quantum ground states.

## Overview

This repository provides an operational, data-driven protocol to extract smooth spatial differential operators directly from a pure quantum ground state without presupposing a background manifold or metric framework.

Using an open isotropic antiferromagnetic Heisenberg spin chain ($N = 6$ to $12$), the script computes the ground-state von Neumann mutual information matrix to construct a normalized, scale-independent relational adjacency matrix $A_{ij}$ and its corresponding graph Laplacian:
$$L = D - A$$

The resulting operator naturally satisfies the definition of a continuous one-dimensional Laplace–Beltrami operator in the continuum limit, matching exact spatial calculus axioms:
* **Constant Zero-Mode:** Confirms $\lambda_0 < 10^{-15}$ with a flat constant eigenvector $v_0 = (1, 1, \dots, 1)^T$.
* **Emergent Neumann Boundaries:** Organically enforces a vanishing spatial derivative at the boundary endpoints due to power-law correlation decay.
* **Quadratic Dispersion Scaling:** Verifies that low-lying discrete modes satisfy $\lambda_k \propto k^2$ with an infrared fit of $R^2 > 0.998$.

## Repository Contents

* `main.tex`: Full publication-ready LaTeX source file.
* `main.pdf`: Compiled, high-resolution six-page pre-print layout.
* `numerical.py`: Self-contained Python implementation script utilizing exact ground-state diagonalisation.
* `/output`: Output spectral plots showing eigenmode wavefunctions, quadratic fits, finite-size scaling, and the relational mutual-information heat map.

---
**ORCID:** [0009-0009-8933-857X](https://orcid.org)
