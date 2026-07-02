# Emergent Geometry from Entanglement: A Numerical Demonstration Using Heisenberg Spin Chains

[![Paper Link](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **"Emergent Geometry from Entanglement: A Numerical Demonstration Using Heisenberg Spin Chains" (2026)** by Paul Jarvis. This work presents a numerical pipeline that constructs a graph Laplacian from the ground-state entanglement structure of Heisenberg spin chains, tests it against the diagnostics one would expect of a continuum Laplace–Beltrami operator, and — critically — stress-tests its own headline result against methodological choices that turn out to matter.

---

## 🔍 Conceptual Overview

The framework treats the ground state of an isotropic antiferromagnetic Heisenberg spin chain as a pure correlation dataset. It constructs an information-derived graph Laplacian from the ground-state mutual-information matrix and analyzes its spectral and structural properties across system sizes ($N = 6$ to $18$), topologies (Open vs. Periodic), and phase boundaries ($\Delta$).

## 🧠 Key Scientific Findings

- **Emergent boundaries.** The pipeline reproduces Neumann-boundary-like behaviour natively from entanglement data under OBC: a constant zero mode and suppressed boundary slopes in the low-lying eigenmodes.
- **A nonzero finite-size intercept — but not a settled mass scale.** Finite-size scaling of the lowest nonzero eigenvalue $\lambda_1$ reveals a nonzero intercept under OBC that persists, and grows in magnitude, under PBC — ruling out a simple boundary-artefact explanation. However, we show this result is **not robust to two methodological choices**: the intercept's magnitude varies by roughly an order of magnitude across different (equally defensible) adjacency-normalization schemes, and an AICc-preferred finite-size model that includes the logarithmic corrections expected near a critical point substantially reduces the OBC intercept and **reverses its sign** under PBC. We report these sensitivities explicitly rather than asserting a "genuine bulk mass scale"; see Sec. III.L of the paper.
- **Dispersion tension.** AICc favours a *linear* rather than quadratic $\lambda_k(k)$ dispersion, which is in tension with the quadratic dispersion expected of a continuum Laplace–Beltrami operator. We flag this explicitly rather than treating the Laplace–Beltrami identification as established — it should be read as a qualitative, low-mode-index similarity pending larger $N$.
- **Criticality dependence.** Low-lying modes remain extended (low IPR) in the critical phase ($-1 < \Delta \le 1$) and become localized in the gapped phase ($\Delta > 1$), signalling a collapse of geometric structure away from criticality.
- **Luttinger-liquid agreement.** The measured mutual-information decay exponent tracks the Luttinger-liquid prediction closely across the XXZ critical line, under both OBC and PBC.

**Bottom line:** the pipeline is a working validation tool for information-theoretic approaches to emergent geometry, but the size and even the sign of the finite-size mass term are not yet settled at the system sizes exact diagonalization can reach. Distinguishing between the competing finite-size models will require larger $N$ (DMRG/MPS).

## ⚙️ The Computational Pipeline

The codebase implements a 6-step process:

1. **Exact Diagonalization (ED):** Generates $|\psi_0\rangle$ for $N = 6, 8, 10, 12, 14, 16, 18$ (up to $N=18$), OBC and PBC.
2. **Quantum Information Processing:** Calculates the mutual-information matrix $I(i, j)$ via reduced density matrices.
3. **Matrix Normalization:** Normalizes $I(i, j)$ to build the adjacency matrix $A_{ij}$ and graph Laplacian $L = D - A$.
4. **Spectral Decomposition:** Solves for eigenvalues and eigenmodes of $L$.
5. **Statistical Validation:** AICc model selection, finite-size scaling, resampling diagnostics, synthetic-Laplacian benchmarking, boundary diagnostics, Luttinger-parameter sweep, and MI-decay exponent extraction.
6. **Robustness Diagnostics (new):** Repeats the finite-size fit under alternative adjacency normalizations (mean, unnormalized) and under finite-size models with logarithmic corrections, compared via AICc, plus a leave-one-out sensitivity check and a ground-state-gap sanity check. See Sec. III.L.

All 6 steps run end-to-end from a single script, `code/testing.py`.

## 📁 Repository Structure

- `main.pdf` / `main.tex` : Manuscript and LaTeX source (includes Sec. III.L, "Robustness of the finite-size intercept").
- `code/testing.py` : Full production pipeline — ED, MI/Laplacian construction, all statistical tests, and all robustness checks, in one script.
- `figures/` : All 16 figures referenced in the manuscript, in high resolution (.png).

## 🔁 Reproducing the results

```bash
pip install qutip numpy scipy matplotlib
python code/testing.py
```

This regenerates every figure and every number quoted in the paper, including the robustness-check tables in Sec. III.L. Runtime is a few minutes on a laptop; $N=18$ exact diagonalization is the most expensive step (~5s per Hamiltonian).

## ⚠️ A note on interpretation

This repository documents both a result and our own audit of that result. The original analysis (Secs. III.A–III.K) reports a statistically significant finite-size intercept under a single normalization and a single finite-size ansatz. Sec. III.L shows that conclusion changes — sometimes substantially — under reasonable alternative choices. We think reporting that openly is more useful than a cleaner-looking but less honest headline result, and we'd rather the repository reflect the same standard the paper argues for: check your pipeline against the alternatives before trusting the number it gives you.

## License

MIT — see `LICENSE`.
