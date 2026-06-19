# Emergent Geometry from Entanglement  
### A Numerical Demonstration Using Heisenberg Spin Chains

This repository contains the full numerical pipeline used in the paper:

**“Emergent Geometry from Entanglement: A Numerical Demonstration Using Heisenberg Spin Chains” (2026)**  
*Paul Jarvis — Independent Researcher, United Kingdom*

The goal of this project is to show, using only raw quantum correlations, that **smooth spatial geometry can emerge directly from the entanglement structure of a many‑body ground state**. No geometric assumptions, coordinates, or background manifold are introduced at any stage.

---

## 🔍 Overview

We study the isotropic antiferromagnetic Heisenberg spin chain (XXZ with Δ = 1) using exact diagonalization for system sizes **N = 6–14**. From the ground state, we compute the **mutual‑information matrix** and use it to construct a **graph Laplacian**.  

A comprehensive suite of numerical tests shows that this operator behaves exactly like the **one‑dimensional Laplace–Beltrami operator with Neumann boundary conditions**.

This provides a concrete, data‑driven demonstration of **emergent geometry from entanglement**.

---

## 🧠 Key Results

### ✔ 1. Constant Zero Mode  
The smallest eigenvalue is numerically zero, and the corresponding eigenvector is constant to machine precision — the defining signature of a Laplace–Beltrami operator.

### ✔ 2. Neumann Boundary Conditions  
Boundary slopes of low‑lying eigenmodes are strongly suppressed relative to bulk slopes, confirming emergent Neumann behaviour without any imposed boundary conditions.

### ✔ 3. Quadratic Dispersion (λₖ ∝ k²)  
Using AICc model selection across 8 modes and 5 system sizes, the **quadratic model is overwhelmingly preferred** over linear, cubic, or quartic alternatives.

### ✔ 4. Finite‑Size Scaling  
The first non‑zero eigenvalue scales as **λ₁ ∝ 1/N²**, with intercept consistent with zero. This is the expected continuum scaling of a second‑derivative operator.

### ✔ 5. Criticality is Necessary  
Sweeping the anisotropy parameter Δ shows:

- Δ = 1 (critical): smooth modes, quadratic dispersion  
- Δ = 2 (gapped): localized modes, quadratic scaling collapses  

Smooth geometry emerges **only** in the critical, power‑law‑entangled phase.

### ✔ 6. Sensitivity to the Luttinger Parameter  
The Laplacian gap λ₁ correlates strongly with the Bethe‑ansatz Luttinger parameter K(Δ):

- Pearson ≈ 0.99  
- Spearman ≈ 1.00  
- Leave‑one‑out correlations stable  

The construction is sensitive to **universal CFT data**, not just microscopic details.

---

## 📁 Repository Structure

*   **`main.pdf`**: The final, production-compiled document
*   **`main.tex`**: Contains the raw LaTeX source text, document preambles, and bibliography tags.
    *   `main.tex`: The unified, consolidated TeX source file ready for local compilation or direct upload to the arXiv production servers.
*   **`code/`**: The complete Python implementation of the numerical models supporting the theoretical framework.
    *   `numerical_test.py`
*   **`output/`**: test output .png files.

---

## ▶️ Running the Pipeline

All figures in the paper can be reproduced by running:

```bash
python main.py
