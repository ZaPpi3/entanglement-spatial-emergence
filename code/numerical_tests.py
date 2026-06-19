"""
Numerical tests for emergent geometry from ground-state entanglement.

This version extends the original script to actually test the claims made
in the text, rather than just asserting them:

  1. Critical vs gapped comparison (XXZ chain, Delta=1 vs Delta=2)
     -> tests whether "substrate criticality is necessary" for smooth
        emergent geometry.
  2. Quantitative boundary-slope test
     -> tests the "Neumann boundary condition" claim directly instead of
        eyeballing the eigenmode plot.
  3. Extended finite-size scaling (more N values, fit + residuals/CI)
  4. Quadratic-vs-alternative model comparison (AIC) instead of relying
     on a high R^2 from a 3-parameter fit to 4 points.
  5. Explicit zero-mode constancy check.
  6. Robustness check with an alternative MI normalization.

Notes on cost: exact diagonalization scales as 2^N. N up to 14 is fine on
a laptop; N=16 is slow but doable; N=18+ needs symmetry-sector reduction
or a different ground-state solver (e.g. DMRG via tenpy/ITensor) which is
out of scope here but flagged where relevant.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import eigh
from qutip import *

# ============================================================
# Utility functions
# ============================================================

def op_on_site(op, i, N):
    """Place single-site operator op at site i in an N-site chain."""
    ops = [qeye(2)] * N
    ops[i] = op
    return tensor(ops)

def build_heisenberg_hamiltonian(N, J=1.0):
    """Isotropic Heisenberg chain (XXZ with Delta=1). Critical/gapless."""
    return build_xxz_hamiltonian(N, J=J, Delta=1.0)

def build_xxz_hamiltonian(N, J=1.0, Delta=1.0):
    """
    XXZ chain: H = J * sum_i [ Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Delta * Sz_i Sz_{i+1} ]

    Delta = 1.0  -> isotropic Heisenberg, gapless/critical (used in the
                    original analysis).
    Delta > 1.0  -> Ising-like easy-axis regime, gapped (Neel-ordered),
                    used here as the negative control for the criticality
                    claim.
    """
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H = 0
    for i in range(N - 1):
        H += J * (op_on_site(sx, i, N) * op_on_site(sx, i+1, N) +
                  op_on_site(sy, i, N) * op_on_site(sy, i+1, N) +
                  Delta * op_on_site(sz, i, N) * op_on_site(sz, i+1, N))
    return H

def mutual_information_matrix(psi0, N):
    """
    Compute NxN mutual-information matrix.

    IMPORTANT: this operates on psi0 (the ket) directly via ptrace,
    rather than first building rho = psi0 * psi0.dag(). For N=14 that
    full dense rho would be a 16384x16384 complex matrix (~4.3 GB), which
    is unnecessary -- qutip's ptrace works directly on kets and avoids
    ever materializing that object. This cuts both memory and runtime
    substantially at the larger N values used in the pooled fit and
    finite-size scaling sweep.

    Also caches single-site reduced states (rho_i depends only on i, not
    on the pair (i,j)), since the original implementation recomputed
    rho_i and rho_j repeatedly inside the i,j loop.
    """
    single_site_cache = {}

    def get_single_site(i):
        if i not in single_site_cache:
            single_site_cache[i] = psi0.ptrace(i)
        return single_site_cache[i]

    I = np.zeros((N, N))

    for i, j in product(range(N), range(N)):
        if i == j:
            continue
        if I[j, i] != 0:  # symmetric: I_ij == I_ji, reuse if already computed
            I[i, j] = I[j, i]
            continue
        rho_i = get_single_site(i)
        rho_j = get_single_site(j)
        rho_ij = psi0.ptrace([i, j])
        Si = entropy_vn(rho_i)
        Sj = entropy_vn(rho_j)
        Sij = entropy_vn(rho_ij)
        I[i, j] = Si + Sj - Sij

    return I

def laplacian_from_MI(I, normalization="max"):
    """
    Construct normalized adjacency and Laplacian.

    normalization:
        "max"  -> A = I / max(I)            (original choice)
        "mean" -> A = I / mean(I[I>0])      (robustness check; tests
                   whether results are an artifact of max-normalization)
    """
    if normalization == "max":
        A = I / np.max(I)
    elif normalization == "mean":
        nz = I[I > 0]
        A = I / np.mean(nz)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    np.fill_diagonal(A, 0)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L

import time

_SPECTRUM_CACHE = {}

def compute_ground_state(N, Delta=1.0):
    print(f"   ... computing ground state (N={N}, Δ={Delta})", flush=True)
    t0 = time.time()
    H = build_xxz_hamiltonian(N, Delta=Delta)
    evals, evecs = H.eigenstates(eigvals=2, sparse=True)
    print(f"   ... ground state done in {time.time()-t0:.1f}s", flush=True)
    return evecs[0]

def laplacian_spectrum(N, Delta=1.0, normalization="max"):
    """
    Cached: identical (N, Delta, normalization) calls reuse the previous
    result instead of redoing the ground-state + MI + diagonalization
    pipeline. This matters because generate_pooled_spectrum_fit and
    generate_scaling_plot both sweep the same N values, and several other
    functions reuse N=12 -- without caching, N=14 alone could get computed
    multiple times.
    """
    key = (N, Delta, normalization)
    if key in _SPECTRUM_CACHE:
        print(f"   ... using cached result for N={N}, Δ={Delta}, "
              f"norm={normalization}", flush=True)
        return _SPECTRUM_CACHE[key]

    psi0 = compute_ground_state(N, Delta=Delta)
    print(f"   ... computing mutual-information matrix (N={N}, "
          f"{N*(N-1)} pairs)", flush=True)
    t0 = time.time()
    I = mutual_information_matrix(psi0, N)
    print(f"   ... MI matrix done in {time.time()-t0:.1f}s", flush=True)
    L = laplacian_from_MI(I, normalization=normalization)
    t0 = time.time()
    evals_L, vecs_L = eigh(L)
    print(f"   ... Laplacian diagonalized in {time.time()-t0:.1f}s", flush=True)

    result = (np.real(evals_L), np.real(vecs_L), I)
    _SPECTRUM_CACHE[key] = result
    return result

# ============================================================
# 1. Spectrum + quadratic-vs-alternative model comparison
#    (spectrum.png)
# ============================================================

def aic(y, yfit, k_params):
    """Akaike Information Criterion for least-squares fits."""
    n = len(y)
    rss = np.sum((y - yfit) ** 2)
    rss = max(rss, 1e-300)  # guard against log(0)
    return n * np.log(rss / n) + 2 * k_params

def aicc(y, yfit, k_params):
    """
    Corrected AIC for small sample sizes (Hurvich & Tsai).
    AICc = AIC + 2*k*(k+1) / (n-k-1)

    Reduces to plain AIC as n -> infinity, but adds a much stronger
    penalty for extra parameters when n is close to k_params+1 -- exactly
    the regime we're in with only a handful of low-lying modes. When
    n <= k_params + 1 the correction term blows up (a model with as many
    parameters as points has zero residual and is *always* "preferred"
    by uncorrected AIC), so we return +inf for those degenerate cases
    instead of a misleadingly finite number.
    """
    n = len(y)
    base = aic(y, yfit, k_params)
    denom = n - k_params - 1
    if denom <= 0:
        return np.inf
    return base + (2 * k_params * (k_params + 1)) / denom

def fit_and_compare(k_vals, low, max_degree=3):
    """
    Fit polynomial models of degree 1..max_degree and compare via AICc
    (preferred) and plain AIC (kept for reference/transparency). With
    few points, AICc is the more trustworthy criterion since plain AIC
    is biased toward high-degree models that simply interpolate the data.
    """
    names = {1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic"}
    results = {}
    n_points = len(k_vals)
    for deg in range(1, max_degree + 1):
        if deg >= n_points:
            # Can't meaningfully fit (or it's a pure interpolation) -- skip.
            continue
        name = names.get(deg, f"degree-{deg}")
        coeffs = np.polyfit(k_vals, low, deg)
        yfit = np.polyval(coeffs, k_vals)
        ss_res = np.sum((low - yfit) ** 2)
        ss_tot = np.sum((low - np.mean(low)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        results[name] = {
            "coeffs": coeffs,
            "r2": r2,
            "aic": aic(low, yfit, deg + 1),
            "aicc": aicc(low, yfit, deg + 1),
        }
    return results

def generate_spectrum_plot(N=12, n_modes=8):
    """
    n_modes=8 (rather than the original 4) so the model comparison has
    enough degrees of freedom to be meaningful. With only 4 points, a
    cubic fit has as many parameters as data points and trivially gets
    zero residual -- AICc with n_modes=4 returns +inf for any model with
    3+ params for exactly this reason. n_modes=8 leaves real headroom to
    distinguish quadratic from cubic/quartic.

    Note: at N=12 there are only 12 sites, so modes much beyond k~8 start
    running into finite-size/lattice effects rather than continuum
    physics -- n_modes=8 is close to the practical ceiling for this N.
    """
    evals_L, vecs_L, I = laplacian_spectrum(N, Delta=1.0)

    k_vals = np.arange(1, n_modes + 1)
    low = evals_L[1:n_modes + 1]

    model_comparison = fit_and_compare(k_vals, low)
    # Use AICc, not plain AIC -- see aicc() docstring for why plain AIC
    # is unreliable here.
    valid = {m: r for m, r in model_comparison.items() if np.isfinite(r["aicc"])}
    best_model = min(valid, key=lambda m: valid[m]["aicc"]) if valid else None

    print(f"[spectrum] N={N}, {n_modes} modes, model comparison "
          f"(lower AICc = better; +inf = degenerate/can't be evaluated):")
    for name, res in model_comparison.items():
        aicc_str = f"{res['aicc']:.2f}" if np.isfinite(res['aicc']) else "inf (degenerate)"
        print(f"   {name:10s} R^2={res['r2']:.6f}  AIC={res['aic']:.2f}  AICc={aicc_str}")
    print(f"   -> preferred model (by AICc): {best_model}")

    coeffs = model_comparison["quadratic"]["coeffs"]
    fit_k = np.linspace(1, n_modes, 200)
    fit_lambda = np.polyval(coeffs, fit_k)

    plt.figure(figsize=(7, 5))
    plt.plot(k_vals, low, 'o', label='Numerical eigenvalues')
    plt.plot(fit_k, fit_lambda, '-', label='Quadratic fit')
    plt.xlabel("Mode number k")
    plt.ylabel("Eigenvalue λ_k")
    title_note = "preferred by AICc" if best_model == "quadratic" else \
                  f"NOTE: AICc prefers {best_model}, not quadratic"
    plt.title(f"Low-lying Laplacian eigenvalues (N={N}, {n_modes} modes)\n{title_note}")
    plt.grid(True)
    plt.legend()
    plt.savefig("spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved spectrum.png")
    return model_comparison

# ============================================================
# 2. Finite-size scaling plot, extended N range + CI
#    (scaling_lambda1.png)
# ============================================================

def first_nonzero_eigenvalue(N, Delta=1.0):
    evals_L, _, _ = laplacian_spectrum(N, Delta=Delta)
    return evals_L[1]

def generate_scaling_plot(Ns=(6, 8, 10, 12, 14)):
    lam1 = [first_nonzero_eigenvalue(N) for N in Ns]
    invN2 = [1.0 / (N ** 2) for N in Ns]

    coeffs, cov = np.polyfit(invN2, lam1, 1, cov=True)
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    fit_x = np.linspace(0, max(invN2), 200)
    fit_y = np.polyval(coeffs, fit_x)

    print(f"[scaling] lambda_1 = ({coeffs[0]:.4f} +/- {slope_err:.4f}) * (1/N^2) "
          f"+ ({coeffs[1]:.5f} +/- {intercept_err:.5f})")
    print(f"          intercept consistent with 0 at N->inf: "
          f"{abs(coeffs[1]) < 2*intercept_err}")

    plt.figure(figsize=(7, 5))
    plt.plot(invN2, lam1, 'o', label=r'Numerical $\lambda_1$')
    plt.plot(fit_x, fit_y, '-',
              label=f'Linear fit (intercept={coeffs[1]:.4f}±{intercept_err:.4f})')
    plt.xlabel(r'$1/N^2$')
    plt.ylabel(r'$\lambda_1$')
    plt.title(f"Finite-size scaling, N = {list(Ns)}")
    plt.grid(True)
    plt.legend()
    plt.savefig("scaling_lambda1.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved scaling_lambda1.png")
    return Ns

def generate_pooled_spectrum_fit(Ns=(6, 8, 10, 12, 14), n_modes=8):
    """
    Pool (k, lambda_k) pairs across all N in Ns into one combined dataset
    and run the model comparison once on the full pool. This is the
    single most powerful test in the script: instead of 4 or 8 points
    from one system size, we get n_modes * len(Ns) points, which gives
    AICc plenty of headroom to actually discriminate between models
    instead of just rewarding whichever curve has the most free
    parameters.

    We fit lambda_k(N) ~ f(k) with N treated as a repeated-measurement
    dimension (i.e. we don't rescale k or lambda by N here -- this tests
    whether a single k-dependence form, e.g. quadratic, holds reasonably
    well across sizes for the modes accessible at each N. This is a
    simpler pooling than a full finite-size scaling collapse, but it
    already directly addresses the "only 4 points" objection.
    """
    all_k, all_lambda = [], []
    for N in Ns:
        evals_L, _, _ = laplacian_spectrum(N, Delta=1.0)
        k_max = min(n_modes, N - 2)  # can't exceed available nonzero modes
        all_k.extend(range(1, k_max + 1))
        all_lambda.extend(evals_L[1:k_max + 1])

    all_k = np.array(all_k)
    all_lambda = np.array(all_lambda)
    n_total = len(all_k)

    comparison = fit_and_compare(all_k, all_lambda, max_degree=4)
    valid = {m: r for m, r in comparison.items() if np.isfinite(r["aicc"])}
    best_model = min(valid, key=lambda m: valid[m]["aicc"]) if valid else None

    print(f"[pooled fit] {n_total} (k, lambda_k) pairs pooled across N={list(Ns)}:")
    for name, res in comparison.items():
        aicc_str = f"{res['aicc']:.2f}" if np.isfinite(res['aicc']) else "inf"
        print(f"   {name:10s} R^2={res['r2']:.6f}  AICc={aicc_str}")
    print(f"   -> preferred model on pooled data: {best_model}")

    plt.figure(figsize=(7, 5))
    for N in Ns:
        evals_L, _, _ = laplacian_spectrum(N, Delta=1.0)
        k_max = min(n_modes, N - 2)
        ks = np.arange(1, k_max + 1)
        plt.plot(ks, evals_L[1:k_max + 1], 'o', alpha=0.6, label=f"N={N}")

    if best_model:
        coeffs = comparison[best_model]["coeffs"]
        fit_k = np.linspace(1, max(all_k), 200)
        plt.plot(fit_k, np.polyval(coeffs, fit_k), 'k--',
                  label=f"Best fit ({best_model})")

    plt.xlabel("Mode number k")
    plt.ylabel("Eigenvalue λ_k")
    plt.title(f"Pooled spectrum across system sizes\npreferred model: {best_model}")
    plt.grid(True)
    plt.legend()
    plt.savefig("pooled_spectrum_fit.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved pooled_spectrum_fit.png")
    return comparison

# ============================================================
# 3. Eigenmode plot + quantitative Neumann boundary-slope test
#    (eigenmodes.png)
# ============================================================

def boundary_slope_test(vecs_L, N, modes=(1, 2, 3)):
    """
    Quantify how flat each eigenvector is at the two boundaries
    (discrete derivative) vs. its typical bulk variation. A small
    ratio supports a Neumann-type (zero-slope) boundary condition;
    a ratio near 1 does not.
    """
    report = {}
    for k in modes:
        v = vecs_L[:, k]
        left_slope = v[1] - v[0]
        right_slope = v[-1] - v[-2]
        bulk_slopes = np.diff(v[1:-1])
        bulk_typical = np.mean(np.abs(bulk_slopes)) if len(bulk_slopes) else np.nan
        ratio_left = abs(left_slope) / bulk_typical if bulk_typical else np.nan
        ratio_right = abs(right_slope) / bulk_typical if bulk_typical else np.nan
        report[k] = dict(left_slope=left_slope, right_slope=right_slope,
                          bulk_typical=bulk_typical,
                          ratio_left=ratio_left, ratio_right=ratio_right)
    return report

def generate_eigenmode_plot(N=12):
    evals_L, vecs_L, I = laplacian_spectrum(N, Delta=1.0)
    x = np.arange(N)

    # Zero-mode constancy check
    mode0 = vecs_L[:, 0]
    print(f"[eigenmodes] zero mode std/mean = "
          f"{np.std(mode0)/abs(np.mean(mode0)):.2e} (should be ~0 for a "
          f"constant zero mode)")

    bt = boundary_slope_test(vecs_L, N)
    print("[eigenmodes] boundary slope / bulk slope ratios (Neumann test):")
    for k, r in bt.items():
        print(f"   mode k={k}: left={r['ratio_left']:.3f}  "
              f"right={r['ratio_right']:.3f}  (≈0 => Neumann-like, ≈1 => not)")

    plt.figure(figsize=(7, 5))
    for k in [1, 2, 3]:
        plt.plot(x, vecs_L[:, k], 'o-', label=f"Mode k={k}")
    plt.xlabel("Site index")
    plt.ylabel("Eigenvector amplitude")
    plt.title("Low-lying Laplacian eigenmodes (critical chain, Δ=1)")
    plt.grid(True)
    plt.legend()
    plt.savefig("eigenmodes.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved eigenmodes.png")
    return bt

# ============================================================
# 4. Mutual-information heatmap (mi_heatmap.png)
# ============================================================

def generate_mi_heatmap(N=12):
    _, _, I = laplacian_spectrum(N, Delta=1.0)

    plt.figure(figsize=(6, 5))
    plt.imshow(I, cmap='viridis', origin='lower')
    plt.colorbar(label="Mutual information")
    plt.title("Mutual-information matrix (critical chain, Δ=1)")
    plt.xlabel("Site j")
    plt.ylabel("Site i")
    plt.savefig("mi_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved mi_heatmap.png")

# ============================================================
# 5. NEW: Critical vs gapped comparison (criticality_test.png)
#    Directly tests "substrate criticality is necessary"
# ============================================================

def generate_criticality_comparison(N=12, deltas=(1.0, 2.0)):
    fig, axes = plt.subplots(1, len(deltas), figsize=(6 * len(deltas), 5),
                               sharey=True)
    if len(deltas) == 1:
        axes = [axes]

    summary = {}
    for ax, Delta in zip(axes, deltas):
        evals_L, vecs_L, I = laplacian_spectrum(N, Delta=Delta)
        x = np.arange(N)
        for k in [1, 2, 3]:
            ax.plot(x, vecs_L[:, k], 'o-', label=f"k={k}")

        k_vals = np.arange(1, 5)
        low = evals_L[1:5]
        comparison = fit_and_compare(k_vals, low)
        r2_quad = comparison["quadratic"]["r2"]
        regime = "critical (gapless)" if Delta == 1.0 else "gapped (Δ>1)"
        ax.set_title(f"Δ={Delta} — {regime}\nquadratic R²={r2_quad:.4f}")
        ax.set_xlabel("Site index")
        ax.grid(True)
        ax.legend()
        summary[Delta] = r2_quad

    axes[0].set_ylabel("Eigenvector amplitude")
    plt.suptitle("Eigenmode smoothness: critical vs gapped substrate")
    plt.tight_layout()
    plt.savefig("criticality_test.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("[criticality] quadratic R^2 by Delta:", summary)
    print("Saved criticality_test.png")
    return summary

# ============================================================
# 6. NEW: Robustness check — alternative MI normalization
# ============================================================

def generate_robustness_check(N=12):
    results = {}
    for norm in ("max", "mean"):
        psi0 = compute_ground_state(N, Delta=1.0)
        I = mutual_information_matrix(psi0, N)
        L = laplacian_from_MI(I, normalization=norm)
        evals_L, _ = eigh(L)
        evals_L = np.real(evals_L)
        k_vals = np.arange(1, 5)
        low = evals_L[1:5]
        comparison = fit_and_compare(k_vals, low)
        results[norm] = comparison["quadratic"]["r2"]

    print("[robustness] quadratic R^2 under different normalizations:", results)
    return results

# ============================================================
# Run all figure generators
# ============================================================

if __name__ == "__main__":
    print("=== [1/7] spectrum plot ===", flush=True)
    generate_spectrum_plot()

    print("=== [2/7] pooled spectrum fit (N up to 14, slowest step) ===", flush=True)
    generate_pooled_spectrum_fit()

    print("=== [3/7] finite-size scaling plot ===", flush=True)
    generate_scaling_plot()

    print("=== [4/7] eigenmode plot ===", flush=True)
    generate_eigenmode_plot()

    print("=== [5/7] MI heatmap ===", flush=True)
    generate_mi_heatmap()

    print("=== [6/7] criticality comparison ===", flush=True)
    generate_criticality_comparison()

    print("=== [7/7] robustness check ===", flush=True)
    generate_robustness_check()

    print("\nAll figures and statistical checks generated successfully.")
