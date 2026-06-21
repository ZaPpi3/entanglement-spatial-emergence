import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import eigh
from qutip import *

# ============================================================
# System sizes
# ============================================================

Ns_extended = (6, 8, 10, 12, 14, 16, 18)
Ns_core = (6, 8, 10, 12, 14)

# ============================================================
# Core operators and Hamiltonian (with PBC support)
# ============================================================

def op_on_site(op, i, N):
    ops = [qeye(2)] * N
    ops[i] = op
    return tensor(ops)

def build_xxz_hamiltonian(N, J=1.0, Delta=1.0, boundary="open"):
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H = 0

    for i in range(N - 1):
        H += J * (
            op_on_site(sx, i, N) * op_on_site(sx, i+1, N) +
            op_on_site(sy, i, N) * op_on_site(sy, i+1, N) +
            Delta * op_on_site(sz, i, N) * op_on_site(sz, i+1, N)
        )

    if boundary == "periodic":
        H += J * (
            op_on_site(sx, N-1, N) * op_on_site(sx, 0, N) +
            op_on_site(sy, N-1, N) * op_on_site(sy, 0, N) +
            Delta * op_on_site(sz, N-1, N) * op_on_site(sz, 0, N)
        )

    return H

def compute_ground_state(N, Delta=1.0, boundary="open"):
    H = build_xxz_hamiltonian(N, Delta=Delta, boundary=boundary)
    evals, evecs = H.eigenstates(eigvals=1, sparse=True)
    return evecs[0]

# ============================================================
# Mutual information and Laplacian
# ============================================================

def mutual_information_matrix(psi0, N):
    cache = {}
    def rho1(i):
        if i not in cache:
            cache[i] = psi0.ptrace(i)
        return cache[i]

    I = np.zeros((N, N))
    for i, j in product(range(N), range(N)):
        if i == j:
            continue
        if I[j, i] != 0:
            I[i, j] = I[j, i]
            continue
        rho_i = rho1(i)
        rho_j = rho1(j)
        rho_ij = psi0.ptrace([i, j])
        I[i, j] = entropy_vn(rho_i) + entropy_vn(rho_j) - entropy_vn(rho_ij)
    return I

def laplacian_from_MI(I, normalization="max"):
    if normalization == "max":
        A = I / np.max(I)
    elif normalization == "mean":
        A = I / np.mean(I[I > 0])
    else:
        raise ValueError("Unknown normalization")
    np.fill_diagonal(A, 0)
    D = np.diag(np.sum(A, axis=1))
    return D - A

_SPECTRUM_CACHE = {}

def laplacian_spectrum(N, Delta=1.0, normalization="max", boundary="open"):
    key = (N, Delta, normalization, boundary)
    if key in _SPECTRUM_CACHE:
        return _SPECTRUM_CACHE[key]

    psi0 = compute_ground_state(N, Delta=Delta, boundary=boundary)
    I = mutual_information_matrix(psi0, N)
    L = laplacian_from_MI(I, normalization)
    evals, vecs = eigh(L)
    result = (np.real(evals), np.real(vecs), I)
    _SPECTRUM_CACHE[key] = result
    return result

# ============================================================
# Luttinger parameter
# ============================================================

def luttinger_K(Delta):
    if not (-1 < Delta <= 1):
        raise ValueError("K only defined for -1 < Δ ≤ 1")
    gamma = np.arccos(Delta)
    return np.pi / (2 * (np.pi - gamma))

# ============================================================
# AICc utilities
# ============================================================

def aic(y, yfit, k):
    n = len(y)
    rss = max(np.sum((y - yfit)**2), 1e-300)
    return n * np.log(rss/n) + 2*k

def aicc(y, yfit, k):
    n = len(y)
    base = aic(y, yfit, k)
    if n - k - 1 <= 0:
        return np.inf
    return base + (2*k*(k+1))/(n-k-1)

# ============================================================
# TEST 1 — Zero mode + Neumann boundary slopes
# ============================================================

def test_zero_mode_and_boundary(N=12, modes=(1,2,3), boundary="open"):
    evals, vecs, _ = laplacian_spectrum(N, Delta=1.0, boundary=boundary)
    mode0 = vecs[:, 0]
    print(f"[TEST 1] Zero‑mode flatness: std/mean = {np.std(mode0)/abs(np.mean(mode0)):.2e}")

    for k in modes:
        v = vecs[:, k]
        left = abs(v[1] - v[0])
        right = abs(v[-1] - v[-2])
        bulk = np.mean(np.abs(np.diff(v[1:-1])))
        print(f"[TEST 1] Boundary‑slope suppression for mode k={k}: left/bulk={left/bulk:.3f}, right/bulk={right/bulk:.3f}")

    plt.figure()
    for k in modes:
        plt.plot(vecs[:, k], 'o-', label=f"k={k}")
    plt.title(f"Eigenmodes (boundary={boundary})")
    plt.legend()
    plt.savefig(f"test1_zero_mode_boundary_{boundary}.png")
    plt.close()

# ============================================================
# TEST 2 — Dispersion model comparison
# ============================================================

def test_dispersion_model(N=12, n_modes=8, boundary="open"):
    evals, _, _ = laplacian_spectrum(N, Delta=1.0, boundary=boundary)
    k = np.arange(1, n_modes+1)
    y = evals[1:n_modes+1]

    def fit(deg):
        coeffs = np.polyfit(k, y, deg)
        yfit = np.polyval(coeffs, k)
        return coeffs, yfit, aicc(y, yfit, deg+1)

    print(f"[TEST 2] Dispersion model comparison (N={N}, boundary={boundary})")
    for deg,name in [(1,"linear"),(2,"quadratic"),(3,"cubic")]:
        coeffs,yfit,score = fit(deg)
        ss_res = np.sum((y - yfit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        print(f"  {name:10s} fit: R²={r2:.4f}, AICc={score:.2f}")

# ============================================================
# TEST 2b — Pooled dispersion
# ============================================================

def test_pooled_dispersion(Ns=Ns_extended, n_modes=8, boundary="open"):
    all_k, all_l = [], []
    for N in Ns:
        evals,_,_ = laplacian_spectrum(N, Delta=1.0, boundary=boundary)
        kmax = min(n_modes, N-2)
        all_k.extend(range(1,kmax+1))
        all_l.extend(evals[1:kmax+1])

    k = np.array(all_k)
    y = np.array(all_l)

    def fit(deg):
        coeffs = np.polyfit(k, y, deg)
        yfit = np.polyval(coeffs, k)
        return coeffs, yfit, aicc(y, yfit, deg+1)

    print(f"[TEST 2b] Pooled dispersion across system sizes (boundary={boundary})")
    for deg,name in [(1,"linear"),(2,"quadratic"),(3,"cubic"),(4,"quartic")]:
        coeffs,yfit,score = fit(deg)
        ss_res = np.sum((y - yfit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        print(f"  {name:10s} fit: R²={r2:.4f}, AICc={score:.2f}")

# ============================================================
# TEST 3 — Finite-size scaling
# ============================================================

def test_finite_size_scaling(Ns=Ns_extended, boundary="open"):
    lam1 = []
    invN2 = []
    for N in Ns:
        evals,_,_ = laplacian_spectrum(N, Delta=1.0, boundary=boundary)
        lam1.append(evals[1])
        invN2.append(1/N**2)

    lam1 = np.array(lam1)
    invN2 = np.array(invN2)

    coeffs, cov = np.polyfit(invN2, lam1, 1, cov=True)
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0,0])
    intercept_err = np.sqrt(cov[1,1])

    print(f"[TEST 3] Finite‑size scaling of λ₁ (boundary={boundary})")
    print(f"λ₁(N) = ({slope:.4f}±{slope_err:.4f})*(1/N²) + ({intercept:.5f}±{intercept_err:.5f})")
    print(f"Intercept significance: {abs(intercept)/intercept_err:.1f}σ")

    xfit = np.linspace(0, max(invN2), 200)
    yfit = np.polyval(coeffs, xfit)

    plt.figure()
    plt.plot(invN2, lam1, 'o')
    plt.plot(xfit, yfit, '-')
    plt.title(f"Finite-size scaling (boundary={boundary})")
    plt.xlabel("1/N²")
    plt.ylabel("λ₁")
    plt.savefig(f"test3_finite_size_scaling_{boundary}.png")
    plt.close()
# ============================================================
# TEST 3b — Competing finite-size models
# ============================================================

def test_finite_size_models(Ns=Ns_extended, boundary="open"):
    lam1 = np.array([laplacian_spectrum(N, Delta=1.0, boundary=boundary)[0][1] for N in Ns])
    invN2 = np.array([1/N**2 for N in Ns])
    invN3 = np.array([1/N**3 for N in Ns])

    # M1: a/N^2 + b
    X1 = np.vstack([invN2, np.ones_like(invN2)]).T
    beta1, *_ = np.linalg.lstsq(X1, lam1, rcond=None)
    fit1 = X1 @ beta1
    aicc1 = aicc(lam1, fit1, 2)

    # M2: a/N^2
    X2 = invN2[:,None]
    beta2, *_ = np.linalg.lstsq(X2, lam1, rcond=None)
    fit2 = X2 @ beta2
    aicc2 = aicc(lam1, fit2, 1)

    # M3: a/N^2 + c/N^3 + b
    X3 = np.vstack([invN2, invN3, np.ones_like(invN2)]).T
    beta3, *_ = np.linalg.lstsq(X3, lam1, rcond=None)
    fit3 = X3 @ beta3
    aicc3 = aicc(lam1, fit3, 3)

    print(f"[TEST 3b] Competing finite‑size models (boundary={boundary})")
    print(f"  M1 (a/N² + b): AICc={aicc1:.3f}")
    print(f"  M2 (a/N²):     AICc={aicc2:.3f}")
    print(f"  M3 (a/N² + c/N³ + b): AICc={aicc3:.3f}")

# ============================================================
# TEST 3c — Bootstrap intercept
# ============================================================

def test_finite_size_bootstrap(Ns=Ns_extended, n_boot=2000, boundary="open"):
    lam1 = np.array([laplacian_spectrum(N, Delta=1.0, boundary=boundary)[0][1] for N in Ns])
    invN2 = np.array([1/N**2 for N in Ns])

    bs = []
    rng = np.random.default_rng()
    for _ in range(n_boot):
        idx = rng.integers(0, len(Ns), size=len(Ns))
        x = invN2[idx]
        y = lam1[idx]
        X = np.vstack([x, np.ones_like(x)]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        bs.append(beta[1])

    bs = np.array(bs)
    print(f"[TEST 3c] Bootstrap analysis of λ₁ intercept (boundary={boundary})")
    print(f"  Bootstrap mean={np.mean(bs):.5f}, std={np.std(bs):.5f}")
    print(f"  Significance={abs(np.mean(bs))/np.std(bs):.1f}σ")

    plt.figure()
    plt.hist(bs, bins=40, density=True)
    plt.axvline(0, color='k', linestyle='--')
    plt.axvline(np.mean(bs), color='r')
    plt.title(f"Bootstrap intercept (boundary={boundary})")
    plt.savefig(f"test3c_bootstrap_intercept_{boundary}.png")
    plt.close()

# ============================================================
# TEST 4 — IPR sweep
# ============================================================

def inverse_participation_ratio(v):
    v2 = v**2
    return np.sum(v2**2) / (np.sum(v2)**2)

def test_criticality_ipr(N=12, deltas=(1,1.5,2,3,4), modes=(1,2,3), boundary="open"):
    print(f"[TEST 4] Localization (IPR) across anisotropy Δ (boundary={boundary})")
    results = {}
    for Delta in deltas:
        evals, vecs, _ = laplacian_spectrum(N, Delta=Delta, boundary=boundary)
        iprs = [inverse_participation_ratio(vecs[:,k]) for k in modes]
        results[Delta] = iprs
        print(f"  Δ={Delta}: {iprs}")

    plt.figure()
    for i,k in enumerate(modes):
        plt.plot(deltas, [results[d][i] for d in deltas], 'o-', label=f"k={k}")
    plt.axhline(1/N, color='gray', linestyle=':')
    plt.title(f"Localization vs Δ (boundary={boundary})")
    plt.xlabel("Δ")
    plt.ylabel("IPR")
    plt.legend()
    plt.savefig(f"test4_criticality_ipr_{boundary}.png")
    plt.close()

# ============================================================
# TEST 5 — λ₁ vs Luttinger K
# ============================================================

def test_lambda1_vs_K(N=12, deltas=np.linspace(-0.9,1.0,11), boundary="open"):
    lam1 = []
    Kvals = []
    for Delta in deltas:
        evals,_,_ = laplacian_spectrum(N, Delta=Delta, boundary=boundary)
        lam1.append(evals[1])
        Kvals.append(luttinger_K(Delta))

    lam1 = np.array(lam1)
    Kvals = np.array(Kvals)

    r = np.corrcoef(lam1, Kvals)[0,1]
    print(f"[TEST 5] Correlation between λ₁ and Luttinger parameter K: r={r:.4f}")

# ============================================================
# TEST 6 — MI decay exponent vs theory
# ============================================================

def test_MI_decay_vs_theory(N=12, deltas=np.linspace(-0.9,1.0,11), boundary="open"):
    print(f"[TEST 6] Mutual‑information decay exponent vs Luttinger prediction (boundary={boundary})")

    predicted = []
    measured = []

    for Delta in deltas:
        K = luttinger_K(Delta)
        alpha = -2 * min(2*K, 1/(2*K))
        predicted.append(alpha)

        evals, vecs, I = laplacian_spectrum(N, Delta=Delta, boundary=boundary)
        r = np.arange(1, N)
        Ivals = np.array([I[i, i+r_i] for r_i in r for i in range(N-r_i)])
        Ivals = Ivals[Ivals > 0]
        log_r = np.log(np.repeat(r, N-r))
        log_I = np.log(Ivals)
        slope, _ = np.polyfit(log_r, log_I, 1)
        measured.append(slope)

        print(f"  Δ={Delta:+.2f}  predicted={alpha:.3f}  measured={slope:.3f}")

    predicted = np.array(predicted)
    measured = np.array(measured)
    r = np.corrcoef(predicted, measured)[0,1]
    print(f"[TEST 6] Correlation between predicted and measured exponents: r={r:.4f}")

    plt.figure()
    plt.plot(predicted, measured, 'o')
    plt.plot(predicted, predicted, '--')
    plt.xlabel("Predicted exponent")
    plt.ylabel("Measured exponent")
    plt.title(f"MI decay exponent (boundary={boundary})")
    plt.savefig(f"test6_MI_decay_vs_theory_{boundary}.png")
    plt.close()

# ============================================================
# TEST 8 — Synthetic periodic ring benchmark
# ============================================================

def test_synthetic_ring(N=12, boundary="open"):
    evals, vecs, _ = laplacian_spectrum(N, Delta=1.0, boundary=boundary)

    # Ideal periodic ring Laplacian eigenvalues
    k = np.arange(N)
    lambda_ring = 2 * (1 - np.cos(2 * np.pi * k / N))

    # Overlaps with ideal ring eigenmodes
    overlaps = [
        abs(np.dot(vecs[:, i],
                   np.cos(2 * np.pi * i * np.arange(N) / N)))
        for i in range(4)
    ]

    print(f"[TEST 8] Synthetic periodic‑ring benchmark (boundary={boundary})")
    for i in range(4):
        print(
            f"  k={i}: λ_num={evals[i]:.4f}, "
            f"λ_ring={lambda_ring[i]:.4f}, "
            f"overlap={overlaps[i]:.4f}"
        )

    plt.figure()
    plt.plot(k, evals, 'o-', label="MI-derived Laplacian")
    plt.plot(k, lambda_ring, 's--', label="Synthetic Ring")
    plt.legend()
    plt.title(f"Spectrum comparison (boundary={boundary})")
    plt.xlabel("Mode index k")
    plt.ylabel("Eigenvalue")

    # Filename unchanged for LaTeX compatibility
    plt.savefig(f"test8_synthetic_neumann_spectrum_{boundary}.png")
    plt.close()

# ============================================================
# MAIN RUNNER — runs ALL tests and outputs ALL PNGs
# ============================================================

if __name__ == "__main__":
    print("\n=== RUNNING ALL TESTS ===\n")

    for boundary in ("open", "periodic"):
        print(f"\n--- Boundary = {boundary} ---\n")

        test_zero_mode_and_boundary(N=12, boundary=boundary)
        test_dispersion_model(N=12, boundary=boundary)
        test_pooled_dispersion(boundary=boundary)
        test_finite_size_scaling(boundary=boundary)
        test_finite_size_models(boundary=boundary)
        test_finite_size_bootstrap(boundary=boundary)
        test_criticality_ipr(boundary=boundary)
        test_lambda1_vs_K(boundary=boundary)
        test_MI_decay_vs_theory(boundary=boundary)
        test_synthetic_ring(boundary=boundary)

    print("\n=== ALL TESTS COMPLETE ===\n")
