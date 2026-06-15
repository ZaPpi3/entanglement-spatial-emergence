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
    """Construct isotropic Heisenberg Hamiltonian for N spins."""
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H = 0
    for i in range(N - 1):
        H += J * (op_on_site(sx, i, N) * op_on_site(sx, i+1, N) +
                  op_on_site(sy, i, N) * op_on_site(sy, i+1, N) +
                  op_on_site(sz, i, N) * op_on_site(sz, i+1, N))
    return H

def mutual_information_matrix(psi0, N):
    """Compute NxN mutual-information matrix."""
    rho = psi0 * psi0.dag()
    I = np.zeros((N, N))

    for i, j in product(range(N), range(N)):
        if i == j:
            continue
        rho_i = rho.ptrace(i)
        rho_j = rho.ptrace(j)
        rho_ij = rho.ptrace([i, j])
        Si = entropy_vn(rho_i)
        Sj = entropy_vn(rho_j)
        Sij = entropy_vn(rho_ij)
        I[i, j] = Si + Sj - Sij

    return I

def laplacian_from_MI(I):
    """Construct normalized adjacency and Laplacian."""
    A = I / np.max(I)
    np.fill_diagonal(A, 0)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L

# ============================================================
# 1. Generate low-lying spectrum plot (spectrum.png)
# ============================================================

def generate_spectrum_plot(N=12):
    H = build_heisenberg_hamiltonian(N)
    evals, evecs = H.eigenstates()
    psi0 = evecs[0]

    I = mutual_information_matrix(psi0, N)
    L = laplacian_from_MI(I)

    evals_L, _ = eigh(L)
    evals_L = np.real(evals_L)

    # Low-lying modes
    k_vals = np.arange(1, 5)
    low = evals_L[1:5]

    # Quadratic fit
    coeffs = np.polyfit(k_vals, low, 2)
    fit_k = np.linspace(1, 4, 200)
    fit_lambda = np.polyval(coeffs, fit_k)

    plt.figure(figsize=(7,5))
    plt.plot(k_vals, low, 'o', label='Numerical eigenvalues')
    plt.plot(fit_k, fit_lambda, '-', label='Quadratic fit')
    plt.xlabel("Mode number k")
    plt.ylabel("Eigenvalue λ_k")
    plt.title("Quadratic scaling of low-lying Laplacian eigenvalues")
    plt.grid(True)
    plt.legend()
    plt.savefig("spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved spectrum.png")

# ============================================================
# 2. Finite-size scaling plot (scaling_lambda1.png)
# ============================================================

def first_nonzero_eigenvalue(N):
    H = build_heisenberg_hamiltonian(N)
    evals, evecs = H.eigenstates()
    psi0 = evecs[0]

    I = mutual_information_matrix(psi0, N)
    L = laplacian_from_MI(I)

    evals_L, _ = eigh(L)
    return np.real(evals_L[1])

def generate_scaling_plot():
    Ns = [6, 8, 10, 12]
    lam1 = [first_nonzero_eigenvalue(N) for N in Ns]
    invN2 = [1.0 / (N**2) for N in Ns]

    coeffs = np.polyfit(invN2, lam1, 1)
    fit_x = np.linspace(min(invN2), max(invN2), 200)
    fit_y = np.polyval(coeffs, fit_x)

    plt.figure(figsize=(7,5))
    plt.plot(invN2, lam1, 'o', label=r'Numerical $\lambda_1$')
    plt.plot(fit_x, fit_y, '-', label='Linear fit')
    plt.xlabel(r'$1/N^2$')
    plt.ylabel(r'$\lambda_1$')
    plt.title("Finite-size scaling of lowest eigenvalue")
    plt.grid(True)
    plt.legend()
    plt.savefig("scaling_lambda1.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved scaling_lambda1.png")

# ============================================================
# 3. Eigenmode plot (eigenmodes.png)
# ============================================================

def generate_eigenmode_plot(N=12):
    H = build_heisenberg_hamiltonian(N)
    evals, evecs = H.eigenstates()
    psi0 = evecs[0]

    I = mutual_information_matrix(psi0, N)
    L = laplacian_from_MI(I)

    evals_L, vecs_L = eigh(L)
    vecs_L = np.real(vecs_L)

    x = np.arange(N)

    plt.figure(figsize=(7,5))
    for k in [1, 2, 3]:
        plt.plot(x, vecs_L[:, k], 'o-', label=f"Mode k={k}")
    plt.xlabel("Site index")
    plt.ylabel("Eigenvector amplitude")
    plt.title("Low-lying Laplacian eigenmodes")
    plt.grid(True)
    plt.legend()
    plt.savefig("eigenmodes.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved eigenmodes.png")

# ============================================================
# 4. Mutual-information heatmap (mi_heatmap.png)
# ============================================================

def generate_mi_heatmap(N=12):
    H = build_heisenberg_hamiltonian(N)
    evals, evecs = H.eigenstates()
    psi0 = evecs[0]

    I = mutual_information_matrix(psi0, N)

    plt.figure(figsize=(6,5))
    plt.imshow(I, cmap='viridis', origin='lower')
    plt.colorbar(label="Mutual information")
    plt.title("Mutual-information matrix")
    plt.xlabel("Site j")
    plt.ylabel("Site i")
    plt.savefig("mi_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved mi_heatmap.png")

# ============================================================
# Run all figure generators
# ============================================================

if __name__ == "__main__":
    generate_spectrum_plot()
    generate_scaling_plot()
    generate_eigenmode_plot()
    generate_mi_heatmap()

    print("\nAll figures generated successfully.")
