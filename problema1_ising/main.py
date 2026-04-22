import numpy as np


# =========================
# Matrices básicas
# =========================
I2 = np.eye(2, dtype=complex)

SIGMA_X = np.array([[0, 1],
                    [1, 0]], dtype=complex)

SIGMA_Z = np.array([[1, 0],
                    [0, -1]], dtype=complex)


# =========================
# Utilidades tensoriales
# =========================
def kron_n(operators):
    """
    Calcula el producto tensorial de una lista de operadores.
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


def one_site_operator(op, site, N):
    """
    Construye un operador que actúa como 'op' en el sitio 'site'
    y como identidad en los demás sitios.

    Parámetros:
    - op: matriz 2x2
    - site: índice del sitio (0, 1, ..., N-1)
    - N: número total de espines
    """
    ops = [I2] * N
    ops[site] = op
    return kron_n(ops)


def two_site_operator(op1, site1, op2, site2, N):
    """
    Construye un operador que actúa como 'op1' en site1,
    como 'op2' en site2, e identidad en el resto.
    """
    ops = [I2] * N
    ops[site1] = op1
    ops[site2] = op2
    return kron_n(ops)


# =========================
# Hamiltoniano de Ising transversal
# =========================
def build_ising_hamiltonian(N, J, B):
    """
    Construye el Hamiltoniano del modelo de Ising transversal:

        H = J * sum_{i=0}^{N-2} sigma_x(i) sigma_x(i+1)
            + B * sum_{i=0}^{N-1} sigma_z(i)

    usando condiciones de borde abiertas.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    # Término de interacción J * sum sigma_x_i sigma_x_{i+1}
    for i in range(N - 1):
        H += J * two_site_operator(SIGMA_X, i, SIGMA_X, i + 1, N)

    # Término de campo transversal B * sum sigma_z_i
    for i in range(N):
        H += B * one_site_operator(SIGMA_Z, i, N)

    return H


# =========================
# Prueba básica
# =========================
def main():
    N = 3
    J = 1.0
    B = 0.5

    H = build_ising_hamiltonian(N, J, B)

    print(f"N = {N}")
    print(f"Dimensión del espacio de Hilbert = {2**N}")
    print(f"Forma de H = {H.shape}")
    print("\nHamiltoniano H:")
    print(H)


if __name__ == "__main__":
    main()