import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from pathlib import Path
import time

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
# Estado inicial |down ... down>
# =========================
DOWN = np.array([0, 1], dtype=complex)


def initial_down_state(N):
    """
    Construye el estado |down down ... down> para N espines.
    """
    psi0 = DOWN
    for _ in range(N - 1):
        psi0 = np.kron(psi0, DOWN)
    return psi0


# =========================
# Evolución temporal
# =========================
def evolve_state(H, psi0, dt, n_steps):
    """
    Evoluciona el estado psi0 usando U = exp(-i H dt)
    y retorna:
    - tiempos
    - lista de estados
    - probabilidad de retorno al estado inicial
    """
    U = expm(-1j * H * dt)

    psi = psi0.copy()
    states = [psi.copy()]
    probs = [np.abs(np.vdot(psi0, psi))**2]
    times = [0.0]

    for k in range(1, n_steps + 1):
        psi = U @ psi
        states.append(psi.copy())
        probs.append(np.abs(np.vdot(psi0, psi))**2)
        times.append(k * dt)

    return np.array(times), states, np.array(probs)


# =========================
# Simulación para distintos B/J
# =========================
def simulate_return_probability(N, J, B_values, dt=0.05, t_max=10.0):
    """
    Simula p(t)=|<psi(t)|psi(0)>|^2 para distintos valores de B.
    """
    psi0 = initial_down_state(N)
    n_steps = int(t_max / dt)

    results = {}

    for B in B_values:
        H = build_ising_hamiltonian(N, J, B)
        times, _, probs = evolve_state(H, psi0, dt, n_steps)
        results[B] = (times, probs)

    return results


def plot_return_probability(results, J, output_dir):
    """
    Grafica la probabilidad de retorno para cada valor de B.
    """
    fig = plt.figure(figsize=(8, 5))

    for B, (times, probs) in results.items():
        ratio = B / J
        plt.plot(times, probs, label=f"B/J = {ratio:.2f}")

    plt.xlabel("t")
    plt.ylabel(r"$p(t)=|\langle \Psi(t)|\Psi(0)\rangle|^2$")
    plt.title("Probabilidad de retorno al estado inicial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(
        output_dir / "probabilidad_retorno.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

def benchmark_hamiltonian(N_values, J, B, n_repeats=5):
    """
    Mide el tiempo promedio de:
    - construcción del Hamiltoniano
    - diagonalización del Hamiltoniano

    para varios tamaños N.
    """
    build_times = []
    diag_times = []

    for N in N_values:
        build_samples = []
        diag_samples = []

        for _ in range(n_repeats):
            # Tiempo de construcción de H
            t0 = time.perf_counter()
            H = build_ising_hamiltonian(N, J, B)
            t1 = time.perf_counter()

            # Tiempo de diagonalización de H
            t2 = time.perf_counter()
            np.linalg.eigh(H)
            t3 = time.perf_counter()

            build_samples.append(t1 - t0)
            diag_samples.append(t3 - t2)

        build_times.append(np.mean(build_samples))
        diag_times.append(np.mean(diag_samples))

    return np.array(build_times), np.array(diag_times)


def plot_benchmark(N_values, build_times, diag_times, output_dir):
    """
    Gráfico lineal de tiempos vs N.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(N_values, build_times, "o-", label="Construcción de H")
    ax.plot(N_values, diag_times, "s-", label="Diagonalización de H")

    ax.set_xlabel("N")
    ax.set_ylabel("Tiempo promedio [s]")
    ax.set_title("Tiempo de ejecución vs número de espines")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_tiempos.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_benchmark_loglog(N_values, build_times, diag_times, output_dir):
    """
    Gráfico log-log de tiempos vs N.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(N_values, build_times, "o-", label="Construcción de H")
    ax.loglog(N_values, diag_times, "s-", label="Diagonalización de H")

    ax.set_xlabel("N")
    ax.set_ylabel("Tiempo promedio [s]")
    ax.set_title("Tiempo de ejecución vs N (escala log-log)")
    ax.legend()
    ax.grid(True, which="both")

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_tiempos_loglog.png", dpi=300, bbox_inches="tight")
    plt.show()


def fit_exponential_model(N_values, times):
    """
    Ajusta un modelo exponencial:
        t(N) ≈ exp(a*N + b)
    equivalente a:
        log t = a*N + b
    """
    a, b = np.polyfit(N_values, np.log(times), 1)
    return a, b


def estimate_times_exponential(a, b, N_targets):
    """
    Estima tiempos para nuevos valores de N usando
    t(N) = exp(a*N + b)
    """
    estimates = {}
    for N in N_targets:
        estimates[N] = np.exp(a * N + b)
    return estimates

def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Prueba básica del inciso (b)
    # -------------------------
    N = 3
    J = 1.0
    B = 0.5

    H = build_ising_hamiltonian(N, J, B)

    print(f"N = {N}")
    print(f"Dimensión del espacio de Hilbert = {2**N}")
    print(f"Forma de H = {H.shape}")

    print("\n¿H es hermítico?")
    print(np.allclose(H, H.conj().T))

    print("\nHamiltoniano H:")
    print(H)

    # -------------------------
    # Inciso (c): evolución temporal
    # -------------------------
    print("\n=== Simulación de evolución temporal ===")

    N = 4
    J = 1.0

    # Elegimos tres regímenes: B/J << 1, B/J = 1, B/J >> 1
    B_values = [0.1 * J, 1.0 * J, 5.0 * J]

    results = simulate_return_probability(
        N=N,
        J=J,
        B_values=B_values,
        dt=0.05,
        t_max=10.0
    )

    plot_return_probability(results, J, output_dir)

    print(f"\nGráfico guardado en: {output_dir / 'probabilidad_retorno.png'}")

    # -------------------------
    # Incisos (d), (e), (f), (g), (h)
    # -------------------------
    print("\n=== Benchmark de construcción y diagonalización ===")

    N_values = np.array([4, 5, 6, 7, 8])
    J_bench = 1.0
    B_bench = 1.0

    build_times, diag_times = benchmark_hamiltonian(
        N_values=N_values,
        J=J_bench,
        B=B_bench,
        n_repeats=5
    )

    print("\nTiempos promedio:")
    for N, tb, td in zip(N_values, build_times, diag_times):
        print(f"N = {N}: construcción = {tb:.6e} s, diagonalización = {td:.6e} s")

    plot_benchmark(N_values, build_times, diag_times, output_dir)
    plot_benchmark_loglog(N_values, build_times, diag_times, output_dir)

    print(f"\nGráfico guardado en: {output_dir / 'benchmark_tiempos.png'}")
    print(f"Gráfico guardado en: {output_dir / 'benchmark_tiempos_loglog.png'}")

    # Ajuste exponencial sobre la diagonalización
    a, b = fit_exponential_model(N_values, diag_times)

    N_targets = [20, 50, 100]
    estimates = estimate_times_exponential(a, b, N_targets)

    print("\n=== Estimación exponencial para tiempos de diagonalización ===")
    for N in N_targets:
        print(f"N = {N}: tiempo estimado = {estimates[N]:.6e} s")

    age_universe = 4.3e17  # segundos
    print(f"\nEdad del universo ≈ {age_universe:.6e} s")

    for N in N_targets:
        ratio = estimates[N] / age_universe
        print(f"N = {N}: tiempo / edad del universo = {ratio:.6e}")

if __name__ == "__main__":
    main()