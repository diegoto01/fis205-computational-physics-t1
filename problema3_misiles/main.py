import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# PROBLEMA 3 - MISILES BALÍSTICOS
# ============================================================


# ============================================================
# 1) PARÁMETROS FÍSICOS
# Esto aporta al inciso (a): formulación del modelo matemático
# ============================================================
g = 9.81                       # m/s^2
rho = 1.225                    # kg/m^3
Cd = 0.3
A = 0.05                       # m^2
m = 100.0                      # kg
phi_deg = 45.0                 # latitud del sitio de lanzamiento [deg]
Omega = 7.292e-5               # rad/s

phi = np.radians(phi_deg)

# Sistema local:
# x = Este, y = Norte, z = Arriba
omega_vec = np.array([0.0, Omega * np.cos(phi), Omega * np.sin(phi)])


# ============================================================
# 2) FUNCIONES AUXILIARES DEL MODELO
# Esto también aporta al inciso (a)
# ============================================================
def velocity_components(v0, theta_deg, psi_deg):
    """
    Convierte rapidez inicial y ángulos a componentes cartesianas.

    Parámetros
    ----------
    v0 : float
        Rapidez inicial [m/s]
    theta_deg : float
        Ángulo de elevación respecto al plano horizontal [deg]
    psi_deg : float
        Azimut en el plano horizontal [deg]

    Retorna
    -------
    np.ndarray
        Vector velocidad inicial [vx, vy, vz]
    """
    theta = np.radians(theta_deg)
    psi = np.radians(psi_deg)

    vx = v0 * np.cos(theta) * np.cos(psi)
    vy = v0 * np.cos(theta) * np.sin(psi)
    vz = v0 * np.sin(theta)

    return np.array([vx, vy, vz], dtype=float)


def missile_dynamics(t, state, use_coriolis=True):
    """
    Ecuaciones diferenciales del movimiento del misil.

    state = [x, y, z, vx, vy, vz]

    Modelo considerado:
    - gravedad
    - roce atmosférico cuadrático
    - término de Coriolis (opcional)

    Esto forma parte del inciso (a).
    """
    x, y, z, vx, vy, vz = state

    v = np.array([vx, vy, vz], dtype=float)
    speed = np.linalg.norm(v)

    # Aceleración gravitatoria
    a_grav = np.array([0.0, 0.0, -g])

    # Aceleración por roce cuadrático
    if speed > 0:
        a_drag = -(0.5 * rho * Cd * A / m) * speed * v
    else:
        a_drag = np.zeros(3)

    # Aceleración de Coriolis
    if use_coriolis:
        a_coriolis = -2.0 * np.cross(omega_vec, v)
    else:
        a_coriolis = np.zeros(3)

    a_total = a_grav + a_drag + a_coriolis

    return [vx, vy, vz, a_total[0], a_total[1], a_total[2]]


def hit_ground_event(t, state):
    """
    Detiene la integración cuando el misil vuelve al suelo (z = 0),
    pero solo al bajar.

    Esto no lo exige explícitamente la tarea, pero mejora la simulación.
    """
    return state[2]


hit_ground_event.terminal = True
hit_ground_event.direction = -1


def simulate_missile(
    r0,
    v0,
    theta_deg,
    psi_deg,
    t0=0.0,
    t_max=200.0,
    num_points=2000,
    use_coriolis=True
):
    """
    Simula numéricamente la trayectoria de un misil.

    Esta función se usa directamente para resolver el inciso (b)
    en el caso del misil objetivo.
    """
    v_init = velocity_components(v0, theta_deg, psi_deg)
    y0 = np.concatenate([np.array(r0, dtype=float), v_init])

    t_eval = np.linspace(t0, t_max, num_points)

    sol = solve_ivp(
        lambda t, y: missile_dynamics(t, y, use_coriolis=use_coriolis),
        (t0, t_max),
        y0,
        t_eval=t_eval,
        events=hit_ground_event,
        rtol=1e-8,
        atol=1e-10
    )

    return sol

def minimum_distance_between_missiles(sol_target, sol_interceptor, tau):
    """
    Calcula la distancia mínima entre dos misiles usando una malla temporal común,
    comparando solo desde que existe el interceptor.
    """
    t_start_compare = tau
    t_end_compare = min(sol_target.t[-1], sol_interceptor.t[-1])

    # Si por alguna razón el interceptor cae antes de empezar a compararse
    if t_end_compare <= t_start_compare:
        return np.inf, None, None

    t_compare = np.linspace(t_start_compare, t_end_compare, 3000)

    # Interpolar trayectoria del objetivo
    x1_i = np.interp(t_compare, sol_target.t, sol_target.y[0])
    y1_i = np.interp(t_compare, sol_target.t, sol_target.y[1])
    z1_i = np.interp(t_compare, sol_target.t, sol_target.y[2])

    # Interpolar trayectoria del interceptor
    x2_i = np.interp(t_compare, sol_interceptor.t, sol_interceptor.y[0])
    y2_i = np.interp(t_compare, sol_interceptor.t, sol_interceptor.y[1])
    z2_i = np.interp(t_compare, sol_interceptor.t, sol_interceptor.y[2])

    d = np.sqrt((x1_i - x2_i)**2 + (y1_i - y2_i)**2 + (z1_i - z2_i)**2)

    idx_min = np.argmin(d)
    d_min = d[idx_min]
    t_min = t_compare[idx_min]

    return d_min, t_min, {
        "x1": x1_i[idx_min], "y1": y1_i[idx_min], "z1": z1_i[idx_min],
        "x2": x2_i[idx_min], "y2": y2_i[idx_min], "z2": z2_i[idx_min],
    }

def animate_missiles_3d(sol_target, sol_interceptor, tau, t_collision, output_path):
    """
    Genera una animación 3D de ambos misiles desde sus lanzamientos
    hasta el instante de colisión (o máxima aproximación).
    """

    # Tiempo común desde t=0 hasta el tiempo de colisión
    t_anim = np.linspace(0.0, t_collision, 300)

    # Interpolar misil objetivo en toda la malla
    x1 = np.interp(t_anim, sol_target.t, sol_target.y[0])
    y1 = np.interp(t_anim, sol_target.t, sol_target.y[1])
    z1 = np.interp(t_anim, sol_target.t, sol_target.y[2])

    # Para el interceptor:
    # antes de tau no existe, así que lo dejamos como NaN
    x2 = np.full_like(t_anim, np.nan, dtype=float)
    y2 = np.full_like(t_anim, np.nan, dtype=float)
    z2 = np.full_like(t_anim, np.nan, dtype=float)

    mask = t_anim >= tau
    x2[mask] = np.interp(t_anim[mask], sol_interceptor.t, sol_interceptor.y[0])
    y2[mask] = np.interp(t_anim[mask], sol_interceptor.t, sol_interceptor.y[1])
    z2[mask] = np.interp(t_anim[mask], sol_interceptor.t, sol_interceptor.y[2])

    # Figura
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Límites de ejes
    all_x = np.concatenate([x1, x2[mask]])
    all_y = np.concatenate([y1, y2[mask]])
    all_z = np.concatenate([z1, z2[mask]])

    ax.set_xlim(np.nanmin(all_x) - 500, np.nanmax(all_x) + 500)
    ax.set_ylim(np.nanmin(all_y) - 500, np.nanmax(all_y) + 500)
    ax.set_zlim(0, np.nanmax(all_z) + 500)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Animación 3D de la intercepción")
    ax.set_box_aspect([1, 1, 0.6])

    # Elementos animados
    line1, = ax.plot([], [], [], lw=2, label="Misil objetivo")
    line2, = ax.plot([], [], [], lw=2, label="Interceptor")
    point1, = ax.plot([], [], [], 'o', markersize=8, label="Objetivo")
    point2, = ax.plot([], [], [], 'o', markersize=8, label="Interceptor")

    # Puntos de lanzamiento fijos
    ax.scatter(sol_target.y[0, 0], sol_target.y[1, 0], sol_target.y[2, 0],
               color='green', s=50, label="Lanzamiento objetivo")
    ax.scatter(sol_interceptor.y[0, 0], sol_interceptor.y[1, 0], sol_interceptor.y[2, 0],
               color='black', s=50, label="Lanzamiento interceptor")

    ax.legend()

    def update(frame):
        # Objetivo
        line1.set_data(x1[:frame], y1[:frame])
        line1.set_3d_properties(z1[:frame])
        point1.set_data([x1[frame]], [y1[frame]])
        point1.set_3d_properties([z1[frame]])

        # Interceptor (solo si ya existe)
        if np.isfinite(x2[frame]):
            valid = np.isfinite(x2[:frame])
            line2.set_data(x2[:frame][valid], y2[:frame][valid])
            line2.set_3d_properties(z2[:frame][valid])
            point2.set_data([x2[frame]], [y2[frame]])
            point2.set_3d_properties([z2[frame]])
        else:
            line2.set_data([], [])
            line2.set_3d_properties([])
            point2.set_data([], [])
            point2.set_3d_properties([])

        return line1, line2, point1, point2

    anim = FuncAnimation(fig, update, frames=len(t_anim), interval=40, blit=False)

    anim.save(output_path, writer=PillowWriter(fps=20))
    plt.show()

# ============================================================
# 3) PROGRAMA PRINCIPAL
# Aquí se resuelve el inciso (b):
# "Simule numéricamente la trayectoria del misil objetivo"
#
# Además, se incluye una validación extra comparando con/sin Coriolis.
# Esto no es obligatorio, pero sí puede ayudar a la discusión del inciso (g).
# ============================================================
def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Datos del misil objetivo (entregados por el enunciado)
    # Inciso (b)
    # --------------------------------------------------------

    # Condiciones iniciales del misil objetivo
    r1_0 = [0.0, 0.0, 0.0]
    v1_0 = 500.0
    theta1 = 45.0
    psi1 = 30.0

    # Condiciones iniciales del misil interceptor
    r2_0 = [5000.0, 2000.0, 0.0]
    v2_0 = 600.0
    theta2 = 50.0
    psi2 = 210.0
    tau = 10.0

    # Simulación principal del misil objetivo con Coriolis
    sol1_coriolis = simulate_missile(
        r1_0, v1_0, theta1, psi1,
        t0=0.0, t_max=200.0, num_points=3000,
        use_coriolis=True
    )

    x = sol1_coriolis.y[0]
    y = sol1_coriolis.y[1]
    z = sol1_coriolis.y[2]
    t = sol1_coriolis.t

    # Simulación del misil interceptor con Coriolis

    sol2_coriolis = simulate_missile(
        r2_0, v2_0, theta2, psi2,
        t0=tau, t_max=200.0, num_points=3000,
        use_coriolis=True
    )

    x2 = sol2_coriolis.y[0]
    y2 = sol2_coriolis.y[1]
    z2 = sol2_coriolis.y[2]
    t2 = sol2_coriolis.t
    # --------------------------------------------------------
    # GRÁFICO 1: proyección x-z
    # Esto sí corresponde directamente al inciso (b)
    # --------------------------------------------------------
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(x, z)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Trayectoria del misil objetivo (proyección x-z)")
    plt.grid(True)
    plt.tight_layout()
    fig1.savefig(output_dir / "grafico_1_trayectoria_objetivo_xz.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------------
    # GRÁFICO 2: planta x-y
    # También ayuda a visualizar el inciso (b)
    # OJO: este gráfico NO representa por sí solo el efecto de Coriolis.
    # La coordenada y está dominada por el azimut inicial.
    # --------------------------------------------------------
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trayectoria del misil objetivo (plano horizontal x-y)")
    plt.grid(True)
    plt.tight_layout()
    fig2.savefig(output_dir / "grafico_2_trayectoria_objetivo_xy.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("=== Simulación del misil objetivo ===")
    print(f"Tiempo final simulado: {t[-1]:.2f} s")
    print(f"Posición final: x = {x[-1]:.2f} m, y = {y[-1]:.2f} m, z = {z[-1]:.2f} m")

    # --------------------------------------------------------
    # VALIDACIÓN EXTRA DEL MODELO:
    # comparación con/sin Coriolis
    #
    # Esto NO lo pide explícitamente la tarea, pero sí suma:
    # permite cuantificar el efecto de la rotación terrestre
    # para usarlo luego en la discusión del inciso (g).
    # --------------------------------------------------------
    sol1_no_coriolis = simulate_missile(
        r1_0, v1_0, theta1, psi1,
        t0=0.0, t_max=200.0, num_points=3000,
        use_coriolis=False
    )

    sol2_no_coriolis = simulate_missile(
        r2_0, v2_0, theta2, psi2,
        t0=tau, t_max=200.0, num_points=3000,
        use_coriolis=False
    )

    # --------------------------------------------------------
    # COMPARACIÓN EN INTERVALO TEMPORAL COMÚN
    #
    # Esto corrige el artefacto numérico que aparece al final
    # cuando ambas trayectorias aterrizan en tiempos ligeramente distintos.
    # --------------------------------------------------------
    t_end_common = min(sol1_coriolis.t[-1], sol1_no_coriolis.t[-1])
    t_common = np.linspace(0.0, t_end_common, 2000)

    # Interpolar ambas soluciones sobre la misma malla temporal
    x_with_i = np.interp(t_common, sol1_coriolis.t, sol1_coriolis.y[0])
    y_with_i = np.interp(t_common, sol1_coriolis.t, sol1_coriolis.y[1])
    z_with_i = np.interp(t_common, sol1_coriolis.t, sol1_coriolis.y[2])

    x_without_i = np.interp(t_common, sol1_no_coriolis.t, sol1_no_coriolis.y[0])
    y_without_i = np.interp(t_common, sol1_no_coriolis.t, sol1_no_coriolis.y[1])
    z_without_i = np.interp(t_common, sol1_no_coriolis.t, sol1_no_coriolis.y[2])

    dx = x_with_i - x_without_i
    dy = y_with_i - y_without_i
    dz = z_with_i - z_without_i

    # --------------------------------------------------------
    # GRÁFICO 3: diferencia lateral real debida a Coriolis
    # Esto sí aísla el efecto de rotación terrestre
    # --------------------------------------------------------
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(t_common, dy)
    plt.xlabel("t [s]")
    plt.ylabel("Δy [m]")
    plt.title("Diferencia lateral debida a Coriolis")
    plt.grid(True)
    plt.tight_layout()
    fig3.savefig(output_dir / "grafico_3_diferencia_lateral_coriolis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n=== Comparación con/sin Coriolis ===")
    print("Diferencia final atribuible a la rotación terrestre:")
    print(f"Δx = {dx[-1]:.4f} m")
    print(f"Δy = {dy[-1]:.4f} m")
    print(f"Δz = {dz[-1]:.4f} m")

    fig4 = plt.figure(figsize=(8,6))

    # misil objetivo
    plt.plot(x, y, label="Objetivo")

    # interceptor
    plt.plot(x2, y2, label="Interceptor")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trayectorias en plano horizontal")
    plt.legend()
    plt.grid()
    fig4.savefig(output_dir / "grafico_4_trayectorias_plano_horizontal.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig5 = plt.figure(figsize=(8,6))

    plt.plot(t, x, label="x objetivo")
    plt.plot(t2, x2, label="x interceptor")

    plt.axvline(tau, linestyle='--', color='k', label="Lanzamiento interceptor")

    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    plt.title("Comparación temporal")
    plt.legend()
    plt.grid()
    fig5.savefig(output_dir / "grafico_5_comparacion_temporal.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGráficos guardados en: {output_dir}")

    # --------------------------------------------------------
    # COMPARACIÓN ENTRE OBJETIVO E INTERCEPTOR
    # Esto prepara los incisos (c), (d) y (e)
    # --------------------------------------------------------
    dtol = 10.0  # tolerancia de colisión [m]

    # Solo tiene sentido comparar desde que existe el interceptor
    t_start_compare = tau
    t_end_compare = min(sol1_coriolis.t[-1], sol2_coriolis.t[-1])

    t_compare = np.linspace(t_start_compare, t_end_compare, 3000)

    # Interpolar misil objetivo en malla común
    x1_i = np.interp(t_compare, sol1_coriolis.t, sol1_coriolis.y[0])
    y1_i = np.interp(t_compare, sol1_coriolis.t, sol1_coriolis.y[1])
    z1_i = np.interp(t_compare, sol1_coriolis.t, sol1_coriolis.y[2])

    # Interpolar misil interceptor en malla común
    x2_i = np.interp(t_compare, sol2_coriolis.t, sol2_coriolis.y[0])
    y2_i = np.interp(t_compare, sol2_coriolis.t, sol2_coriolis.y[1])
    z2_i = np.interp(t_compare, sol2_coriolis.t, sol2_coriolis.y[2])

    # Distancia entre ambos misiles
    d = np.sqrt((x1_i - x2_i)**2 + (y1_i - y2_i)**2 + (z1_i - z2_i)**2)

    # Distancia mínima
    idx_min = np.argmin(d)
    d_min = d[idx_min]
    t_min = t_compare[idx_min]

    print("\n=== Comparación objetivo vs interceptor ===")
    print(f"Distancia mínima = {d_min:.4f} m")
    print(f"Tiempo de mínima distancia = {t_min:.4f} s")

    # Verificar colisión
    collision_indices = np.where(d < dtol)[0]

    if len(collision_indices) > 0:
        idx_col = collision_indices[0]
        t_col = t_compare[idx_col]

        x_col = x1_i[idx_col]
        y_col = y1_i[idx_col]
        z_col = z1_i[idx_col]

        print("Sí hubo colisión.")
        print(f"Tiempo de colisión = {t_col:.4f} s")
        print(f"Punto de colisión aproximado = ({x_col:.2f}, {y_col:.2f}, {z_col:.2f}) m")
    else:
        print("No hubo colisión con estos parámetros.")
        print("Punto de máxima aproximación aproximado:")
        print(f"Objetivo    = ({x1_i[idx_min]:.2f}, {y1_i[idx_min]:.2f}, {z1_i[idx_min]:.2f}) m")
        print(f"Interceptor = ({x2_i[idx_min]:.2f}, {y2_i[idx_min]:.2f}, {z2_i[idx_min]:.2f}) m")

    # --------------------------------------------------------
    # GRÁFICO 6: distancia entre misiles
    # --------------------------------------------------------
    fig6 = plt.figure(figsize=(8, 6))
    plt.plot(t_compare, d, label="Distancia entre misiles")
    plt.axhline(dtol, color='r', linestyle='--', label="Tolerancia de colisión")
    plt.xlabel("t [s]")
    plt.ylabel("d(t) [m]")
    plt.title("Distancia entre misil objetivo e interceptor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig6.savefig(output_dir / "grafico_6_distancia_entre_misiles.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------------
    # BÚSQUEDA GRUESA DE PARÁMETROS DEL INTERCEPTOR
    # Esto sí empieza a atacar el inciso (c)
    # --------------------------------------------------------
    v2_values = np.linspace(500.0, 900.0, 7)     # 500, 566..., 900
    theta2_values = np.linspace(35.0, 75.0, 9)   # 35 a 75 deg
    psi2_values = np.linspace(180.0, 240.0, 13)  # 180 a 240 deg

    best_result = {
        "d_min": np.inf,
        "t_min": None,
        "v2_0": None,
        "theta2": None,
        "psi2": None,
        "points": None,
    }

    print("\n=== Búsqueda gruesa de parámetros del interceptor ===")

    for v2_test in v2_values:
        for theta2_test in theta2_values:
            for psi2_test in psi2_values:
                sol2_test = simulate_missile(
                    r2_0,
                    v2_test,
                    theta2_test,
                    psi2_test,
                    t0=tau,
                    t_max=200.0,
                    num_points=2000,
                    use_coriolis=True
                )

                d_min_test, t_min_test, points_test = minimum_distance_between_missiles(
                    sol1_coriolis, sol2_test, tau
                )

                if d_min_test < best_result["d_min"]:
                    best_result["d_min"] = d_min_test
                    best_result["t_min"] = t_min_test
                    best_result["v2_0"] = v2_test
                    best_result["theta2"] = theta2_test
                    best_result["psi2"] = psi2_test
                    best_result["points"] = points_test

    print(f"Mejor distancia mínima encontrada = {best_result['d_min']:.4f} m")
    print(f"Tiempo correspondiente = {best_result['t_min']:.4f} s")
    print(f"Mejor v2,0 = {best_result['v2_0']:.4f} m/s")
    print(f"Mejor theta2 = {best_result['theta2']:.4f} deg")
    print(f"Mejor psi2 = {best_result['psi2']:.4f} deg")

    if best_result["points"] is not None:
        p = best_result["points"]
        print("Puntos en la máxima aproximación de la mejor combinación:")
        print(f"Objetivo    = ({p['x1']:.2f}, {p['y1']:.2f}, {p['z1']:.2f}) m")
        print(f"Interceptor = ({p['x2']:.2f}, {p['y2']:.2f}, {p['z2']:.2f}) m")

    # --------------------------------------------------------
    # BÚSQUEDA FINA ALREDEDOR DE LA MEJOR COMBINACIÓN GRUESA
    # Esto refina el inciso (c)
    # --------------------------------------------------------
    v2_center = best_result["v2_0"]
    theta2_center = best_result["theta2"]
    psi2_center = best_result["psi2"]

    v2_values_fine = np.linspace(v2_center - 60.0, v2_center + 60.0, 13)
    theta2_values_fine = np.linspace(theta2_center - 6.0, theta2_center + 6.0, 13)
    psi2_values_fine = np.linspace(psi2_center - 6.0, psi2_center + 6.0, 13)

    best_fine_result = {
        "d_min": np.inf,
        "t_min": None,
        "v2_0": None,
        "theta2": None,
        "psi2": None,
        "points": None,
    }

    print("\n=== Búsqueda fina de parámetros del interceptor ===")

    for v2_test in v2_values_fine:
        for theta2_test in theta2_values_fine:
            for psi2_test in psi2_values_fine:
                sol2_test = simulate_missile(
                    r2_0,
                    v2_test,
                    theta2_test,
                    psi2_test,
                    t0=tau,
                    t_max=200.0,
                    num_points=2500,
                    use_coriolis=True
                )

                d_min_test, t_min_test, points_test = minimum_distance_between_missiles(
                    sol1_coriolis, sol2_test, tau
                )

                if d_min_test < best_fine_result["d_min"]:
                    best_fine_result["d_min"] = d_min_test
                    best_fine_result["t_min"] = t_min_test
                    best_fine_result["v2_0"] = v2_test
                    best_fine_result["theta2"] = theta2_test
                    best_fine_result["psi2"] = psi2_test
                    best_fine_result["points"] = points_test

    print(f"Mejor distancia mínima fina = {best_fine_result['d_min']:.4f} m")
    print(f"Tiempo correspondiente = {best_fine_result['t_min']:.4f} s")
    print(f"Mejor v2,0 fino = {best_fine_result['v2_0']:.4f} m/s")
    print(f"Mejor theta2 fino = {best_fine_result['theta2']:.4f} deg")
    print(f"Mejor psi2 fino = {best_fine_result['psi2']:.4f} deg")

    if best_fine_result["points"] is not None:
        p = best_fine_result["points"]
        print("Puntos en la máxima aproximación de la búsqueda fina:")
        print(f"Objetivo    = ({p['x1']:.2f}, {p['y1']:.2f}, {p['z1']:.2f}) m")
        print(f"Interceptor = ({p['x2']:.2f}, {p['y2']:.2f}, {p['z2']:.2f}) m")

    if best_fine_result["d_min"] < dtol:
        print("\n>>> Con la búsqueda fina SÍ se encontró colisión dentro de la tolerancia.")
    else:
        print("\n>>> Con la búsqueda fina TODAVÍA no se alcanza la tolerancia de colisión.")

    # --------------------------------------------------------
    # INCISO (e): Gráfico final con ambas trayectorias y colisión
    # --------------------------------------------------------

    # Usamos los mejores parámetros encontrados
    v2_best = best_fine_result["v2_0"]
    theta2_best = best_fine_result["theta2"]
    psi2_best = best_fine_result["psi2"]

    # Re-simular interceptor con parámetros óptimos
    sol2_best = simulate_missile(
        r2_0,
        v2_best,
        theta2_best,
        psi2_best,
        t0=tau,
        t_max=200.0,
        num_points=3000,
        use_coriolis=True
    )

    # Trayectoria objetivo
    x1 = sol1_coriolis.y[0]
    y1 = sol1_coriolis.y[1]
    z1 = sol1_coriolis.y[2]

    # Trayectoria interceptor óptimo
    x2 = sol2_best.y[0]
    y2 = sol2_best.y[1]
    z2 = sol2_best.y[2]

    # Punto de colisión (aproximado)
    p = best_fine_result["points"]

    # Crear gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Trayectorias
    ax.plot(x1, y1, z1, label="Misil objetivo")
    ax.plot(x2, y2, z2, label="Interceptor")

    # Punto de colisión
    ax.scatter(
        p["x1"], p["y1"], p["z1"],
        color='red', s=80, label="Colisión"
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Trayectorias de misiles y punto de colisión")

    ax.legend()
    plt.tight_layout()

    fig.savefig(output_dir / "grafico_7_colision_3D.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------------
    # INCISO (f): animación de ambos misiles hasta la colisión
    # --------------------------------------------------------
    if best_fine_result["d_min"] < dtol:
        t_collision = best_fine_result["t_min"]
    else:
        t_collision = best_fine_result["t_min"]  # máxima aproximación si no hubiera colisión

    animate_missiles_3d(
        sol1_coriolis,
        sol2_best,
        tau,
        t_collision,
        output_dir / "animacion_intercepcion.gif"
    )

    print("Animación guardada en:", output_dir / "animacion_intercepcion.gif")

if __name__ == "__main__":
    main()