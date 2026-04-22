import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def generar_senal(N, fs, f1, f2): #inciso a

    # ¿Qué son las variables N, fs, f1 y f2?
    # N: Número de muestras de la señal, es como decir, "cuanto dura la grabación" o "cuántos datos vamos a tener".
    # fs: Frecuencia de muestreo en Hz, que determina cuántas muestras se toman o graban por segundo.
    # f1: Frecuencia de la primera componente sinusoidal en Hz.
    # f2: Frecuencia de la segunda componente sinusoidal en Hz.

    t = np.arange(N) / fs # Δt = n * Ts, donde Ts = 1/fs es el período de muestreo.
    x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) # la señal discreta definida en el enunciado

    return t, x


def dft_directa(x): #inciso b

    N = len(x) # tamaño del vector de entrada
    X = np.zeros(N, dtype=complex) # vector de salida para almacenar los coeficientes de la DFT, inicializado con ceros y tipo complejo

    for k in range(N): # para cada frecuencia k, se calcula la suma de la DFT
        suma = 0.0j # variable temporal para acumular la suma de la DFT, inicializada como un número complejo con parte imaginaria cero
        for n in range(N): # para cada muestra n, se multiplica la muestra x[n] por el factor de la DFT, que es una exponencial compleja que depende de k y n, y se acumula en suma
            suma += x[n] * np.exp(-2j * np.pi * k * n / N) # la fórmula de la DFT
        X[k] = suma # se asigna el resultado de la suma al coeficiente X[k] de la DFT

    return X

def medir_tiempo_duncion(funcion, *args, repeticiones=3): # Mide el tiempo promedio de ejecución de una función

    tiempos = []
    for _ in range(repeticiones):
        start_time = time.perf_counter() # tiempo de inicio
        funcion(*args) # se ejecuta la función con los argumentos proporcionados
        end_time = time.perf_counter() # tiempo de fin
        tiempos.append(end_time - start_time) # se calcula el tiempo transcurrido y se almacena en la lista

    return np.mean(tiempos) # se devuelve el tiempo promedio de ejecución

def medir_tiempos (N_values, fs,f1, f2,repeticiones=3): # Mide los tiempos de DFT directa y FFT para diferentes tamaños de señal
    tiempos_dft = []
    tiempos_fft = []

    for N in N_values:
        _, x = generar_senal(N, fs, f1, f2) # se genera la señal para el tamaño N

        tiempo_dft = medir_tiempo_duncion(dft_directa, x, repeticiones=repeticiones) # se mide el tiempo de ejecución de la DFT directa
        tiempos_dft.append(tiempo_dft) # se almacena el tiempo en la lista

        tiempo_fft = medir_tiempo_duncion(np.fft.fft, x, repeticiones=repeticiones) # se mide el tiempo de ejecución de la FFT de numpy
        tiempos_fft.append(tiempo_fft) # se almacena el tiempo en la lista

    return np.array(tiempos_dft), np.array(tiempos_fft) # se devuelven las listas de tiempos para DFT directa y FFT
    
def estimar_pendiente_loglog(N_values, tiempos): #Ajusta una recta en escala log-log: log(t) = m * log(N) + b
    logN = np.log(N_values) # se calcula el logaritmo de los tamaños de señal
    logT = np.log(tiempos) # se calcula el logaritmo de los tiempos
    pendiente, intercepto = np.polyfit(logN, logT, 1) # se ajusta una recta a los datos en escala log-log
    return pendiente, intercepto # se devuelve la pendiente y el intercepto de la recta ajustada

def encuentra_N (N_values, tiempos_dft, tiempos_fft): # Encuentra el tamaño N a partir del cual la FFT es más rápida que la DFT directa
    razones = tiempos_dft / tiempos_fft # se calcula la razón entre los tiempos de DFT directa y FFT
    for N, razon in zip(N_values, razones): # se itera sobre los tamaños
        if razon >= 100:
            return N, razon
    return None, None # si no se encuentra un tamaño N que cumpla la condición, se devuelve None    
    

def main():

    ## PARTE 1 : Señal y espectro.

    # valores de ejemplo para la generación de la señal
    N = 256
    fs = 256.0
    f1 = 20.0
    f2 = 50.0

    # generamos señal
    t, x = generar_senal(N, fs, f1, f2)
 
    # calculamos DFT directa
    X_dft = dft_directa(x)

    # FFT de numpy
    X_fft = np.fft.fft(x)

    # frecuencias asociadas
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Grafico señal temporal
    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("x(t)")
    plt.title("Señal compuesta por dos frecuencias")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "signal_time_domain.png", dpi=300)

    # Grafico espectro
    plt.figure(figsize=(10, 4))
    plt.stem(freqs[:N // 2], np.abs(X_dft[:N // 2]), basefmt=" ") #inciso c 
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel(r"$|X_k|$")
    plt.title("Espectro obtenido con DFT directa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "spectrum_dft.png", dpi=300)

    # Gráfico comparativo entre DFT directa y FFT
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:N // 2], np.abs(X_dft[:N // 2]), label="DFT directa")
    plt.plot(freqs[:N // 2], np.abs(X_fft[:N // 2]), "--", label="FFT numpy")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.title("Comparación entre DFT directa y FFT")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "comparison_dft_fft.png", dpi=300)

    ## PARTE 2 : Comparación de tiempos de ejecución.
    N_values = np.array([10**2, 10**3, 10**4, 10**5]) # tamaños de señal a evaluar, opté por estos valores dado que con [10**2,10**3,10**4,10**5], la DFT se vuelve muy lenta.
    repeticiones = 3

    tiempos_dft, tiempos_fft = medir_tiempos(N_values, fs, f1, f2, repeticiones)

    # Gráfico tiempo vs N
    plt.figure(figsize=(10, 4))
    plt.plot(N_values, tiempos_dft, "o-", label="DFT directa")
    plt.plot(N_values, tiempos_fft, "s-", label="FFT numpy")
    plt.xlabel("N")
    plt.ylabel("Tiempo [s]")
    plt.title("Tiempo de ejecución vs tamaño de señal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "time_vs_N.png", dpi=300)

    # Gráfico log-log
    plt.figure(figsize=(10, 4))
    plt.loglog(N_values, tiempos_dft, "o-", label="DFT directa")
    plt.loglog(N_values, tiempos_fft, "s-", label="FFT numpy")
    plt.xlabel("N")
    plt.ylabel("Tiempo [s]")
    plt.title("Tiempo de ejecución en escala log-log")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "loglog_time_vs_N.png", dpi=300)

     # Estimar pendientes en log-log
    pendiente_dft, _ = estimar_pendiente_loglog(N_values, tiempos_dft)
    pendiente_fft, _ = estimar_pendiente_loglog(N_values, tiempos_fft)

    print("\n--- Escalamiento experimental ---")
    print(f"Pendiente DFT  ≈ {pendiente_dft:.3f}")
    print(f"Pendiente FFT  ≈ {pendiente_fft:.3f}")

    # Encontrar el tamaño N a partir del cual la FFT es 100 veces más rápida que la DFT directa
    N_encontrado, razon_encontrada = encuentra_N(N_values, tiempos_dft, tiempos_fft)
    if N_encontrado is not None:
        print(f"\n--- Tamaño de señal ---")
        print(f"Tamaño N a partir del cual FFT es más rápida: {N_encontrado}")
        print(f"Razón (DFT/FFT): {razon_encontrada:.2f}")

    plt.show()


if __name__ == "__main__": # Ejecuta "main()" esto solo si corro este archivo directamente
    main()

'''
Inciso i: El algoritmo FFT es fundamental porque transforma un problema 
que originalmente tiene una complejidad de O(N^2) (como la DFT directa) 
a una complejidad de O(N log N). Esto significa que para señales grandes,
la FFT es mucho más eficiente y rápida que la DFT directa, lo que permite
procesar grandes cantidades de datos en aplicaciones como procesamiento de
señales, análisis de audio, imágenes, etc. La FFT ha revolucionado el campo
del procesamiento digital de señales al hacer posible el análisis espectral
en tiempo real y con alta resolución.
'''