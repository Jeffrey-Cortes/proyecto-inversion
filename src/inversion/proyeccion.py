"""
Proyección hacia adelante (Monte Carlo / block bootstrap) de estrategias
de DCA. No es una predicción puntual — nadie sabe qué va a hacer el
mercado — es una forma honesta de comunicar el RANGO de resultados
posibles si el futuro se parece, estadísticamente, a los retornos diarios
históricos usados para generar los bloques.

Motor vectorizado con numpy (no pandas), a propósito: se necesita simular
miles de trayectorias, y el motor de `backtest.simular_dca` (con
`df.iterrows()`) es demasiado lento para eso. Aquí solo se simula la
variante "solo compra" (`vender_en_señal_negativa=False`), que es la que
quedó recomendada tras el Paso 4.
"""
import numpy as np
import pandas as pd


def bootstrap_retornos(retornos: np.ndarray, dias: int, n_simulaciones: int,
                        tamaño_bloque: int = 21, rng: np.random.Generator | None = None) -> np.ndarray:
    """Block bootstrap: muestrea bloques de `tamaño_bloque` días
    consecutivos de `retornos` (con reemplazo) hasta juntar `dias` días
    simulados, por cada una de `n_simulaciones` trayectorias. Preserva
    algo de la autocorrelación/clustering de volatilidad real de los
    datos — un bootstrap día-a-día independiente la destruiría."""
    rng = rng or np.random.default_rng()
    n_bloques = int(np.ceil(dias / tamaño_bloque))
    n_disponibles = len(retornos) - tamaño_bloque
    inicios = rng.integers(0, n_disponibles, size=(n_simulaciones, n_bloques))
    offsets = np.arange(tamaño_bloque)
    idx = inicios[:, :, None] + offsets[None, None, :]
    matriz = retornos[idx].reshape(n_simulaciones, -1)[:, :dias]
    return matriz


def construir_señal_sma(precios_con_semilla: np.ndarray, dias_semilla: int,
                         corta: int = 50, larga: int = 200) -> np.ndarray:
    """precios_con_semilla: (n_simulaciones, dias_semilla + dias_futuro),
    con los primeros `dias_semilla` días siendo precios REALES (historia
    reciente) para que SMA(200) ya esté completa desde el primer día
    simulado. Devuelve la señal 0/1 recortada a solo los días futuros."""
    df = pd.DataFrame(precios_con_semilla.T)
    sma_corta = df.rolling(corta).mean().to_numpy().T
    sma_larga = df.rolling(larga).mean().to_numpy().T
    señal = (sma_corta > sma_larga).astype(float)
    return señal[:, dias_semilla:]


def simular_dca_montecarlo(precios_futuro: np.ndarray, señal_futuro: np.ndarray,
                            aporte: float, dias_por_aporte: int,
                            costo_bps: float) -> tuple[np.ndarray, np.ndarray]:
    """Corre, en paralelo (vectorizado) para todas las simulaciones, DCA
    simple (siempre invierte) y DCA + SMA (solo compra, nunca vende —
    variante recomendada del Paso 4) mes a mes. Devuelve el valor diario
    del portafolio para ambas: (valor_simple, valor_sma), cada una
    (n_simulaciones, dias)."""
    n_sim, dias = precios_futuro.shape

    unidades_simple = np.zeros(n_sim)
    reserva_simple = np.zeros(n_sim)
    unidades_sma = np.zeros(n_sim)
    reserva_sma = np.zeros(n_sim)
    posicion_sma_ayer = np.zeros(n_sim)  # señal.shift(1): empieza en 0, anti look-ahead

    valor_simple = np.empty((n_sim, dias))
    valor_sma = np.empty((n_sim, dias))

    for t in range(dias):
        precio = precios_futuro[:, t]
        es_dia_aporte = (t % dias_por_aporte == 0)

        if es_dia_aporte:
            reserva_simple += aporte
            reserva_sma += aporte
            costo = reserva_simple * costo_bps / 10_000
            unidades_simple += (reserva_simple - costo) / precio
            reserva_simple[:] = 0.0

        invertir = posicion_sma_ayer == 1
        costo_sma = reserva_sma * costo_bps / 10_000
        unidades_sma = np.where(invertir, unidades_sma + (reserva_sma - costo_sma) / precio, unidades_sma)
        reserva_sma = np.where(invertir, 0.0, reserva_sma)

        valor_simple[:, t] = unidades_simple * precio + reserva_simple
        valor_sma[:, t] = unidades_sma * precio + reserva_sma

        posicion_sma_ayer = señal_futuro[:, t]

    return valor_simple, valor_sma


def percentiles_por_horizonte(valores: np.ndarray, dias_por_año: int,
                               horizontes_años: list[int],
                               percentiles: list[int] = [10, 25, 50, 75, 90]) -> pd.DataFrame:
    """valores: (n_simulaciones, dias). Devuelve una tabla de percentiles
    del valor del portafolio en cada horizonte (años) pedido."""
    filas = []
    for años in horizontes_años:
        dia = min(años * dias_por_año, valores.shape[1]) - 1
        fila = {"Años": años}
        for p in percentiles:
            fila[f"p{p}"] = np.percentile(valores[:, dia], p)
        filas.append(fila)
    return pd.DataFrame(filas)
