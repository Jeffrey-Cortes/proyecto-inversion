"""
Validación walk-forward: obligatoria para cualquier modelo cuyos
parámetros se ajustan (fit) sobre los datos — a diferencia de las reglas
del Paso 2 (parámetros clásicos, no ajustados). Ventana expansiva: cada
fold entrena con TODO lo disponible hasta la fecha de corte y prueba en el
bloque siguiente, nunca al revés (evita look-ahead bias a nivel de
validación, no solo a nivel de features).
"""
import pandas as pd

from . import backtest


def folds_expanding(fin_entrenamiento_inicial: str, fin_desarrollo: str, paso: str = "365D"):
    """Genera (fecha_corte, fecha_fin_test) para folds de ventana
    expansiva, sin salirse del periodo de desarrollo (antes del holdout
    final). El primer fold entrena con datos hasta `fin_entrenamiento_inicial`
    y prueba en los siguientes `paso` días; cada fold posterior expande el
    entrenamiento hasta donde terminó el fold anterior."""
    corte = pd.Timestamp(fin_entrenamiento_inicial)
    fin_dev = pd.Timestamp(fin_desarrollo)
    paso_td = pd.Timedelta(paso)
    while True:
        fin_test = corte + paso_td
        if fin_test > fin_dev:
            break
        yield corte, fin_test
        corte = fin_test


def evaluar_modelo(modelo_fn, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
                    folds, capital_inicial: float = 10_000.0, costo_bps: float = 0.0,
                    nombre: str = "modelo") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena y evalúa `modelo_fn()` (factory de un estimador sklearn-like
    nuevo por fold, para no filtrar estado entre folds) en cada fold de
    `folds`. `X`/`y` deben venir de `features.preparar_dataset(df)` (mismo
    índice que un subconjunto de `df`).

    Devuelve:
    - tabla de métricas, una fila por fold (para revisar consistencia
      entre folds, no solo el promedio).
    - serie walk-forward: los retornos de cada fold de prueba concatenados
      en orden cronológico (out-of-sample continuo), lista para graficar
      junto a los benchmarks del Paso 1/2.

    Nota: el primer retorno de cada fold se trata como 0 (mismo criterio
    que `backtest.simular` usa para el primer día de cualquier periodo) —
    una pequeña aproximación en el borde de cada fold, no en la validación
    en sí.
    """
    fechas = df.loc[X.index, "fecha"]
    filas_resumen = []
    tramos_oos = []

    for i, (corte, fin_test) in enumerate(folds, start=1):
        train_mask = fechas <= corte
        test_mask = (fechas > corte) & (fechas <= fin_test)
        if test_mask.sum() == 0:
            continue

        modelo = modelo_fn()
        modelo.fit(X[train_mask], y[train_mask])
        pred = modelo.predict(X[test_mask])
        señal_test = pd.Series(pred, index=X[test_mask].index, dtype=float)

        df_test = df.loc[señal_test.index]
        df_fold = backtest.recortar_periodo(df_test, señal_test, fecha_inicio=df_test["fecha"].min())
        df_fold, valor_fold = backtest.simular(df_fold, capital_inicial, costo_bps)

        fila = backtest.resumen(df_fold, valor_fold, capital_inicial, f"{nombre} (fold {i})")
        fila["fold"] = i
        fila["fecha_inicio"] = df_fold["fecha"].min()
        fila["fecha_fin"] = df_fold["fecha"].max()
        filas_resumen.append(fila)
        tramos_oos.append(df_fold[["fecha", "retorno_estrategia"]])

    tabla_folds = pd.DataFrame(filas_resumen)
    serie_oos = pd.concat(tramos_oos, ignore_index=True)
    return tabla_folds, serie_oos
