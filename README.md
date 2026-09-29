# Modelo de inversión con ML — benchmark, reglas técnicas y ML clásico sobre SPY

Proyecto de investigación aplicada: ¿puede un modelo de Machine Learning
superar, de forma consistente y validada, a simplemente comprar y mantener
un ETF diversificado? La respuesta corta, después de cinco etapas de
trabajo con disciplina metodológica explícita (walk-forward, holdout
intocado, auditoría propia), es que no — y ese resultado, obtenido con
rigor en vez de buscado a la fuerza, es en sí mismo el hallazgo central
del proyecto.

**Stack**: Python, pandas/numpy, scikit-learn, XGBoost, matplotlib,
Jupyter. Datos: SPY vía `yfinance`.

## Motivación

Empecé este proyecto con dos objetivos en paralelo: aprender a invertir
desde cero, y construir un modelo de ML que generara señales de compra/
venta como ejercicio aplicado de ciencia de datos sobre un dominio
genuinamente difícil — series de tiempo financieras, con ruido alto,
señal débil, no estacionariedad y un mercado adversarial (los patrones
simples y explotables tienden a arbitrarse).

Fui explícito desde el inicio en que esto no se trataba de "hacerse
rico": los mercados son razonablemente eficientes, y la mayoría de las
estrategias caseras no le ganan de forma sostenida a un ETF diversificado
después de costos. El valor real del proyecto está en tres cosas:

1. **Aprendizaje aplicado** de una disciplina que exige el mismo rigor
   contra el overfitting que cualquier otro problema de ML con datos
   escasos y ruidosos.
2. **Disciplina** — reglas explícitas en vez de decisiones emocionales.
3. **Gestión de riesgo** — diversificación, control de drawdown, tamaño
   de posición; terreno con más evidencia real de valor que "predecir
   precios".

Que el modelo superara al benchmark era aspiracional, no garantizado. Si
no lo lograba de forma consistente y validada, el benchmark ganaba por
defecto — y ese desenlace también contaba como éxito del proyecto,
siempre que se llegara a él con metodología seria.

## Principios metodológicos

Dos decisiones, tomadas antes de escribir la primera línea de código de
modelado, moldearon todo lo demás:

- **ML clásico antes que Deep Learning.** Un ETF con 15-20 años de
  historia diaria son ~4,000-5,000 filas — nada para DL. Preferí
  regresión regularizada (Ridge/Lasso/Logística) y árboles (Random
  Forest, XGBoost): buen balance rendimiento/interpretabilidad, relevante
  porque las señales se revisan y ejecutan manualmente, no en caja negra.
- **"Probar varios modelos y quedarse con el mejor" tiene una trampa
  de p-hacking.** Para evitarla: un conjunto de prueba final que nadie
  toca hasta el final absoluto, validación **walk-forward** (nunca CV
  aleatoria, que filtraría información del futuro), comparación siempre
  contra el benchmark simple, y métricas ajustadas a riesgo (Sharpe,
  Sortino, drawdown máximo), no solo rendimiento bruto. El cuello de
  botella real casi nunca es el modelo — es la ingeniería de features, la
  definición del target, evitar look-ahead bias, y la metodología de
  validación.

## Estructura del proyecto

```
proyecto-inversion/
├── src/inversion/         # Lógica reutilizable, un archivo por responsabilidad
│   ├── data.py            #   descarga/limpieza de datos de precios (OHLCV)
│   ├── indicators.py      #   indicadores puros (RSI, retorno a n días)
│   ├── strategies.py      #   señales de cada regla técnica (SMA, momentum, RSI)
│   ├── features.py        #   features + target para ML
│   ├── walkforward.py     #   validación walk-forward (ventana expansiva)
│   ├── backtest.py        #   motor de backtest compartido (señal -> curva de valor + métricas)
│   ├── proyeccion.py      #   Monte Carlo / block bootstrap, vectorizado con numpy
│   ├── metrics.py         #   CAGR, IRR, drawdown, Sharpe, Sortino
│   └── viz.py             #   paleta y funciones de graficado compartidas
├── notebooks/              # Orquestación y exploración — importan de src/, no repiten lógica
│   ├── 01_paso1_benchmark.ipynb
│   ├── 02_paso2_reglas_simples.ipynb
│   ├── 03_paso3_ml_clasico.ipynb
│   ├── 04_paso4_dca_con_timing.ipynb
│   └── 05_auditoria_y_proyeccion.ipynb
├── data/processed/         # Datos de precios ya limpios (input de los notebooks)
├── outputs/                 # Todo lo generado: series/, metrics/, figures/
└── requirements.txt
```

`src/inversion/` concentra toda la lógica de negocio — cada archivo tiene
una sola responsabilidad y no sabe nada de cómo se presentan los
resultados. Los notebooks son la capa de orquestación: solo importan
funciones de `inversion` y grafican, nunca contienen lógica que deba
reusarse entre ellos (esto evitó, por ejemplo, que las fórmulas de
CAGR/Sharpe/Sortino terminaran triplicadas entre las tres estrategias
técnicas — se extrajeron a `metrics.py` en cuanto apareció la tercera
repetición).

## Cómo correr el proyecto

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name proyecto-inversion --display-name "Python (proyecto-inversion .venv)"
```

Abre los notebooks en `notebooks/` (VS Code, JupyterLab, etc.) usando el
kernel `proyecto-inversion`, y corre las celdas en orden — `01` antes que
`02`. Cada notebook guarda sus resultados en `outputs/` y, si hace falta,
sus datos de entrada en `data/processed/`.

Para correr un notebook headless desde terminal (regenerar todo sin abrir
la UI):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_paso1_benchmark.ipynb
```

## Paso 1 — Benchmark (Buy & Hold vs. DCA)

**[`notebooks/01_paso1_benchmark.ipynb`](notebooks/01_paso1_benchmark.ipynb)**
simula dos escenarios sobre los mismos $10,000 USD:
- **Buy & Hold**: todo el dinero invertido de una sola vez al inicio.
- **DCA**: el mismo capital total, repartido en aportaciones mensuales
  iguales a lo largo de todo el periodo.

Calcula valor final, CAGR (Buy & Hold) o IRR anualizado (DCA, la métrica
correcta cuando el dinero entra en momentos distintos), y el drawdown
máximo de cada uno.

### Fuente de datos

En un primer momento no tuve acceso a Yahoo Finance (bloqueado por la red
desde la que trabajaba), así que usé como alternativa un dataset público
(`datasets/s-and-p-500` en GitHub, basado en datos de Robert Shiller). El
CSV crudo no viaja con este repo — el notebook parte directamente de
`data/processed/sp500_historico.csv`, que ya es el resultado limpio de
ese paso.

**Limitaciones de ese dataset:**
- Es el **nivel del índice S&P 500**, no el precio exacto del ETF SPY.
- Es **mensual**, no diario.
- **No incluye dividendos reinvertidos** recientes.

Confirmé después que `yfinance` sí funciona en mi entorno local — esa es
la fuente usada de ahí en adelante (`src/inversion/data.py`).

### Resultados (2000–2026, $10,000 USD)

| Escenario | Capital invertido | Valor final | CAGR / IRR anualizado | Drawdown máximo |
|---|---|---|---|---|
| Buy & Hold | $10,000 | $54,092 | 6.55% | -50.82% |
| DCA (mensual) | $10,000 | $45,546 | 9.96% | -41.46% |

**Lectura clave:** Buy & Hold terminó con más dinero en términos absolutos
porque estuvo expuesto al mercado todo el periodo (26.6 años). DCA tuvo un
drawdown máximo menor (evitó comprar todo de golpe justo antes de la
caída de 2000-2002) y, medido en la métrica correcta para dinero que entra
en momentos distintos (IRR), cada dólar de DCA rindió más por año. Ninguna
de las dos es "la respuesta correcta" — son trade-offs distintos entre
rendimiento total y control de riesgo, y es exactamente el tipo de
comparación contra la que se mide cualquier modelo futuro.

### Archivos generados

- `data/processed/sp500_historico.csv`
- `outputs/series/series_comparacion.csv`
- `outputs/metrics/resumen_metricas.csv`
- `outputs/figures/comparacion_buyhold_vs_dca.png`

## Paso 2 — Reglas simples basadas en indicadores técnicos

**[`notebooks/02_paso2_reglas_simples.ipynb`](notebooks/02_paso2_reglas_simples.ipynb)**
corre tres estrategias sobre datos **diarios** reales de SPY (vía
`yfinance`, precio ajustado por dividendos) — ya no el proxy mensual del
índice del Paso 1.

Las tres comparan contra el mismo Buy & Hold, recalculado sobre los mismos
datos diarios para que la comparación sea limpia. La señal calculada al
cierre del día *t* se ejecuta en el retorno del día *t+1* (anti
look-ahead bias). Ninguna necesita walk-forward: los parámetros son
valores clásicos de la literatura, no ajustados sobre estos datos —
walk-forward se vuelve obligatorio recién en el Paso 3, donde sí hay
ajuste de parámetros.

- **SMA 50/200** (trend-following): invertido cuando SMA(50) > SMA(200).
- **Momentum 12 meses** (trend-following): invertido cuando el retorno de
  los últimos 252 días de trading es positivo (ventana clásica de la
  literatura de time-series momentum, Moskowitz/Ooi/Pedersen 2012).
- **RSI(14) 30/70** (mean-reversion, contrarian): entra cuando el RSI
  (suavizado de Wilder) cruza por debajo de 30, sale cuando cruza por
  arriba de 70, con histéresis.

### Resultados (2000–2026, $10,000 USD, mismos datos diarios de SPY)

| Escenario | Valor final | CAGR | Drawdown máximo | Sharpe (rf=0) | Sortino (rf=0) |
|---|---|---|---|---|---|
| Buy & Hold | $84,017 | 8.28% | -55.19% | 0.51 | 0.65 |
| SMA 50/200 | $81,667 | 8.17% | -33.72% | 0.65 | 0.70 |
| Momentum 12m | $66,343 | 7.33% | -31.17% | 0.60 | 0.65 |
| RSI(14) 30/70 | $27,997 | 3.92% | -55.19% | 0.32 | 0.26 |

**Lectura general:** las dos reglas de tendencia mejoran el perfil de
riesgo (menor drawdown, mejor Sharpe/Sortino) a cambio de algo de
rendimiento total — un trade-off razonable.

**Hallazgo del RSI:** compró en la caída de 2008 (sobreventa) y **nunca
recibió señal de salida durante toda la crisis** — se quedó deprimido sin
volver a cruzar 70 mientras el precio seguía cayendo. Terminó con
exactamente el mismo drawdown máximo que Buy & Hold (-55.19%, mismo día:
2009-03-09), pero con mucho peor rendimiento. Lección concreta: una señal
de entrada sin una de salida robusta (o sin stop-loss) puede dejar
atrapado el capital en una caída completa — "sobreventa" no garantiza
rebote.

Ninguna de las tres reglas le gana a Buy & Hold en rendimiento absoluto
todavía, y sigue sin haber comisiones, slippage ni impuestos modelados.

### Archivos generados

- `data/processed/spy_diario.csv`
- `outputs/series/series_comparacion_{sma,momentum,rsi,todas}.csv`
- `outputs/metrics/resumen_metricas_{sma,momentum,rsi,todas}.csv`
- `outputs/figures/comparacion_buyhold_vs_{sma,momentum,rsi}.png`
- `outputs/figures/comparacion_todas_estrategias.png`

## Paso 3 — ML clásico

**[`notebooks/03_paso3_ml_clasico.ipynb`](notebooks/03_paso3_ml_clasico.ipynb)**
responde la pregunta central del proyecto: ¿le gana un modelo de ML a
comprar y no hacer nada? A diferencia del Paso 2, aquí los modelos sí
ajustan parámetros a los datos — por eso este paso agrega dos piezas de
disciplina: **validación walk-forward** (ventana expansiva, nunca se
prueba con datos anteriores al entrenamiento) y **un holdout final que se
evalúa una sola vez**, sin iterar después de verlo.

- **Target**: señal binaria — 1 si el retorno de los próximos 21 días de
  trading (~1 mes) es positivo. Horizonte mensual elegido porque el
  retorno diario de un ETF es casi un random walk (señal demasiado débil
  para aprender sin sobreajustar) y porque las señales se ejecutan
  manualmente, con una cadencia de revisión que no tiene sentido hacer
  diaria.
- **Features** (`src/inversion/features.py`, 11 en total, todas con solo
  datos pasados): retornos rezagados (1/5/21 días), distancia a SMA50/
  SMA200, RSI(14) continuo, volatilidad móvil (21/63 días), momentum 12m
  continuo, volumen relativo.
- **Modelos candidatos**: Logística L1, Logística L2, Random Forest,
  XGBoost, comparados vía walk-forward (12 folds anuales, 2010–2021), con
  costos de transacción incluidos desde la selección, no solo al final.
- **Selección**: criterio fijado de antemano (mayor Sharpe promedio entre
  folds), no elegido a ojo después de ver resultados. Ganó Logística L2,
  por un margen mínimo sobre XGBoost — y los 4 candidatos tuvieron
  desviación estándar entre folds **mayor que su media**, señal de que
  ninguno generalizó de forma consistente año con año.
- **Holdout**: 2022-01-03 a 2026-08-28 (~4.7 años), nunca tocado hasta
  evaluar la configuración ya elegida.

### Resultados (holdout 2022–2026, $10,000 USD, con costos)

| Escenario | Valor final | CAGR | Drawdown máximo | Sharpe (rf=0) | Sortino (rf=0) | Operaciones |
|---|---|---|---|---|---|---|
| Buy & Hold | $17,108 | 12.23% | -24.53% | 0.75 | 1.04 | 0 |
| SMA 50/200 | $15,412 | 9.74% | -18.76% | 0.76 | 0.91 | 4 |
| RSI(14) 30/70 | $15,419 | 9.75% | -20.56% | 0.73 | 0.68 | 10 |
| Momentum 12m | $14,298 | 7.99% | -22.63% | 0.64 | 0.71 | 14 |
| **ML (Logística L2)** | **$12,878** | **5.59%** | -18.76% | 0.49 | 0.54 | **94** |

**El ML quedó último de las cinco estrategias en todas las métricas** —
peor incluso que el RSI, la peor regla del Paso 2. La alerta ya estaba en
el walk-forward (ningún candidato generalizó de forma confiable), y el
holdout la confirmó: 94 operaciones en ~4.7 años (vs. 4-14 de las reglas
simples) es la firma típica de un modelo reaccionando a ruido, no a señal
real, pagando costos en cada vuelta.

Siguiendo la regla del proyecto desde el Paso 1 ("si el modelo no le gana
de forma consistente y validada, el benchmark gana por defecto"): Buy &
Hold siguió siendo la estrategia ganadora en este punto del proyecto. No
es un fracaso del experimento — es exactamente lo que la disciplina
metodológica (walk-forward + holdout intocado) está diseñada para
producir de forma honesta, en vez de un resultado casual que no se
sostendría.

### Archivos generados

- `data/processed/spy_diario.csv` (regenerado con OHLCV completo)
- `outputs/metrics/resumen_metricas_holdout_ml.csv`
- `outputs/series/series_comparacion_holdout_ml.csv`
- `outputs/figures/walkforward_consistencia_sharpe.png`
- `outputs/figures/comparacion_holdout_ml_vs_reglas.png`

## Elección de bróker

Comparé tres opciones para ejecutar las señales manualmente (sin API de
trading automático, por decisión de diseño):

| | GBM+ | XTB | Interactive Brokers |
|---|---|---|---|
| Regulador | CNBV/SHCP (México) | FSC Belice — más débil | SEC/FINRA (EE.UU.), SIPC |
| Comisión en ETFs EE.UU. | ~0.25% | 0% hasta €100k/mes, luego 0.2% | $0 (Lite) |
| API de Python | No | No | Sí (TWS API / ib_insync) |

Decisión: **GBM+**. La estrategia que terminó validada opera muy poco
(Buy & Hold: 0 operaciones; SMA: 4-26 en 26+ años), lo que le baja peso a
la comisión por operación y a tener API de Python — los datos de señales
salen de `yfinance`, no del bróker, y la ejecución es manual por diseño.
El criterio que más pesó fue protección regulatoria: entre GBM+ (CNBV,
custodia en Indeval, cuenta local) e IBKR (SIPC, cuenta en EE.UU.), elegí
GBM+ por la comodidad de operar bajo el regulador y la jurisdicción
locales. XTB quedó descartado solo por ese criterio (regulación más
débil).

## Paso 4 — DCA con señal de timing

**[`notebooks/04_paso4_dca_con_timing.ipynb`](notebooks/04_paso4_dca_con_timing.ipynb)**
adapta el proyecto a un escenario de inversión más realista para un
ahorrador que empieza de cero: sin capital inicial grande, aportando una
cantidad pequeña y recurrente cada mes (~$12 USD, del orden de 200-300
MXN). Eso descarta Buy & Hold puro (asume capital ya disponible) — la
pregunta pasa a ser si conviene combinar **DCA** con una **señal de
timing** que decida, cada mes, si esa aportación se invierte o se guarda
en efectivo hasta que la señal sea favorable.

Reuso las reglas fijas del Paso 2 (SMA, Momentum, RSI) — no el modelo de
ML del Paso 3, que ya quedó último de cinco estrategias en su holdout y
no hay motivo para esperar que mejore solo por cambiar cómo entra el
capital; probarlo aquí exigiría además un holdout nuevo. Al ser
parámetros fijos, reusar las reglas del Paso 2 en este mecanismo de
aportación distinto no reabre ningún riesgo de sobreajuste.

### Resultados (DCA mensual, $12 USD/mes, 2000–2026, con costos reales de GBM+: 25 bps)

| Escenario | Valor final | IRR anualizado | Drawdown máximo | Sharpe (rf=0) |
|---|---|---|---|---|
| DCA simple | $23,449 | 11.64% | -47.53% | 0.52 |
| **DCA + SMA 50/200** | **$23,661** | **11.69%** | **-43.24%** | **0.55** |
| DCA + Momentum 12m | $23,074 | 11.54% | -43.75% | 0.53 |
| DCA + RSI(14) 30/70 | $23,334 | 11.61% | -47.64% | 0.54 |

**¿Aportar semanal en vez de mensual?** No ayuda — sale ligeramente peor
en las 4 variantes, con el mismo presupuesto total. Razón mecánica, no
solo ruido: aportar semanal implica ~4.3x más eventos de compra, cada uno
pagando el costo de transacción.

### ¿Y si la señal también vende lo ya invertido?

La tabla de arriba usa un mecanismo que **solo controla el dinero
nuevo** — las unidades ya compradas nunca se venden, aunque la señal se
apague después. Probé la versión que también vende: el resultado fue
contundente, **vender es dramáticamente peor**, no solo un poco.

| Escenario | Valor final (solo compra) | Valor final (compra y vende) |
|---|---|---|
| SMA 50/200 | $23,661 | $14,224 (-40%) |
| Momentum 12m | $23,074 | $13,805 (-40%) |
| RSI(14) 30/70 | $23,334 | $9,023 (-61%, peor que no hacer timing) |

Razón: al vender también, el costo de transacción se paga sobre **todo el
portafolio acumulado** en cada cambio de señal, no solo sobre la
aportación del mes — y el portafolio crece durante 26.8 años, así que los
cambios tardíos salen carísimos. Vender también expone años de ganancias
ya compuestas al riesgo de que la señal se equivoque. El diseño "solo
compra" no era una limitación a corregir: resultó ser la mejor decisión
de las dos.

### Archivos generados

- `outputs/figures/dca_con_timing_mensual.png`

## Paso 5 — Auditoría del proyecto y proyección hacia adelante

**[`notebooks/05_auditoria_y_proyeccion.ipynb`](notebooks/05_auditoria_y_proyeccion.ipynb)**
hace dos cosas: una revisión crítica de todo lo construido hasta ese
punto (¿se hicieron bien los benchmarks?, ¿algo se pudo haber hecho
distinto?), y una simulación nueva: con DCA + SMA, $12 USD/mes, ¿cuánto
tendría en distintos horizontes?

### Hallazgos de la auditoría

1. **Costo de transacción corregido.** Los Pasos 3 y 4 usaban 5 bps
   (supuesto inicial de bajo costo); el bróker finalmente elegido fue
   GBM+, con ~25 bps reales en ETFs de EE.UU. — 5 veces más. Corregido en
   el Paso 4 (tabla de arriba ya con el costo real). No cambió la
   conclusión, pero confirmó que no dependía de un supuesto de costo
   optimista.
2. **El Paso 1 no es directamente comparable con el resto** del proyecto
   (índice mensual sin dividendos vs. SPY diario con dividendos desde el
   Paso 2) — limitación documentada, no corregida retroactivamente.
3. **Un desfase de un día** entre el motor de backtest del Paso 2/3 y el
   del Paso 4 — no es look-ahead bias (si acaso es más conservador), y no
   cambia ninguna conclusión porque aplica por igual dentro de cada
   comparación.
4. **El holdout del Paso 3 (2022-2026) fue una racha favorable para
   Buy & Hold** — una sola corrección moderada, nada como 2008. La
   conclusión de ese paso es específica de ese tramo histórico.
5. El efectivo fuera de mercado se modeló con 0% de rendimiento en todo
   el proyecto — conservador en contra de las estrategias de timing.
6. No encontré errores de look-ahead bias ni bugs que invaliden las
   conclusiones generales tras revisar cada motor con cuidado.

### Proyección hacia adelante: block bootstrap, no una predicción

Con $12 USD/mes, DCA simple, 2,000 futuros simulados (block bootstrap de
retornos históricos de SPY — sin pretender predecir el mercado):

| Años | p10 (mal escenario) | p50 (mediana) | p90 (buen escenario) |
|---|---|---|---|
| 5 | $742 | $967 | $1,239 |
| 10 | $1,753 | $2,579 | $3,805 |
| 15 | $3,201 | $5,343 | $8,974 |
| 20 | $5,446 | $9,918 | $18,724 |

### El hallazgo más importante del proyecto

En el backtest histórico, la SMA reducía el drawdown ~4 puntos
porcentuales frente a DCA simple. **En la proyección hacia adelante
(miles de futuros alternativos, recombinando los mismos retornos
históricos en otro orden), esa ventaja casi desaparece** — resultado
consistente probando tres tamaños de bloque distintos. DCA simple terminó
ligeramente arriba de DCA+SMA en las tres pruebas, y la reducción de
drawdown bajó de ~4pp a menos de 1pp.

La ventaja histórica de la SMA vino de evitar, de forma muy específica,
los 18 meses seguidos fuera de mercado durante 2008-2009 — una racha
particular de esa trayectoria histórica exacta. Al generar miles de
futuros alternativos, esa racha específica no se repite de la misma
forma, y el efecto se cancela en promedio.

### Conclusión final

Con toda la evidencia junta (walk-forward, holdout, backtest histórico, y
la proyección hacia adelante), **ninguna señal de timing probada — ni ML,
ni SMA, ni Momentum, ni RSI — mostró una ventaja que se sostenga fuera de
la trayectoria histórica específica donde se midió por primera vez.**

**Decisión final del proyecto: DCA simple, mensual, automatizado, sin
señal de timing, vía GBM+.** No es "no se me ocurrió nada mejor" — es lo
único que sobrevivió a cada intento honesto de superarlo. Es tan buena
opción como cualquier otra que probé, y mucho más simple de operar (cero
revisión mensual, se puede dejar en automático).

### Archivos generados

- `src/inversion/proyeccion.py`
- `outputs/figures/proyeccion_dca_20_años.png`

## Limitaciones conocidas

- **Un solo activo** (SPY / S&P 500 de EE.UU.) — nunca se probó
  diversificación entre clases de activos, que en teoría tiene más
  evidencia real de valor que intentar predecir el precio de un solo
  activo.
- **Sin impuestos modelados** en ningún paso.
- **Riesgo cambiario MXN/USD no incorporado** al análisis cuantitativo —
  todo el backtest está en USD; las aportaciones reales entran en pesos.
- **Efectivo fuera de mercado modelado a 0% de rendimiento** — conservador
  en contra de las estrategias de timing.
- **El holdout del Paso 3 cubre un periodo (2022-2026) sin una caída
  prolongada tipo 2008** — la conclusión de ese paso es válida para esa
  ventana específica, no necesariamente para cualquier futuro.
- Deep Learning quedó fuera de alcance deliberadamente: con ~5,500 filas
  de datos utilizables, no hay suficiente historia para justificarlo
  (aun un modelo clásico simple, con muchos menos parámetros, ya mostró
  señales de sobreajuste). Se consideraría solo con un caso de uso
  concreto (ej. NLP sobre sentiment de noticias, o expansión a muchos
  activos con datos masivos).
