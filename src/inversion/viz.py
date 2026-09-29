"""
Paleta y funciones de graficado compartidas. Colores por identidad de
serie, fijos en todo el proyecto y nunca reciclados entre series distintas
(skill de dataviz: slots categóricos en orden fijo).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"
FUERA_MERCADO = "#c3c2b7"

# Identidad fija por serie en todo el proyecto (slots categóricos 1-5)
COLOR_BUY_AND_HOLD = "#2a78d6"  # slot 1 - blue
COLOR_DCA = "#eb6834"           # slot 2 - orange
COLOR_SMA = "#1baf7a"           # slot 3 - aqua
COLOR_MOMENTUM = "#eda100"      # slot 4 - yellow
COLOR_RSI = "#e87ba4"           # slot 5 - magenta

plt.rcParams["font.family"] = "DejaVu Sans"


def _estilo_ejes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(BASELINE)


def graficar_estrategia_vs_benchmark(
    df: pd.DataFrame, col_benchmark: str, col_estrategia: str, col_posicion: str,
    color_estrategia: str, label_estrategia: str, titulo: str, archivo_salida: str,
):
    """Gráfica de 2 paneles (valor del portafolio + drawdown) para una
    estrategia vs. Buy & Hold, con sombreado en los periodos fuera de
    mercado. Mismo formato para SMA, Momentum y RSI."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=SURFACE,
    )

    fuera = df[col_posicion] == 0
    cambios = fuera.ne(fuera.shift()).cumsum()
    for _, tramo in df[fuera].groupby(cambios[fuera]):
        ax1.axvspan(tramo["fecha"].iloc[0], tramo["fecha"].iloc[-1],
                    color=FUERA_MERCADO, alpha=0.18, linewidth=0)

    ax1.plot(df["fecha"], df[col_benchmark], color=COLOR_BUY_AND_HOLD, linewidth=2, label="Buy & Hold")
    ax1.plot(df["fecha"], df[col_estrategia], color=color_estrategia, linewidth=2, label=label_estrategia)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_ylabel("Valor del portafolio (USD, escala log)", color=INK_SECONDARY, fontsize=10)
    ax1.set_title(titulo, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12)
    _estilo_ejes(ax1)
    ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    finales = sorted([(col_benchmark, COLOR_BUY_AND_HOLD), (col_estrategia, color_estrategia)],
                      key=lambda t: df[t[0]].iloc[-1])
    for (col, color), dy in zip(finales, [-8, 8]):
        ax1.annotate(f"${df[col].iloc[-1]:,.0f}", xy=(df["fecha"].iloc[-1], df[col].iloc[-1]),
                     xytext=(6, dy), textcoords="offset points",
                     color=color, fontsize=9, fontweight="bold", va="center")

    pico_bh = df[col_benchmark].cummax()
    dd_bh = (df[col_benchmark] - pico_bh) / pico_bh
    pico_est = df[col_estrategia].cummax()
    dd_est = (df[col_estrategia] - pico_est) / pico_est
    ax2.fill_between(df["fecha"], dd_bh * 100, 0, color=COLOR_BUY_AND_HOLD, alpha=0.15, linewidth=0)
    ax2.plot(df["fecha"], dd_bh * 100, color=COLOR_BUY_AND_HOLD, linewidth=1.5)
    ax2.fill_between(df["fecha"], dd_est * 100, 0, color=color_estrategia, alpha=0.15, linewidth=0)
    ax2.plot(df["fecha"], dd_est * 100, color=color_estrategia, linewidth=1.5)
    ax2.axhline(0, color=BASELINE, linewidth=1)
    ax2.set_ylabel("Drawdown (%)", color=INK_SECONDARY, fontsize=10)
    _estilo_ejes(ax2)

    fig.text(0.01, 0.005,
              f"Fuente: SPY (Yahoo Finance vía yfinance), precio ajustado por dividendos. "
              f"Zonas sombreadas = {label_estrategia} fuera del mercado (en efectivo, 0% modelado). "
              "No es asesoría de inversión.", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(archivo_salida, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.show()


def graficar_todas_las_estrategias(df: pd.DataFrame, series: list[tuple[str, str, str]],
                                    titulo: str, archivo_salida: str):
    """Una sola gráfica de líneas (escala log) con varias series encima.
    `series` es una lista de (columna, color, etiqueta)."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)

    for col, color, label in series:
        ax.plot(df["fecha"], df[col], color=color, linewidth=2, label=label)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_ylabel("Valor del portafolio (USD, escala log)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(titulo, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12)
    _estilo_ejes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    finales = sorted(series, key=lambda t: df[t[0]].iloc[-1])
    offsets = [-24, -8, 8, 24][:len(finales)]
    for (col, color, _label), dy in zip(finales, offsets):
        ax.annotate(f"${df[col].iloc[-1]:,.0f}", xy=(df["fecha"].iloc[-1], df[col].iloc[-1]),
                     xytext=(6, dy), textcoords="offset points",
                     color=color, fontsize=9, fontweight="bold", va="center")

    fig.text(0.01, 0.005,
              "Fuente: SPY (Yahoo Finance vía yfinance), precio ajustado por dividendos. "
              "No es asesoría de inversión.", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(archivo_salida, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.show()
