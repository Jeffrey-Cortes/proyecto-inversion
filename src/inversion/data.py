"""
Descarga y limpieza de datos de precios. Este es el único módulo que toca
fuentes externas (yfinance, CSV crudos) — el resto del paquete solo trabaja
con DataFrames ya limpios (columnas 'fecha', 'cierre').
"""
import pandas as pd
import yfinance as yf


def limpiar_sp500_mensual(raw_path: str, fecha_inicio: str = "2000-01-01") -> pd.DataFrame:
    """Limpia el dataset mensual del S&P 500 (nivel del índice, datos de
    Robert Shiller vía GitHub: datasets/s-and-p-500).

    Nota: el CSV crudo no viaja con este repo — se descargó en una sesión
    cloud anterior donde Yahoo Finance estaba bloqueado por política de red
    (ver Paso 1 en CLAUDE.md). Si no tienes `raw_path`, usa directamente
    data/processed/sp500_historico.csv, que ya es el resultado de esta
    función.
    """
    df = pd.read_csv(raw_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "SP500"]].rename(columns={"Date": "fecha", "SP500": "cierre"})
    df = df[df["fecha"] >= fecha_inicio].dropna(subset=["cierre"])
    return df.reset_index(drop=True)


def descargar_spy_diario() -> pd.DataFrame:
    """Descarga el histórico diario completo de SPY vía yfinance (cotiza
    desde 1993), precio ajustado por dividendos y splits (OHLCV completo:
    la Fase 3 Paso 3 necesita volumen para features de ML, no solo el
    cierre). Requiere acceso a Yahoo Finance — funciona desde una máquina
    local normal, pero puede estar bloqueado en algunos entornos cloud
    restringidos.
    """
    df = yf.download("SPY", period="max", auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    df.columns = ["fecha", "apertura", "maximo", "minimo", "cierre", "volumen"]
    return df.dropna(subset=["cierre"]).sort_values("fecha").reset_index(drop=True)
