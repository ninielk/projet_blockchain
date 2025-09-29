# src/etl/transform.py
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

def read_raw(alias: str) -> pd.DataFrame:
    """
    Charge un CSV brut (Date, Close) et force Close en numérique.
    """
    path = RAW_DIR / f"{alias}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    # sécuriser le type
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df

def add_log_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Ajoute 'ret' = ln(Pt / Pt-1) et enlève la première NA.
    """
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=[price_col])
    out["ret"] = np.log(out[price_col] / out[price_col].shift(1))
    out = out.dropna(subset=["ret"])
    return out

def merge_btc_spx(df_btc: pd.DataFrame, df_spx: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne sur Date et renomme les colonnes de rendement.
    """
    m = (
        df_btc[["Date", "ret"]].rename(columns={"ret": "ret_btc"})
        .merge(
            df_spx[["Date", "ret"]].rename(columns={"ret": "ret_spx"}),
            on="Date",
            how="inner",
        )
    )
    return m

# ---------- Features simples ----------

def rolling_vol(series: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    vol = series.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol

def rolling_corr(x: pd.Series, y: pd.Series, window: int = 90) -> pd.Series:
    return x.rolling(window).corr(y)

def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["vol30_btc"] = rolling_vol(out["ret_btc"], 30, True)
    out["vol30_spx"] = rolling_vol(out["ret_spx"], 30, True)
    out["corr90"] = rolling_corr(out["ret_btc"], out["ret_spx"], 90)
    return out
