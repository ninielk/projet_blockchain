# utils existantes (gardées)
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def read_raw(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{name}.csv", parse_dates=["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"]).sort_values("Date")

def add_log_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out[price_col] / out[price_col].shift(1))
    return out

def rolling_vol(series: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    vol = series.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol

def rolling_corr(x: pd.Series, y: pd.Series, window: int = 90) -> pd.Series:
    return x.rolling(window).corr(y)

# ---- nouveau : merge à 3 (BTC, SPX, TECH)
def merge_btc_spx_tech() -> pd.DataFrame:
    btc  = add_log_return(read_raw("btc"))[["Date", "Close", "ret"]].rename(columns={"Close": "btc_close",  "ret": "ret_btc"})
    spx  = add_log_return(read_raw("spx"))[["Date", "Close", "ret"]].rename(columns={"Close": "spx_close",  "ret": "ret_spx"})
    tech = add_log_return(read_raw("tech"))[["Date", "Close", "ret"]].rename(columns={"Close": "tech_close", "ret": "ret_tech"})

    df = (
        btc.merge(spx, on="Date", how="outer")
           .merge(tech, on="Date", how="outer")
           .sort_values("Date")
           .reset_index(drop=True)
    )
    return df

# ---- nouveau : features étendus
def build_features_3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vol30_btc"]  = rolling_vol(out["ret_btc"],  window=30)
    out["vol30_spx"]  = rolling_vol(out["ret_spx"],  window=30)
    out["vol30_tech"] = rolling_vol(out["ret_tech"], window=30)
    out["corr90_btc_spx"]  = rolling_corr(out["ret_btc"], out["ret_spx"],  window=90)
    out["corr90_btc_tech"] = rolling_corr(out["ret_btc"], out["ret_tech"], window=90)
    return out

def save_processed(df: pd.DataFrame, name: str = "btc_spx_tech.csv") -> Path:
    path = PROC_DIR / name
    df.to_csv(path, index=False)
    return path
