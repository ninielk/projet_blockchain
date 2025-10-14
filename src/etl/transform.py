# src/etl/transform.py
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def read_raw(name: str) -> pd.DataFrame:
    """
    Attend un CSV data/raw/<name>.csv avec colonnes: Date, Close
    """
    df = pd.read_csv(RAW_DIR / f"{name}.csv", parse_dates=["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"]).sort_values("Date")

def add_log_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    # log-return: ln(P_t / P_{t-1})
    out["ret"] = np.log(out[price_col] / out[price_col].shift(1))
    return out

def rolling_vol(series: pd.Series, window: int = 30, annualize: bool = True) -> pd.Series:
    vol = series.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol

def rolling_corr(x: pd.Series, y: pd.Series, window: int = 90) -> pd.Series:
    return x.rolling(window).corr(y)

# ---- existant : merge à 3 (BTC, SPX, TECH)
def merge_btc_spx_tech(join: str = "outer") -> pd.DataFrame:
    btc  = add_log_return(read_raw("btc"))[["Date", "Close", "ret"]].rename(columns={"Close": "btc_close",  "ret": "ret_btc"})
    spx  = add_log_return(read_raw("spx"))[["Date", "Close", "ret"]].rename(columns={"Close": "spx_close",  "ret": "ret_spx"})
    tech = add_log_return(read_raw("tech"))[["Date", "Close", "ret"]].rename(columns={"Close": "tech_close", "ret": "ret_tech"})

    df = (
        btc.merge(spx, on="Date", how=join)
           .merge(tech, on="Date", how=join)
           .sort_values("Date")
           .reset_index(drop=True)
    )
    return df

# ---- existant : features à 3
def build_features_3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vol30_btc"]  = rolling_vol(out["ret_btc"],  window=30)
    out["vol30_spx"]  = rolling_vol(out["ret_spx"],  window=30)
    out["vol30_tech"] = rolling_vol(out["ret_tech"], window=30)
    out["corr90_btc_spx"]  = rolling_corr(out["ret_btc"], out["ret_spx"],  window=90)
    out["corr90_btc_tech"] = rolling_corr(out["ret_btc"], out["ret_tech"], window=90)
    return out

# ==================== NOUVEAU : GOLD ====================

def merge_btc_spx_tech_gold(join: str = "inner") -> pd.DataFrame:
    """
    Fusionne BTC, SPX, TECH, GOLD sur la clé Date.
    join="inner" par défaut (intersection) pour éviter des NaN dans les features.
    """
    btc  = add_log_return(read_raw("btc")) [ ["Date","Close","ret"] ].rename(columns={"Close":"btc_close",  "ret":"ret_btc"})
    spx  = add_log_return(read_raw("spx")) [ ["Date","Close","ret"] ].rename(columns={"Close":"spx_close",  "ret":"ret_spx"})
    tech = add_log_return(read_raw("tech"))[ ["Date","Close","ret"] ].rename(columns={"Close":"tech_close", "ret":"ret_tech"})
    gold = add_log_return(read_raw("gold"))[ ["Date","Close","ret"] ].rename(columns={"Close":"gold_close", "ret":"ret_gold"})

    df = (
        btc.merge(spx,  on="Date", how=join)
           .merge(tech, on="Date", how=join)
           .merge(gold, on="Date", how=join)
           .sort_values("Date")
           .reset_index(drop=True)
    )
    return df

def build_features_4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute :
      - vols 30j annualisées pour btc/spx/tech/gold
      - corrélations 90j, incl. BTC~GOLD (+ autres paires utiles)
    """
    out = df.copy()

    # vol 30j (annualisées)
    out["vol30_btc"]   = rolling_vol(out["ret_btc"],  window=30)
    out["vol30_spx"]   = rolling_vol(out["ret_spx"],  window=30)
    out["vol30_tech"]  = rolling_vol(out["ret_tech"], window=30)
    out["vol30_gold"]  = rolling_vol(out["ret_gold"], window=30)

    # corr 90j (focus demandé + bonus)
    out["corr90_btc_spx"]   = rolling_corr(out["ret_btc"],  out["ret_spx"],  window=90)
    out["corr90_btc_tech"]  = rolling_corr(out["ret_btc"],  out["ret_tech"], window=90)
    out["corr90_btc_gold"]  = rolling_corr(out["ret_btc"],  out["ret_gold"], window=90)
    # Optionnel : autres paires (utile en annexe/QA)
    out["corr90_spx_tech"]  = rolling_corr(out["ret_spx"],  out["ret_tech"], window=90)
    out["corr90_spx_gold"]  = rolling_corr(out["ret_spx"],  out["ret_gold"], window=90)
    out["corr90_tech_gold"] = rolling_corr(out["ret_tech"], out["ret_gold"], window=90)

    return out

def save_processed(df: pd.DataFrame, name: str = "btc_spx_tech_gold.csv") -> Path:
    path = PROC_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
