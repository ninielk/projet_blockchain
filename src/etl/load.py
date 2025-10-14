# src/etl/load.py
from pathlib import Path
import pandas as pd

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def save_processed(df: pd.DataFrame, name: str = "btc_spx_tech_gold.csv") -> Path:
    """
    Sauvegarde le DataFrame traité dans data/processed/
    par défaut sous le nom btc_spx_tech_gold.csv
    """
    out_path = PROC_DIR / name
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"✓ Fichier prêt : {out_path} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    print(f"Période couverte : {df['Date'].min()} → {df['Date'].max()}")
    return out_path
