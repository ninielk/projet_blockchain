from pathlib import Path
import pandas as pd

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def save_processed(df: pd.DataFrame, name: str = "btc_spx_returns.csv") -> Path:
    out_path = PROC_DIR / name
    df.to_csv(out_path, index=False)
    print(f"✓ Fichier prêt : {out_path}")
    return out_path
