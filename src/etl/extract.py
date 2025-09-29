from pathlib import Path
import yfinance as yf
import pandas as pd

START_DATE = "2016-01-01"
END_DATE = None          # None = aujourd'hui (ne pas mettre une date en dur)
INTERVAL = "1d"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "btc": "BTC-USD",
    "spx": "^GSPC",
}

def download_one(alias: str, yfticker: str) -> Path:
    print(f"→ Téléchargement {alias} ({yfticker}) ...")
    df = yf.download(yfticker, start=START_DATE, end=END_DATE,
                     interval=INTERVAL, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"Aucune donnée pour {alias} ({yfticker}).")

    out = df.reset_index()[["Date", "Close"]]
    out_path = RAW_DIR / f"{alias}.csv"
    out.to_csv(out_path, index=False)
    print(f"  ✓ {alias} sauvegardé : {out_path}")
    return out_path

def extract_all() -> dict[str, Path]:
    paths = {}
    for alias, yfticker in TICKERS.items():
        paths[alias] = download_one(alias, yfticker)
    print("✓ Extract terminé.")
    return paths
