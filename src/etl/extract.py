import yfinance as yf
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("data/processed/btc_spx_tech_gold.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def download_data(start: str = "2016-01-01"):
    tickers = {
        "BTC-USD": "btc",
        "^GSPC":   "spx",
        "QQQ":     "tech",
        "GC=F":    "gold",   # Or (futures continus). Alternative possible: "GLD"
    }

    dfs = []
    for t, name in tickers.items():
        df = yf.download(
            t,
            start=start,
            auto_adjust=True,
            progress=False,
            interval="1d",
        )
        if df.empty:
            raise RuntimeError(f"Aucune donnée téléchargée pour {t}")
        df = df[["Close"]].rename(columns={"Close": f"price_{name}"})
        dfs.append(df)

    # Alignement sur l'intersection des dates et nettoyage
    df_all = pd.concat(dfs, axis=1).dropna(how="any")

    # Conversions explicites en float
    for col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce").astype(float)

    # Rendements
    df_all["ret_btc"]   = df_all["price_btc"].pct_change()
    df_all["ret_spx"]   = df_all["price_spx"].pct_change()
    df_all["ret_tech"]  = df_all["price_tech"].pct_change()
    df_all["ret_gold"]  = df_all["price_gold"].pct_change()

    # Remet 'Date' en colonne
    df_all = df_all.reset_index()

    # Sauvegarde
    df_all.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")

    # Log clair
    date_min = df_all["Date"].min().date()
    date_max = df_all["Date"].max().date()
    print(f"Saved {OUTPUT_PATH} with shape {df_all.shape}")
    print("Date range after alignment:", date_min, "→", date_max)
    print("Dtypes:")
    print(df_all.dtypes)

if __name__ == "__main__":
    download_data()