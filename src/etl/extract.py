# src/etl/extract.py
import yfinance as yf
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("data/processed/btc_spx_tech.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def download_data():
    tickers = {
        "BTC-USD": "btc",
        "^GSPC":   "spx",
        "QQQ":     "tech",
    }

    dfs = []
    for t, name in tickers.items():
        df = yf.download(
            t,
            start="2016-01-01",
            auto_adjust=True,
            progress=False,
            interval="1d",
        )
        if df.empty:
            raise RuntimeError(f"Aucune donnée téléchargée pour {t}")
        df = df[["Close"]].rename(columns={"Close": f"price_{name}"})
        dfs.append(df)

    df_all = pd.concat(dfs, axis=1).dropna(how="any")

    # conversions explicites en float
    for col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce").astype(float)

    # rendements
    df_all["ret_btc"]  = df_all["price_btc"].pct_change()
    df_all["ret_spx"]  = df_all["price_spx"].pct_change()
    df_all["ret_tech"] = df_all["price_tech"].pct_change()

    df_all = df_all.reset_index()  # remet 'Date' en colonne

    # Supprimer colonnes parasites (ex: Ticker si présent)
    for col in df_all.columns:
        if df_all[col].dtype == "object" and col != "Date":
            df_all = df_all.drop(columns=[col])

    df_all.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")
    print(f"Saved {OUTPUT_PATH} with shape {df_all.shape} and dtypes:")
    print(df_all.dtypes)


if __name__ == "__main__":
    download_data()
