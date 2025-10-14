# src/quick_plots.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN = "data/processed/btc_spx_tech_gold.csv"
OUT = Path("figs"); OUT.mkdir(parents=True, exist_ok=True)

BRAND = {
    "btc":  "#B10967",  # magenta
    "spx":  "#412761",  # indigo
    "tech": "#007078",  # teal
    "gold": "#F8AF00",  # gold
}

df = pd.read_csv(IN, parse_dates=["Date"]).dropna()

# ---- Volatilité annualisée 30j
plt.figure()
plt.plot(df["Date"], df["ret_btc"].rolling(30).std(ddof=1) * (252 ** 0.5),  label="BTC vol30",  color=BRAND["btc"])
plt.plot(df["Date"], df["ret_spx"].rolling(30).std(ddof=1) * (252 ** 0.5),  label="S&P500 vol30", color=BRAND["spx"])
plt.plot(df["Date"], df["ret_tech"].rolling(30).std(ddof=1) * (252 ** 0.5), label="Tech vol30",  color=BRAND["tech"])
plt.plot(df["Date"], df["ret_gold"].rolling(30).std(ddof=1) * (252 ** 0.5), label="Gold vol30",  color=BRAND["gold"])
plt.title("Volatilité annualisée (30 jours)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "vol30.png", dpi=180)

# ---- Corrélations roulantes 90j : BTC vs marchés (incl. Gold)
plt.figure()
plt.plot(df["Date"], df["ret_btc"].rolling(90).corr(df["ret_spx"]),   label="BTC ~ S&P500",  color=BRAND["spx"])
plt.plot(df["Date"], df["ret_btc"].rolling(90).corr(df["ret_tech"]),  label="BTC ~ Tech",    color=BRAND["tech"])
plt.plot(df["Date"], df["ret_btc"].rolling(90).corr(df["ret_gold"]),  label="BTC ~ Gold",    color=BRAND["gold"])
plt.axhline(0, linestyle="--", color="black", linewidth=1)
plt.title("Corrélation roulante 90j")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "corr90.png", dpi=180)

# ---- Scatter BTC vs marchés (incl. Gold)
plt.figure()
plt.scatter(df["ret_spx"],  df["ret_btc"], s=6, alpha=0.6, label="BTC vs S&P500")
plt.scatter(df["ret_tech"], df["ret_btc"], s=6, alpha=0.6, label="BTC vs Tech")
plt.scatter(df["ret_gold"], df["ret_btc"], s=6, alpha=0.6, label="BTC vs Gold", color=BRAND["gold"])
plt.xlabel("r_X"); plt.ylabel("r_BTC")
plt.title("Scatter rendements BTC vs Marchés")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "scatter.png", dpi=180)

print("PNG prêts dans ./figs : vol30.png, corr90.png, scatter.png")
