from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN = "data/processed/btc_spx_tech.csv"
OUT = Path("figs"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN, parse_dates=["Date"]).dropna()

plt.figure()
plt.plot(df["Date"], df["ret_btc"].rolling(30).std() * (252**0.5), label="BTC vol30")
plt.plot(df["Date"], df["ret_spx"].rolling(30).std() * (252**0.5), label="S&P500 vol30")
plt.plot(df["Date"], df["ret_tech"].rolling(30).std() * (252**0.5), label="Tech vol30")
plt.title("Volatilité annualisée (30 jours)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT/"vol30.png", dpi=180)

plt.figure()
plt.plot(df["Date"], df["ret_btc"].rolling(90).corr(df["ret_spx"]), label="BTC ~ S&P500")
plt.plot(df["Date"], df["ret_btc"].rolling(90).corr(df["ret_tech"]), label="BTC ~ Tech")
plt.axhline(0, linestyle="--", color="black", linewidth=1)
plt.title("Corrélation roulante 90j")
plt.legend(); plt.tight_layout()
plt.savefig(OUT/"corr90.png", dpi=180)

plt.figure()
plt.scatter(df["ret_spx"], df["ret_btc"], s=6, label="BTC vs S&P500")
plt.scatter(df["ret_tech"], df["ret_btc"], s=6, label="BTC vs Tech", alpha=0.6)
plt.xlabel("r_X"); plt.ylabel("r_BTC")
plt.title("Scatter rendements BTC vs Marchés")
plt.legend(); plt.tight_layout()
plt.savefig(OUT/"scatter.png", dpi=180)

print("PNG prêts dans ./figs : vol30.png, corr90.png, scatter.png")
