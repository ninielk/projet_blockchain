from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

IN = "data/processed/btc_spx_returns.csv"
OUT = Path("figs"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN, parse_dates=["Date"])
df = df.dropna(subset=["ret_btc","ret_spx"])  # on enlève les tout-débuts

# 1) Volatilité 30j (annualisée)
plt.figure()
plt.plot(df["Date"], df["vol30_btc"], label="BTC vol30")
plt.plot(df["Date"], df["vol30_spx"], label="S&P500 vol30")
plt.title("Volatilité annualisée (30 jours)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT/"vol30.png", dpi=180)

# 2) Corrélation 90j
plt.figure()
plt.plot(df["Date"], df["corr90"])
plt.axhline(0, linestyle="--")
plt.title("Corrélation roulante 90j : BTC vs S&P500")
plt.tight_layout()
plt.savefig(OUT/"corr90.png", dpi=180)

# 3) Scatter rendements (visuel beta)
plt.figure()
plt.scatter(df["ret_spx"], df["ret_btc"], s=6)
plt.title("BTC vs S&P500 (rendements journaliers)")
plt.xlabel("r_SPX"); plt.ylabel("r_BTC")
plt.tight_layout()
plt.savefig(OUT/"scatter.png", dpi=180)

print(" PNG prêts dans ./figs : vol30.png, corr90.png, scatter.png")
