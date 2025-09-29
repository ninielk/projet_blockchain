import pandas as pd

df = pd.read_csv("data/processed/btc_spx_returns.csv", parse_dates=["Date"])
print(df.head(3))          # premières lignes
print("\nColonnes:", df.columns.tolist())
print("\nDernière date:", df["Date"].max())
