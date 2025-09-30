import pandas as pd

df = pd.read_csv("data/processed/btc_spx_tech.csv", parse_dates=["Date"])
print(df.head(3))
print("\nColonnes:", df.columns.tolist())
print("\nDernière date:", df["Date"].max())
print("\nTypes:\n", df.dtypes)
