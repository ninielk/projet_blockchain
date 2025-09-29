import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/btc_spx_returns.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df

def base100(series):
    return (series / series.iloc[0]) * 100.0

st.set_page_config(page_title="BTC vs S&P – Risk Dashboard", layout="wide")
st.title("BTC vs S&P500 — Risk & Correlation Dashboard")

df = load_data()

# === Controls ===
col1, col2, col3 = st.columns(3)
with col1:
    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    start = st.date_input("Début", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end = st.date_input("Fin", value=max_date, min_value=min_date, max_value=max_date)
with col3:
    win_vol = st.number_input("Fenêtre Vol (jours)", 30, 180, 30, step=5)
    win_corr = st.number_input("Fenêtre Corr (jours)", 30, 180, 90, step=5)

mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
d = df.loc[mask].copy()

# Recalcule vol/corr si fenêtres changent
d["vol_btc"] = d["ret_btc"].rolling(win_vol).std() * np.sqrt(252)
d["vol_spx"] = d["ret_spx"].rolling(win_vol).std() * np.sqrt(252)
d["corr"] = d["ret_btc"].rolling(win_corr).corr(d["ret_spx"])

# === KPIs ===
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Vol BTC (annual.)", f"{d['vol_btc'].iloc[-1]:.1%}" if d['vol_btc'].notna().any() else "N/A")
with c2:
    st.metric("Vol S&P (annual.)", f"{d['vol_spx'].iloc[-1]:.1%}" if d['vol_spx'].notna().any() else "N/A")
with c3:
    st.metric("Corr 90j", f"{d['corr'].iloc[-1]:.2f}" if d['corr'].notna().any() else "N/A")
with c4:
    # Max drawdown simple en base 100
    prices_btc = (d["ret_btc"].add(1)).cumprod()
    prices_spx = (d["ret_spx"].add(1)).cumprod()
    dd_btc = (prices_btc / prices_btc.cummax() - 1).min() if len(prices_btc) else 0
    dd_spx = (prices_spx / prices_spx.cummax() - 1).min() if len(prices_spx) else 0
    st.metric("Max Drawdown BTC / S&P", f"{dd_btc:.0%} / {dd_spx:.0%}")

st.markdown("---")

# === Graph 1 : Volatilité ===
st.subheader("Volatilité annualisée")
fig1, ax1 = plt.subplots()
ax1.plot(d["Date"], d["vol_btc"], label="BTC")
ax1.plot(d["Date"], d["vol_spx"], label="S&P500")
ax1.set_ylabel("Vol (annualisée)")
ax1.legend()
st.pyplot(fig1)

# === Graph 2 : Corrélation ===
st.subheader("Corrélation roulante")
fig2, ax2 = plt.subplots()
ax2.plot(d["Date"], d["corr"])
ax2.axhline(0, linestyle="--")
ax2.set_ylabel("Corrélation")
st.pyplot(fig2)

# === Graph 3 : Indice base 100 (prix reconstruits) ===
st.subheader("Évolution comparée (base 100)")
# On reconstruit un indice à partir des rendements
idx_btc = base100((d["ret_btc"].add(1)).cumprod())
idx_spx = base100((d["ret_spx"].add(1)).cumprod())
fig3, ax3 = plt.subplots()
ax3.plot(d["Date"], idx_btc, label="BTC (base 100)")
ax3.plot(d["Date"], idx_spx, label="S&P500 (base 100)")
ax3.legend()
st.pyplot(fig3)

# === Graph 4 : Scatter + Beta OLS (option pro) ===
st.subheader("Scatter rendements & Beta OLS")
fig4, ax4 = plt.subplots()
ax4.scatter(d["ret_spx"], d["ret_btc"], s=6)
ax4.set_xlabel("r_SPX"); ax4.set_ylabel("r_BTC")
st.pyplot(fig4)

# Estimation beta (si assez de points)
if d[["ret_btc","ret_spx"]].dropna().shape[0] > 50:
    X = sm.add_constant(d["ret_spx"].dropna())
    y = d.loc[X.index, "ret_btc"]
    model = sm.OLS(y, X).fit()
    st.code(model.summary().as_text())
else:
    st.info("Pas assez de points pour une OLS propre sur la période sélectionnée.")
