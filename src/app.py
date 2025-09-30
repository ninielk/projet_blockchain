# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/btc_spx_tech.csv")

# ---------- helpers ----------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def base100_from_price(p: pd.Series) -> pd.Series:
    p = p.astype(float).dropna()
    if p.empty:
        return p
    return (p / p.iloc[0]) * 100.0

def base100_from_returns(r: pd.Series) -> pd.Series:
    r = r.copy()
    if len(r) and pd.isna(r.iloc[0]):
        r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod() * 100.0

def eq_curve_from_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()

def drawdown_series_from_returns(r: pd.Series) -> pd.Series:
    eq = eq_curve_from_returns(r)
    return (eq / eq.cummax()) - 1.0

def max_drawdown_from_returns(r: pd.Series) -> float:
    dd = drawdown_series_from_returns(r)
    return float(dd.min()) if len(dd) else np.nan

def ols_summary(df: pd.DataFrame, ycol: str, xcol: str):
    sub = df[[ycol, xcol]].dropna()
    if sub.shape[0] <= 50:
        return None
    X = sm.add_constant(sub[xcol])
    y = sub[ycol]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model

# ---------- app ----------
st.set_page_config(page_title="BTC vs Marchés — Risk Dashboard", layout="wide")
st.title("BTC vs S&P500 & Tech — Risk & Correlation Dashboard")

df = load_data(DATA_PATH)

# contrôles
col1, col2, col3 = st.columns(3)
with col1:
    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    start = st.date_input("Début", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end = st.date_input("Fin", value=max_date, min_value=min_date, max_value=max_date)
with col3:
    win_vol = st.number_input("Fenêtre Vol (jours)", min_value=10, max_value=252, value=30, step=5)
    win_corr = st.number_input("Fenêtre Corr (jours)", min_value=20, max_value=252, value=90, step=5)

# période
mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
d = df.loc[mask].copy()

# vol/corr
for a in ["btc", "spx", "tech"]:
    d[f"vol_{a}"] = d[f"ret_{a}"].rolling(int(win_vol)).std() * np.sqrt(252)

d["corr_btc_spx"] = d["ret_btc"].rolling(int(win_corr)).corr(d["ret_spx"])
d["corr_btc_tech"] = d["ret_btc"].rolling(int(win_corr)).corr(d["ret_tech"])

# KPIs — ligne 1
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Vol BTC (annual.)", f"{d['vol_btc'].dropna().iloc[-1]:.1%}" if d['vol_btc'].notna().any() else "N/A")
with k2:
    st.metric("Vol S&P (annual.)", f"{d['vol_spx'].dropna().iloc[-1]:.1%}" if d['vol_spx'].notna().any() else "N/A")
with k3:
    st.metric("Vol Tech (annual.)", f"{d['vol_tech'].dropna().iloc[-1]:.1%}" if d['vol_tech'].notna().any() else "N/A")
with k4:
    last_corr = d["corr_btc_tech"].dropna().iloc[-1] if d["corr_btc_tech"].notna().any() else np.nan
    st.metric("Corr BTC~Tech (fenêtre)", f"{last_corr:.2f}" if np.isfinite(last_corr) else "N/A")

# KPIs — ligne 2 (Max Drawdown BTC / S&P / Tech)
dd1, dd2, dd3 = st.columns(3)
with dd1:
    dd_btc = max_drawdown_from_returns(d["ret_btc"])
    st.metric("Max Drawdown BTC", f"{dd_btc:.0%}" if np.isfinite(dd_btc) else "N/A")
with dd2:
    dd_spx = max_drawdown_from_returns(d["ret_spx"])
    st.metric("Max Drawdown S&P500", f"{dd_spx:.0%}" if np.isfinite(dd_spx) else "N/A")
with dd3:
    dd_tech = max_drawdown_from_returns(d["ret_tech"])
    st.metric("Max Drawdown Tech (QQQ)", f"{dd_tech:.0%}" if np.isfinite(dd_tech) else "N/A")

st.markdown("---")

# onglets
tab1, tab2, tab3, tabDD, tab4, tab5 = st.tabs(
    ["Volatilité", "Corrélation", "Base 100", "Drawdown", "OLS: BTC ~ S&P500", "OLS: BTC ~ Tech"]
)

# Vol
with tab1:
    st.subheader("Volatilité annualisée")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(d["Date"], d["vol_btc"],  label="BTC")
    ax.plot(d["Date"], d["vol_spx"],  label="S&P500")
    ax.plot(d["Date"], d["vol_tech"], label="Tech (QQQ)")
    ax.set_ylabel("Vol annualisée")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# Corr
with tab2:
    st.subheader("Corrélation roulante BTC vs marchés")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(d["Date"], d["corr_btc_spx"],  label="BTC ~ S&P500", linewidth=1.5)
    ax.plot(d["Date"], d["corr_btc_tech"], label="BTC ~ Tech (QQQ)", linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Corrélation (fenêtre)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# Base 100
with tab3:
    st.subheader("Évolution comparée")

    def target_vol_index(r: pd.Series, window: int = 30, target_ann_vol: float = 0.20) -> pd.Series:
        vol_ann = r.rolling(window).std() * np.sqrt(252)
        scale = target_ann_vol / vol_ann.replace(0, np.nan)
        r_scaled = (r * scale).fillna(0.0)
        return (1.0 + r_scaled).cumprod() * 100.0

    mode = st.radio(
        "Échelle / Méthode",
        ["Base 100 (log)", "Base 100 (linéaire)", "Risque normalisé (vol cible)"],
        index=0,
        horizontal=True,
    )

    have_prices = {"price_btc", "price_spx", "price_tech"}.issubset(d.columns)
    if have_prices:
        idx_lin = pd.DataFrame({
            "Date": d["Date"],
            "BTC":        base100_from_price(d["price_btc"]),
            "S&P500":     base100_from_price(d["price_spx"]),
            "Tech (QQQ)": base100_from_price(d["price_tech"]),
        }).dropna()
    else:
        idx_lin = pd.DataFrame({
            "Date": d["Date"],
            "BTC":        base100_from_returns(d["ret_btc"]),
            "S&P500":     base100_from_returns(d["ret_spx"]),
            "Tech (QQQ)": base100_from_returns(d["ret_tech"]),
        }).dropna()

    if mode == "Risque normalisé (vol cible)":
        idx = pd.DataFrame({
            "Date": d["Date"],
            "BTC":        target_vol_index(d["ret_btc"]),
            "S&P500":     target_vol_index(d["ret_spx"]),
            "Tech (QQQ)": target_vol_index(d["ret_tech"]),
        }).dropna()
        ylab = "Indice (vol cible 20%)"
        use_log = False
    else:
        idx = idx_lin.copy()
        ylab = "Indice base 100"
        use_log = (mode == "Base 100 (log)")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(idx["Date"], idx["BTC"],        label="BTC (base 100)")
    ax.plot(idx["Date"], idx["S&P500"],     label="S&P500 (base 100)")
    ax.plot(idx["Date"], idx["Tech (QQQ)"], label="Tech (QQQ) (base 100)")
    if use_log:
        ax.set_yscale("log")
    ax.set_ylabel(ylab)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# Drawdown (courbes)
with tabDD:
    st.subheader("Drawdown sur la période sélectionnée")
    dd_btc_s = drawdown_series_from_returns(d["ret_btc"])
    dd_spx_s = drawdown_series_from_returns(d["ret_spx"])
    dd_tech_s = drawdown_series_from_returns(d["ret_tech"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(d["Date"], dd_btc_s, label="BTC")
    ax.plot(d["Date"], dd_spx_s, label="S&P500")
    ax.plot(d["Date"], dd_tech_s, label="Tech (QQQ)")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.set_ylim(-1.0, 0.05)
    st.pyplot(fig, clear_figure=True)

# OLS BTC ~ S&P
with tab4:
    st.subheader("Scatter & OLS — BTC ~ S&P500")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(d["ret_spx"], d["ret_btc"], s=6)
    ax.set_xlabel("r_SPX"); ax.set_ylabel("r_BTC")
    st.pyplot(fig, clear_figure=True)
    model = ols_summary(d, ycol="ret_btc", xcol="ret_spx")
    if model is not None:
        st.code(model.summary().as_text())
    else:
        st.info("Pas assez de points pour estimer une OLS propre.")

# OLS BTC ~ Tech
with tab5:
    st.subheader("Scatter & OLS — BTC ~ Tech (QQQ)")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(d["ret_tech"], d["ret_btc"], s=6)
    ax.set_xlabel("r_Tech (QQQ)"); ax.set_ylabel("r_BTC")
    st.pyplot(fig, clear_figure=True)
    model = ols_summary(d, ycol="ret_btc", xcol="ret_tech")
    if model is not None:
        st.code(model.summary().as_text())
    else:
        st.info("Pas assez de points pour estimer une OLS propre.")
