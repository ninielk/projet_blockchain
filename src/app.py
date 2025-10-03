# src/app.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import statsmodels.api as sm
import streamlit as st

from metrics import (
    TRADING_DAYS, Z_95,
    AnnParams, annualize_mean_vol, sample_vol, sample_mean,
    capm_beta, capm_mu_ann_from_series, mu_ann_from_premium,
    dt_critical, z_from_conf, annual_to_daily_rate,
    fisher_variance_test
)

DATA_PATH = Path("data/processed/btc_spx_tech.csv")

# ===================== Thème & palette =====================
BRAND = {
    "btc":  "#B10967",  # magenta
    "spx":  "#412761",  # violet / indigo
    "tech": "#007078",  # teal
}

plt.rcParams.update({
    "axes.facecolor":       "#0E1117",
    "figure.facecolor":     "#0E1117",
    "axes.edgecolor":       "#C3C7CF",
    "axes.labelcolor":      "#E1E5EA",
    "text.color":           "#E1E5EA",
    "xtick.color":          "#C3C7CF",
    "ytick.color":          "#C3C7CF",
    "grid.color":           "#2A2F3A",
    "grid.alpha":           0.35,
    "axes.grid":            True,
    "axes.grid.which":      "both",
    "legend.frameon":       False,
    "lines.linewidth":      1.6,
})

def _fmt_compact(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{x:.0f}"

# ===================== Helpers data =====================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def base100_from_price(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").astype(float).dropna()
    if p.empty:
        return p
    return (p / p.iloc[0]) * 100.0

def base100_from_returns(r: pd.Series) -> pd.Series:
    r = pd.to_numeric(r, errors="coerce").astype(float).fillna(0.0)
    return (1.0 + r).cumprod() * 100.0

def drawdown_from_returns(r: pd.Series) -> pd.Series:
    eq = (1.0 + pd.to_numeric(r, errors="coerce").fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0

def ols_summary(df: pd.DataFrame, ycol: str, xcol: str):
    sub = df[[ycol, xcol]].dropna()
    if sub.shape[0] <= 50:
        return None
    X = sm.add_constant(sub[xcol])
    y = sub[ycol]
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

# ===================== App =====================
st.set_page_config(page_title="BTC vs Marchés — Risk Dashboard", layout="wide")
st.title("BTC vs S&P500 & Tech — Risk & Correlation Dashboard")

df = load_data(DATA_PATH)

# Filtres de période
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    start = st.date_input("Début", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end = st.date_input("Fin", value=max_date, min_value=min_date, max_value=max_date)
with col3:
    win_vol = st.number_input("Fenêtre Vol (jours)", min_value=10, max_value=252, value=30, step=5)
with col4:
    win_corr = st.number_input("Fenêtre Corr (jours)", min_value=20, max_value=252, value=90, step=5)

mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
d = df.loc[mask].copy()

# Volatilités roulantes (annualisées)
for a in ["btc", "spx", "tech"]:
    d[f"vol_{a}"] = d[f"ret_{a}"].rolling(int(win_vol)).std(ddof=1) * np.sqrt(TRADING_DAYS)

# Corrélations roulantes
d["corr_btc_spx"]  = d["ret_btc"].rolling(int(win_corr)).corr(d["ret_spx"])
d["corr_btc_tech"] = d["ret_btc"].rolling(int(win_corr)).corr(d["ret_tech"])

# ==================== Paramètres risque / CAPM ====================
st.markdown("### Paramètres de risque / CAPM")

c1, c2, c3, c4 = st.columns([1.1, 1.3, 1.3, 1.3])
with c1:
    rf_choice = st.selectbox(
        "Taux sans risque (annuel)",
        options=["0.00%", "2.00%", "Custom"],
        index=1
    )
    if rf_choice == "Custom":
        rf_ann = st.number_input("rf annuel (%)", value=2.00, step=0.25, format="%.2f")/100.0
    else:
        rf_ann = float(rf_choice.strip("%"))/100.0

with c2:
    mu_method = st.selectbox(
        "Méthode μ",
        options=[
            "Réel (moyenne empirique)",
            "CAPM (pas d'alpha) — μ_mkt empirique",
            "CAPM (pas d'alpha) — μ_mkt = rf + prime"
        ],
        index=0
    )

with c3:
    use_ann_sample = st.radio(
        "Vol sample à afficher",
        options=["Quotidienne", "Annualisée"],
        horizontal=True,
        index=1
    )

with c4:
    conf_choice = st.selectbox(
        "Niveau de confiance pour z",
        options=["80%", "90%", "95%", "97.5%", "99%", "z personnalisé"],
        index=2
    )
    if conf_choice == "z personnalisé":
        z_value = st.number_input("z", value=Z_95, step=0.1, format="%.2f")
    else:
        conf_map = {"80%":0.80,"90%":0.90,"95%":0.95,"97.5%":0.975,"99%":0.99}
        z_value = z_from_conf(conf_map[conf_choice], two_sided=True)

# Prime marché si CAPM (prime)
if mu_method.endswith("prime"):
    prime_ann = st.number_input("Prime de risque du marché (annuelle, %)", value=5.00, step=0.25, format="%.2f")/100.0
else:
    prime_ann = None

# ==================== Stats annualisées / CAPM ====================
p_btc: AnnParams  = annualize_mean_vol(d["ret_btc"])
p_spx: AnnParams  = annualize_mean_vol(d["ret_spx"])
p_tech: AnnParams = annualize_mean_vol(d["ret_tech"])

# μ selon la méthode choisie
if mu_method == "Réel (moyenne empirique)":
    mu_btc, mu_spx, mu_tech = p_btc.mu_ann, p_spx.mu_ann, p_tech.mu_ann

elif mu_method == "CAPM (pas d'alpha) — μ_mkt empirique":
    mu_spx_mkt = p_spx.mu_ann  # marché = S&P empirique
    mu_btc, beta_btc   = capm_mu_ann_from_series(d["ret_btc"],  d["ret_spx"], rf_ann, mu_spx_mkt)
    mu_tech, beta_tech = capm_mu_ann_from_series(d["ret_tech"], d["ret_spx"], rf_ann, mu_spx_mkt)
    mu_spx = mu_spx_mkt
else:
    mu_mkt_capm = mu_ann_from_premium(rf_ann, prime_ann or 0.0)
    rf_daily = annual_to_daily_rate(rf_ann)
    beta_btc  = capm_beta(d["ret_btc"],  d["ret_spx"], rf_daily=rf_daily)
    beta_tech = capm_beta(d["ret_tech"], d["ret_spx"], rf_daily=rf_daily)
    mu_btc  = mu_ann_from_premium(rf_ann, (beta_btc  or np.nan) * (mu_mkt_capm - rf_ann))
    mu_tech = mu_ann_from_premium(rf_ann, (beta_tech or np.nan) * (mu_mkt_capm - rf_ann))
    mu_spx  = mu_mkt_capm

# ==================== KPIs ====================
k1,k2,k3,k4,k5,k6 = st.columns(6)
with k1:
    st.metric("Vol BTC (annual.)", f"{d['vol_btc'].dropna().iloc[-1]:.1%}" if d['vol_btc'].notna().any() else "N/A")
with k2:
    st.metric("Vol S&P (annual.)", f"{d['vol_spx'].dropna().iloc[-1]:.1%}" if d['vol_spx'].notna().any() else "N/A")
with k3:
    st.metric("Vol Tech (annual.)", f"{d['vol_tech'].dropna().iloc[-1]:.1%}" if d['vol_tech'].notna().any() else "N/A")
with k4:
    st.metric("Max Drawdown BTC", f"{drawdown_from_returns(d['ret_btc']).min():.0%}")
with k5:
    st.metric("Max Drawdown S&P500", f"{drawdown_from_returns(d['ret_spx']).min():.0%}")
with k6:
    st.metric("Max Drawdown Tech (QQQ)", f"{drawdown_from_returns(d['ret_tech']).min():.0%}")

# Vol échantillon (quotidienne / annualisée)
disp_ann = (use_ann_sample == "Annualisée")
s_btc  = sample_vol(d["ret_btc"],  annualized=disp_ann)
s_spx  = sample_vol(d["ret_spx"],  annualized=disp_ann)
s_tech = sample_vol(d["ret_tech"], annualized=disp_ann)

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    st.metric(f"Vol sample BTC ({'ann.' if disp_ann else 'j'})", f"{s_btc:.2%}")
with c2:
    st.metric(f"Vol sample S&P ({'ann.' if disp_ann else 'j'})", f"{s_spx:.2%}")
with c3:
    st.metric(f"Vol sample Tech ({'ann.' if disp_ann else 'j'})", f"{s_tech:.2%}")

# Horizons critiques t*
t_btc  = dt_critical(mu_btc,  p_btc.sigma_ann,  rf_ann, z=z_value)
t_spx  = dt_critical(mu_spx,  p_spx.sigma_ann,  rf_ann, z=z_value)
t_tech = dt_critical(mu_tech, p_tech.sigma_ann, rf_ann, z=z_value)

with c4:
    st.metric("dt* BTC (> r_f)",  f"{t_btc:.2f} ans"  if np.isfinite(t_btc)  else "∞")
with c5:
    st.metric("dt* S&P (> r_f)",  f"{t_spx:.2f} ans"  if np.isfinite(t_spx)  else "∞")
with c6:
    st.metric("dt* Tech (> r_f)", f"{t_tech:.2f} ans" if np.isfinite(t_tech) else "∞")

st.divider()

# ==================== Tabs ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Volatilité", "Corrélation", "Base 100", "Drawdown",
     "OLS: BTC ~ S&P500", "OLS: BTC ~ Tech",
     "Temps > r_f & CAPM", "Test variances (Fisher)"]
)

# -------- Volatilité --------
with tab1:
    st.subheader("Volatilité annualisée")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(d["Date"], d["vol_btc"],  label="BTC",        color=BRAND["btc"])
    ax.plot(d["Date"], d["vol_spx"],  label="S&P500",     color=BRAND["spx"])
    ax.plot(d["Date"], d["vol_tech"], label="Tech (QQQ)", color=BRAND["tech"])
    ax.set_ylabel("Vol annualisée")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -------- Corrélation --------
with tab2:
    st.subheader("Corrélation roulante BTC vs marchés")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(d["Date"], d["corr_btc_spx"],  label="BTC ~ S&P500", linewidth=1.5, color=BRAND["spx"])
    ax.plot(d["Date"], d["corr_btc_tech"], label="BTC ~ Tech (QQQ)", linewidth=1.5, color=BRAND["tech"])
    ax.axhline(0.0, linestyle="--", linewidth=1, color="#657089", alpha=0.7)
    ax.set_ylabel("Corrélation (fenêtre)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -------- Base 100 --------
with tab3:
    st.subheader("Évolution comparée")

    # Modes plus riches pour éviter l’effet “plat”
    mode = st.radio(
        "Échelle / Méthode",
        [
            "Base 100 (log)",
            "Base 100 (linéaire)",
            "Base 100 (linéaire, axes séparés)",
            "Base 100 (linéaire, min-max 0–100)",
            "Risque normalisé (vol cible)"
        ],
        index=0, horizontal=True
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

    def _style_axes(ax, ylabel):
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_compact))
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.35)

    if mode == "Risque normalisé (vol cible)":
        def target_vol_index(r: pd.Series, window: int = 30, target_ann_vol: float = 0.20) -> pd.Series:
            vol_ann = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
            scale = target_ann_vol / vol_ann.replace(0, np.nan)
            r_scaled = (pd.to_numeric(r, errors="coerce") * scale).fillna(0.0)
            return (1.0 + r_scaled).cumprod() * 100.0

        idx = pd.DataFrame({
            "Date": d["Date"],
            "BTC":        target_vol_index(d["ret_btc"]),
            "S&P500":     target_vol_index(d["ret_spx"]),
            "Tech (QQQ)": target_vol_index(d["ret_tech"]),
        }).dropna()

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(idx["Date"], idx["BTC"],        label="BTC (vol cible)",        color=BRAND["btc"])
        ax.plot(idx["Date"], idx["S&P500"],     label="S&P500 (vol cible)",     color=BRAND["spx"])
        ax.plot(idx["Date"], idx["Tech (QQQ)"], label="Tech (QQQ) (vol cible)", color=BRAND["tech"])
        _style_axes(ax, "Indice (vol cible 20%)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    elif mode == "Base 100 (log)":
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(idx_lin["Date"], idx_lin["BTC"],        label="BTC (base 100)",        color=BRAND["btc"])
        ax.plot(idx_lin["Date"], idx_lin["S&P500"],     label="S&P500 (base 100)",     color=BRAND["spx"])
        ax.plot(idx_lin["Date"], idx_lin["Tech (QQQ)"], label="Tech (QQQ) (base 100)", color=BRAND["tech"])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.set_ylabel("Indice base 100 (log)")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    elif mode == "Base 100 (linéaire)":
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(idx_lin["Date"], idx_lin["BTC"],        label="BTC (base 100)",        color=BRAND["btc"])
        ax.plot(idx_lin["Date"], idx_lin["S&P500"],     label="S&P500 (base 100)",     color=BRAND["spx"])
        ax.plot(idx_lin["Date"], idx_lin["Tech (QQQ)"], label="Tech (QQQ) (base 100)", color=BRAND["tech"])
        _style_axes(ax, "Indice base 100")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    elif mode == "Base 100 (linéaire, axes séparés)":
        fig, ax1 = plt.subplots(figsize=(10, 4.8))
        ax1.plot(idx_lin["Date"], idx_lin["S&P500"],     label="S&P500",     color=BRAND["spx"])
        ax1.plot(idx_lin["Date"], idx_lin["Tech (QQQ)"], label="Tech (QQQ)", color=BRAND["tech"])
        _style_axes(ax1, "Indice base 100 (S&P/Tech)")

        ax2 = ax1.twinx()
        ax2.plot(idx_lin["Date"], idx_lin["BTC"], label="BTC", color=BRAND["btc"], alpha=0.9)
        ax2.set_ylabel("Indice base 100 (BTC)")
        ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_compact))

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")
        st.pyplot(fig, clear_figure=True)

    elif mode == "Base 100 (linéaire, min-max 0–100)":
        mm = idx_lin.copy()
        for col in ["BTC", "S&P500", "Tech (QQQ)"]:
            x = mm[col].values
            lo, hi = np.nanmin(x), np.nanmax(x)
            mm[col] = 100.0 * (x - lo) / (hi - lo) if hi > lo else 0.0

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(mm["Date"], mm["BTC"],        label="BTC (min-max)",        color=BRAND["btc"])
        ax.plot(mm["Date"], mm["S&P500"],     label="S&P500 (min-max)",     color=BRAND["spx"])
        ax.plot(mm["Date"], mm["Tech (QQQ)"], label="Tech (QQQ) (min-max)", color=BRAND["tech"])
        ax.set_ylabel("Échelle normalisée (0–100)")
        ax.set_ylim(-3, 103)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

# -------- Drawdown --------
with tab4:
    st.subheader("Drawdown cumulatif")
    dd = pd.DataFrame({
        "Date":  d["Date"],
        "BTC":   drawdown_from_returns(d["ret_btc"]),
        "S&P500":drawdown_from_returns(d["ret_spx"]),
        "Tech":  drawdown_from_returns(d["ret_tech"]),
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(dd["Date"], dd["BTC"],   label="BTC",        color=BRAND["btc"])
    ax.plot(dd["Date"], dd["S&P500"],label="S&P500",     color=BRAND["spx"])
    ax.plot(dd["Date"], dd["Tech"],  label="Tech (QQQ)", color=BRAND["tech"])
    ax.set_ylabel("Drawdown")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# -------- OLS BTC ~ S&P --------
with tab5:
    st.subheader("Scatter & OLS — BTC ~ S&P500")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(d["ret_spx"], d["ret_btc"], s=8, alpha=0.6, color=BRAND["btc"])
    ax.set_xlabel("r_SPX"); ax.set_ylabel("r_BTC")
    st.pyplot(fig, clear_figure=True)
    m = ols_summary(d, "ret_btc", "ret_spx")
    st.code(m.summary().as_text() if m is not None else "Pas assez de points.")

# -------- OLS BTC ~ Tech --------
with tab6:
    st.subheader("Scatter & OLS — BTC ~ Tech (QQQ)")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(d["ret_tech"], d["ret_btc"], s=8, alpha=0.6, color=BRAND["btc"])
    ax.set_xlabel("r_Tech (QQQ)"); ax.set_ylabel("r_BTC")
    st.pyplot(fig, clear_figure=True)
    m = ols_summary(d, "ret_btc", "ret_tech")
    st.code(m.summary().as_text() if m is not None else "Pas assez de points.")

# -------- Temps > r_f & CAPM --------
with tab7:
    st.subheader("Temps minimal pour battre le taux sans risque")

    st.caption(
        f"Taux sans risque annuel: **{rf_ann:.2%}** — Méthode μ: **{mu_method}** — z = **{z_value:.2f}** (bilatéral)."
    )
    st.markdown(
        f"""
        • **BTC**: μ={mu_btc:.2%}, σ={p_btc.sigma_ann:.2%} → **t\*** = {('∞' if not np.isfinite(t_btc) else f'{t_btc:.2f} ans')}  
        • **S&P500**: μ={mu_spx:.2%}, σ={p_spx.sigma_ann:.2%} → **t\*** = {('∞' if not np.isfinite(t_spx) else f'{t_spx:.2f} ans')}  
        • **Tech (QQQ)**: μ={mu_tech:.2%}, σ={p_tech.sigma_ann:.2%} → **t\*** = {('∞' if not np.isfinite(t_tech) else f'{t_tech:.2f} ans')}
        """
    )
    st.caption("Formule : t* = ( z · σ / (μ − r_f) )², z au niveau choisi, μ et σ annualisés ; t* en années.")

    st.markdown("**Paramètres utilisés (annualisés)**")
    params_df = pd.DataFrame({
        "Actif": ["BTC","S&P500","Tech (QQQ)"],
        "μ_ann": [mu_btc, mu_spx, mu_tech],
        "σ_ann (sample)": [p_btc.sigma_ann, p_spx.sigma_ann, p_tech.sigma_ann],
        "t* (ans)": [t_btc, t_spx, t_tech],
    })
    st.dataframe(
        params_df.style.format({"μ_ann":"{:.2%}","σ_ann (sample)":"{:.2%}","t* (ans)":"{:.2f}"}),
        use_container_width=True
    )

# -------- Fisher --------
with tab8:
    st.subheader("Test d’égalité des variances (Fisher) — rendements quotidiens")
    rows = []
    for (label, a, b) in [
        ("BTC vs S&P",  d["ret_btc"],  d["ret_spx"]),
        ("BTC vs Tech", d["ret_btc"],  d["ret_tech"]),
        ("S&P vs Tech", d["ret_spx"],  d["ret_tech"]),
    ]:
        F, p, df1, df2 = fisher_variance_test(a, b)
        rows.append((label, F, p, df1, df2))
    out = pd.DataFrame(rows, columns=["Paire","F","p-value","df1","df2"])
    st.dataframe(out.style.format({"F":"{:.4f}","p-value":"{:.2e}"}), use_container_width=True)
    st.caption("H0: variances égales. Rejet si p<0.05 (bilatéral).")

# ========= Note explicative vol sample =========
st.markdown(
    "<small>**Note** — *Vol sample (quotidienne)* = écart-type des rendements journaliers sur la période filtrée. "
    "*Vol sample (annualisée)* = vol quotidienne × √252. Les KPIs de tête sont des **volatilités roulantes annualisées** sur la fenêtre choisie.*</small>",
    unsafe_allow_html=True
)
