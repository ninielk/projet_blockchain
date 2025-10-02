# src/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass

# --- constantes de marché ---
TRADING_DAYS: int = 252
Z_95: float = 1.96
_DDOF = 1
_EPS = 1e-12


# -------------------- data classes --------------------
@dataclass
class AnnParams:
    """Paramètres annualisés dérivés d'une série de rendements quotidiens."""
    mu_ann: float        # moyenne annualisée (mean_daily * 252)
    sigma_ann: float     # vol annualisée (std_daily * sqrt(252))
    sigma_sample: float  # vol quotidienne (std_daily)


# -------------------- utilitaires de base --------------------
def _clean_series(r: pd.Series) -> pd.Series:
    """Cast -> float, drop NaN."""
    return pd.to_numeric(r, errors="coerce").astype(float).dropna()


def daily_to_annual_mean(mu_daily: float) -> float:
    return float(mu_daily) * TRADING_DAYS


def annual_to_daily_rate(rate_ann: float) -> float:
    """Convertit un taux annualisé en équivalent *quotidien simple* (additif)."""
    return float(rate_ann) / TRADING_DAYS


def daily_to_annual_vol(sig_daily: float) -> float:
    return float(sig_daily) * np.sqrt(TRADING_DAYS)


def z_from_conf(conf: float, two_sided: bool = False) -> float:
    """
    conf in (0,1). 0.95 -> 1.96 (bilat), 1.645 (unilat).
    """
    import scipy.stats as st  # lazy import
    conf = float(conf)
    conf = np.clip(conf, _EPS, 1 - _EPS)
    if two_sided:
        alpha = 1 - conf
        return float(st.norm.ppf(1 - alpha / 2))
    else:
        alpha = 1 - conf
        return float(st.norm.ppf(1 - alpha))


# -------------------- stats annualisées à partir d'une série --------------------
def annualize_mean_vol(r: pd.Series) -> AnnParams:
    """
    r : rendements quotidiens.
    Retourne (mu_ann, sigma_ann, sigma_sample).
    """
    r = _clean_series(r)
    if r.empty:
        return AnnParams(np.nan, np.nan, np.nan)

    mu_daily = r.mean()
    sig_daily = r.std(ddof=_DDOF)

    mu_ann = daily_to_annual_mean(mu_daily)
    sigma_ann = daily_to_annual_vol(sig_daily)

    return AnnParams(mu_ann=float(mu_ann),
                     sigma_ann=float(sigma_ann),
                     sigma_sample=float(sig_daily))


def sample_vol(r: pd.Series, annualized: bool = False) -> float:
    r = _clean_series(r)
    if r.empty:
        return np.nan
    s = r.std(ddof=_DDOF)
    return float(daily_to_annual_vol(s) if annualized else s)


def sample_mean(r: pd.Series, annualized: bool = True) -> float:
    r = _clean_series(r)
    if r.empty:
        return np.nan
    m = r.mean()
    return float(daily_to_annual_mean(m) if annualized else m)


# -------------------- CAPM --------------------
def capm_beta(asset_r: pd.Series, mkt_r: pd.Series, rf_daily: float = 0.0) -> float:
    """
    β via régression OLS sur rendements EXCESS (r - rf_daily).
    """
    x = _clean_series(mkt_r)
    y = _clean_series(asset_r)
    idx = x.index.intersection(y.index)
    if len(idx) < 30:
        return np.nan

    x_ex = x.loc[idx] - rf_daily
    y_ex = y.loc[idx] - rf_daily
    X = sm.add_constant(x_ex.values)
    model = sm.OLS(y_ex.values, X, missing="drop").fit()
    if model.params.shape[0] < 2:
        return np.nan
    return float(model.params[1])  # slope = beta


def capm_mu_ann(beta: float, mu_mkt_ann: float, rf_ann: float) -> float:
    """μ_ann = rf + β(μ_mkt - rf)."""
    if not np.isfinite(beta):
        return np.nan
    return float(rf_ann + beta * (mu_mkt_ann - rf_ann))


def capm_mu_ann_from_series(asset_r: pd.Series,
                            mkt_r: pd.Series,
                            rf_ann: float,
                            mu_mkt_ann: float | None = None) -> tuple[float, float]:
    """
    Calcule β sur series quotidiennes, puis μ_ann CAPM (sans alpha).
    Si mu_mkt_ann n'est pas fourni, il est estimé empiriquement sur mkt_r.
    Retourne (mu_ann_capm, beta).
    """
    rf_daily = annual_to_daily_rate(rf_ann)
    beta = capm_beta(asset_r, mkt_r, rf_daily=rf_daily)
    if mu_mkt_ann is None:
        mu_mkt_ann = sample_mean(mkt_r, annualized=True)
    mu_capm = capm_mu_ann(beta, mu_mkt_ann, rf_ann)
    return float(mu_capm), float(beta)


def mu_ann_from_premium(rf_ann: float, risk_premium_ann: float) -> float:
    """μ_ann = rf_ann + prime_annuelle."""
    return float(rf_ann + risk_premium_ann)


# -------------------- horizon critique t* --------------------
def dt_critical(mu_ann: float,
                sigma_ann: float,
                rf_ann: float,
                z: float = Z_95) -> float:
    """
    t* = ( z * sigma_ann / (mu_ann - rf_ann) )^2  (en années).
    Si mu <= rf -> inf.
    """
    mu_ann = float(mu_ann)
    sigma_ann = float(sigma_ann)
    rf_ann = float(rf_ann)
    z = float(z)

    if not np.isfinite(mu_ann) or not np.isfinite(sigma_ann) or sigma_ann <= 0:
        return np.nan
    excess = mu_ann - rf_ann
    if not np.isfinite(excess) or excess <= 0:
        return np.inf
    return float((z * sigma_ann / excess) ** 2)


# -------------------- tests d'égalité des variances --------------------
def fisher_variance_test(x: pd.Series, y: pd.Series):
    """
    Test F bilatéral d'égalité des variances.
    Retourne (F, p_value, df1, df2) avec F >= 1.
    """
    import scipy.stats as st  # nécessite scipy

    x = _clean_series(x)
    y = _clean_series(y)
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan, n1 - 1, n2 - 1

    s1 = x.var(ddof=_DDOF)
    s2 = y.var(ddof=_DDOF)
    if s1 >= s2:
        F, df1, df2 = s1 / (s2 + _EPS), n1 - 1, n2 - 1
    else:
        F, df1, df2 = s2 / (s1 + _EPS), n2 - 1, n1 - 1

    # p-value bilatérale
    p_upper = 1 - st.f.cdf(F, df1, df2)
    p_two = min(1.0, 2 * min(p_upper, st.f.cdf(1 / F, df1, df2)))
    return float(F), float(p_two), int(df1), int(df2)
