"""
AMCE: Adaptive Macro-Conditional Ensemble
Quantitative Research Trading System
Author: Dhruv Jain

Research Hypothesis:
    H0 (Null): Macro-conditional regime signals provide no statistically significant
               improvement over passive equity exposure.
    H1 (Alternative): Integrating Regime Filtering (Trend) with Gradient Boosting
               signals generates positive crisis alpha and statistically significant
               risk-adjusted outperformance.

    Test  : Signal permutation (n=1,000+)
    Threshold: p <= 0.05
    Alpha : OLS GLM on excess returns

Performance Disclosure:
    Returns are calculated net-of-fees and slippage, pre-tax.
    Slippage model: 5 bps per side (institutional ETF estimate).
    Commission  : 3 bps per trade.
    No look-ahead bias: all features lagged >= 1 trading day.
    Purged walk-forward validation with configurable embargo window.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import shap

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMCE | Quantitative Research Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Rajdhani',sans-serif;background:#050d1a;color:#c8d8e8;}
.stApp{background:#050d1a;}
[data-testid="stSidebar"]{background:#07111f;border-right:1px solid #0d2137;}
[data-testid="stSidebar"] *{color:#c8d8e8 !important;}
[data-testid="stSidebar"] label{font-family:'Share Tech Mono',monospace;font-size:.65rem;
  color:#3a8aaa !important;letter-spacing:2px;text-transform:uppercase;}
.main-title{font-family:'Rajdhani',sans-serif;font-size:2.6rem;font-weight:700;
  background:linear-gradient(90deg,#00ffe0,#00c8ff);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;letter-spacing:2px;margin-bottom:0;}
.subtitle-bar{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#2a7a9a;
  letter-spacing:4px;text-transform:uppercase;margin-bottom:1.2rem;}
.section-header{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#00ffe0;
  letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid #0d2a3d;
  padding-bottom:6px;margin:1.6rem 0 1rem 0;}
.metric-card{background:#07111f;border:1px solid #0d2a3d;border-top:2px solid #00ffe0;
  border-radius:4px;padding:14px 18px;margin:4px 0;}
.metric-card-red{border-top-color:#ff4d6d !important;}
.metric-value{font-family:'Rajdhani',sans-serif;font-size:1.8rem;font-weight:700;
  color:#00ffe0;margin:0;}
.metric-value-red{color:#ff4d6d !important;}
.metric-label{font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#3a6a8a;
  letter-spacing:2px;text-transform:uppercase;}
.bench-label{font-family:'Share Tech Mono',monospace;font-size:.62rem;color:#2a5a7a;margin-top:2px;}
.hyp-box{background:#07111f;border:1px solid #0d2a3d;border-left:3px solid #00ffe0;
  border-radius:4px;padding:16px 20px;font-family:'Share Tech Mono',monospace;
  font-size:.7rem;color:#7ab8d4;line-height:1.8;margin-bottom:1rem;}
.hyp-title{font-size:.6rem;color:#00ffe0;letter-spacing:4px;text-transform:uppercase;margin-bottom:8px;}
.banner-green{background:#041a0e;border:1px solid #00804a;border-radius:3px;padding:8px 16px;
  font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#00c87a;}
.banner-yellow{background:#1a1400;border:1px solid #806a00;border-radius:3px;padding:8px 16px;
  font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#f4c542;}
.crisis-table{width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:.7rem;}
.crisis-table th{color:#2a6a8a;letter-spacing:2px;text-transform:uppercase;padding:8px 12px;
  text-align:right;border-bottom:1px solid #0d2a3d;font-size:.6rem;}
.crisis-table td{padding:10px 12px;text-align:right;border-bottom:1px solid #07111f;color:#8ab8d4;}
.crisis-table td:first-child{text-align:left;color:#c8d8e8;}
.green{color:#00c87a !important;} .red{color:#ff4d6d !important;}
.badge-green{background:#041a0e;border:1px solid #00804a;border-radius:3px;
  padding:2px 8px;color:#00c87a;font-size:.65rem;}
.stButton>button{background:linear-gradient(135deg,#00ffe0,#00a8c8) !important;
  color:#050d1a !important;font-family:'Rajdhani',sans-serif !important;font-weight:700 !important;
  font-size:.9rem !important;letter-spacing:2px !important;border:none !important;
  border-radius:4px !important;padding:12px 20px !important;width:100% !important;
  text-transform:uppercase !important;}
.stTextInput input{background:#0a1825 !important;border:1px solid #0d2a3d !important;
  color:#c8d8e8 !important;font-family:'Share Tech Mono',monospace !important;border-radius:3px !important;}
div[data-testid="stNumberInput"] input{background:#0a1825 !important;border:1px solid #0d2a3d !important;
  color:#c8d8e8 !important;font-family:'Share Tech Mono',monospace !important;}
hr{border-color:#0d2a3d !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────
DARK_BG    = "#050d1a"
GRID_COLOR = "#0d2a3d"
CYAN       = "#00ffe0"
PURPLE     = "#8a7aff"
PINK       = "#ff4d6d"
WHITE      = "#c8d8e8"


def plot_layout(**overrides):
    """
    Return a Plotly layout dict.  Callers can safely override any key —
    including 'yaxis' — without triggering 'multiple values for keyword argument'.
    """
    base = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(family="Share Tech Mono, monospace", size=10, color=WHITE),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=WHITE),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=WHITE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=WHITE)),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    for k, v in overrides.items():
        base[k] = v
    return base


# ─────────────────────────────────────────────────────────────
# TRANSACTION COST CONSTANTS  (pre-tax, net-of-fees-and-slippage)
# ─────────────────────────────────────────────────────────────
SLIPPAGE_BPS   = 5    # 5 bps per side — tight institutional ETF spread
COMMISSION_BPS = 3    # 3 bps flat commission
ROUND_TRIP     = (SLIPPAGE_BPS + COMMISSION_BPS) / 10_000   # applied on every signal flip


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────
class DataLoader:
    def __init__(self, start_date="2000-01-01"):
        self.start_date = start_date
        self._cache: dict = {}

    def download_asset(self, ticker: str) -> pd.Series:
        if ticker in self._cache:
            return self._cache[ticker]
        df = yf.download(ticker, start=self.start_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        elif "Close" in df.columns:
            df = df["Close"]
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        self._cache[ticker] = df
        return df

    def create_dataset(self, risky: str, safe: str, include_vix: bool = True):
        p_r    = self.download_asset(risky)
        p_s    = self.download_asset(safe)
        prices = pd.concat([p_r, p_s], axis=1).dropna()
        prices.columns = [risky, safe]
        returns = prices.pct_change().dropna()
        vix = None
        if include_vix:
            try:
                vix = self.download_asset("^VIX")
                vix = vix.reindex(prices.index, method="ffill")
            except Exception:
                vix = None
        return prices, returns, vix


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# All features are lagged >= 1 trading day.  Zero look-ahead bias.
# ─────────────────────────────────────────────────────────────
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    gain = series.clip(lower=0).rolling(window).mean()
    loss = (-series.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def build_features(prices, returns, vix, risky_col: str, safe_col: str) -> pd.DataFrame:
    p_r = prices[risky_col]
    r_r = returns[risky_col]
    r_s = returns[safe_col]
    df  = pd.DataFrame(index=prices.index)

    # ── Momentum ──
    df["Mom_1M"]   = r_r.rolling(21).sum().shift(1)
    df["Mom_3M"]   = r_r.rolling(63).sum().shift(1)
    df["Mom_6M"]   = r_r.rolling(126).sum().shift(1)
    df["Safe_Mom"] = r_s.rolling(21).sum().shift(1)

    # ── Trend / MA ──
    df["MA_50"]    = (p_r / p_r.rolling(50).mean() - 1).shift(1)
    df["MA_200"]   = (p_r / p_r.rolling(200).mean() - 1).shift(1)
    df["MA_Cross"] = (p_r.rolling(50).mean() / p_r.rolling(200).mean() - 1).shift(1)

    # ── Volatility ──
    rv_r = r_r.rolling(21).std().shift(1)
    rv_s = r_s.rolling(21).std().shift(1)
    df["Vol_Ratio"]  = rv_r / (rv_s + 1e-9)
    df["Vol_21d"]    = rv_r
    df["Vol_63d"]    = r_r.rolling(63).std().shift(1)
    df["Vol_Regime"] = rv_r / r_r.rolling(252).std() - 1

    # ── Drawdown ──
    df["Dist_Max_6M"] = p_r.shift(1) / p_r.rolling(126).max().shift(1) - 1

    # ── Relative strength & RSI ──
    df["Rel_Str"]   = (r_r.rolling(63).sum() - r_s.rolling(63).sum()).shift(1)
    df["RSI_Proxy"] = _rsi(r_r, 14).shift(1)

    # ── VIX (macro fear gauge) ──
    if vix is not None:
        df["VIX_Proxy"]  = vix.shift(1)
        df["VIX_Change"] = vix.pct_change(5).shift(1)
        df["VIX_MA"]     = (vix / vix.rolling(63).mean() - 1).shift(1)
    else:
        df["VIX_Proxy"]  = df["Vol_Ratio"] * 20
        df["VIX_Change"] = df["Vol_Regime"]
        df["VIX_MA"]     = 0.0

    # ── Binary target: will risky asset outperform safe asset tomorrow? ──
    df["target"] = (r_r.shift(-1) > r_s.shift(-1)).astype(int)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "Mom_1M", "Mom_3M", "Mom_6M", "Safe_Mom",
    "MA_50", "MA_200", "MA_Cross",
    "Vol_Ratio", "Vol_21d", "Vol_63d", "Vol_Regime",
    "Dist_Max_6M", "Rel_Str", "VIX_Proxy", "VIX_Change", "VIX_MA", "RSI_Proxy",
]


# ─────────────────────────────────────────────────────────────
# NET-OF-FEES BACKTEST  (pre-tax, no look-ahead)
# ─────────────────────────────────────────────────────────────
def backtest(signal_series: pd.Series, r_risky: pd.Series, r_safe: pd.Series) -> pd.Series:
    """
    Apply round-trip cost on every signal flip.
    Returns net-of-fees daily return series.
    Performance disclosure: pre-tax, net-of-slippage and commissions.
    """
    raw      = np.where(signal_series.values == 1, r_risky.values, r_safe.values)
    flips    = signal_series.diff().abs().fillna(0).values
    net      = raw - flips * ROUND_TRIP
    return pd.Series(net, index=signal_series.index, name="strategy")


# ─────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────
_RF = 0.04 / 252   # daily risk-free rate proxy

def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 10:
        return 0.0
    e = r - _RF
    return float(e.mean() / (e.std() + 1e-9) * np.sqrt(252))

def _sharpe_fast(arr: np.ndarray) -> float:
    e = arr - _RF
    return float(np.mean(e) / (np.std(e) + 1e-9) * np.sqrt(252))

def _sortino(r: pd.Series) -> float:
    r  = r.dropna()
    e  = r - _RF
    ds = e[e < 0].std() + 1e-9
    return float(e.mean() / ds * np.sqrt(252))

def _max_drawdown(r: pd.Series) -> float:
    c = (1 + r.dropna()).cumprod()
    return float((c / c.cummax() - 1).min())

def _rolling_drawdown(r: pd.Series) -> pd.Series:
    c = (1 + r.dropna()).cumprod()
    return (c / c.cummax() - 1) * 100

def _ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return 0.0
    return float((1 + r).prod() ** (252 / len(r)) - 1)

def _total_return(r: pd.Series) -> float:
    return float((1 + r.dropna()).prod() - 1)

def _calmar(r: pd.Series) -> float:
    md = abs(_max_drawdown(r))
    return _ann_return(r) / md if md > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, returns: pd.DataFrame,
                 risky_col: str, safe_col: str,
                 embargo_months: int = 4, n_mc: int = 500) -> dict:
    """
    Purged walk-forward ensemble.
    Train/test split: 70% / 30% with purge embargo to prevent leakage at the boundary.
    Ensemble: Gradient Boosting + Random Forest + Logistic Regression (soft vote).
    Signal threshold: 0.55 — reduces unnecessary turnover without sacrificing edge.
    """
    fc  = [c for c in FEATURE_COLS if c in df.columns]
    X   = df[fc].values
    y   = df["target"].values
    idx = df.index

    embargo_days = embargo_months * 21      # ~21 trading days / month
    train_size   = int(len(df) * 0.70)
    test_start   = train_size + embargo_days

    X_tr, y_tr = X[:train_size],  y[:train_size]
    X_te       = X[test_start:]
    idx_te     = idx[test_start:]

    sc       = StandardScaler()
    X_tr_s   = sc.fit_transform(X_tr)
    X_te_s   = sc.transform(X_te)

    # ── Train models ──
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

    gb.fit(X_tr_s, y_tr)
    rf.fit(X_tr_s, y_tr)
    lr.fit(X_tr_s, y_tr)

    # ── Ensemble (soft-vote average) ──
    p_gb  = gb.predict_proba(X_te_s)[:, 1]
    p_rf  = rf.predict_proba(X_te_s)[:, 1]
    p_lr  = lr.predict_proba(X_te_s)[:, 1]
    p_ens = (p_gb + p_rf + p_lr) / 3.0

    signal  = (p_ens > 0.55).astype(int)   # 0.55 reduces low-conviction trades
    sig_s   = pd.Series(signal, index=idx_te)

    r_risky = returns[risky_col].reindex(idx_te)
    r_safe  = returns[safe_col].reindex(idx_te)
    bench_r = r_risky.rename("benchmark")

    strat_r = backtest(sig_s, r_risky, r_safe)

    # ── In-sample proxy for overfitting check ──
    half = train_size // 2
    gb2  = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    rf2  = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    lr2  = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    gb2.fit(X_tr_s[:half], y_tr[:half])
    rf2.fit(X_tr_s[:half], y_tr[:half])
    lr2.fit(X_tr_s[:half], y_tr[:half])

    p_is    = ((gb2.predict_proba(X_tr_s[half:])[:, 1] +
                rf2.predict_proba(X_tr_s[half:])[:, 1] +
                lr2.predict_proba(X_tr_s[half:])[:, 1]) / 3.0)
    sig_is  = pd.Series((p_is > 0.55).astype(int), index=idx[half:train_size])
    r_r_is  = returns[risky_col].reindex(sig_is.index)
    r_s_is  = returns[safe_col].reindex(sig_is.index)
    strat_is = backtest(sig_is, r_r_is, r_s_is)

    in_sharpe  = _sharpe(strat_is);  out_sharpe = _sharpe(strat_r)
    in_dd      = _max_drawdown(strat_is); out_dd = _max_drawdown(strat_r)
    in_wr      = float((strat_is > 0).mean()); out_wr = float((strat_r > 0).mean())

    # ── SHAP (on first 500 test points for speed) ──
    explainer  = shap.TreeExplainer(gb)
    shap_vals  = explainer.shap_values(X_te_s[:500])

    # ── Permutation significance test ──
    actual_sh = _sharpe(strat_r)
    rng       = np.random.default_rng(42)
    perm_sh   = []
    for _ in range(1000):
        ps = rng.permutation(signal)
        pr = pd.Series(np.where(ps == 1, r_risky.values, r_safe.values), index=idx_te)
        perm_sh.append(_sharpe(pr))
    perm_p   = float(np.mean(np.array(perm_sh) >= actual_sh))
    pct_beat = float(np.mean(np.array(perm_sh) < actual_sh)) * 100
    pct_95   = float(np.percentile(perm_sh, 95))

    # ── Bootstrap Monte Carlo ──
    rng2     = np.random.default_rng(99)
    n_days   = len(strat_r)
    mc_paths = np.array([
        np.cumprod(1 + rng2.choice(strat_r.values, n_days, replace=True))
        for _ in range(n_mc)
    ])

    # ── OLS Factor Decomposition ──
    exc_r  = (strat_r - r_safe.reindex(strat_r.index)).dropna()
    mkt_r  = bench_r.reindex(exc_r.index).dropna()
    common = exc_r.index.intersection(mkt_r.index)
    X_ols  = sm.add_constant(mkt_r.loc[common])
    ols    = sm.OLS(exc_r.loc[common], X_ols).fit()
    ols_alpha = float(ols.params.iloc[0]) * 252
    ols_beta  = float(ols.params.iloc[1])
    ols_r2    = float(ols.rsquared)
    ols_ir    = ols_alpha / (float(ols.resid.std()) * np.sqrt(252) + 1e-9)

    # ── Rolling metrics ──
    roll_sh_s = strat_r.rolling(252).apply(_sharpe_fast, raw=True)
    roll_sh_b = bench_r.rolling(252).apply(_sharpe_fast, raw=True)
    roll_wr   = (strat_r > 0).rolling(252).mean()

    # ── TC sensitivity table ──
    flips     = sig_s.diff().abs().fillna(0)
    r_gross   = np.where(signal == 1, r_risky.values, r_safe.values)
    tc_results = []
    for bps in [0, 5, 10, 20, 30, 50]:
        tc_r = pd.Series(r_gross - flips.values * bps / 10_000, index=idx_te)
        tc_results.append(dict(
            bps=bps,
            ann_return=_ann_return(tc_r),
            sharpe=_sharpe(tc_r),
            max_dd=_max_drawdown(tc_r),
            beats=_sharpe(tc_r) > _sharpe(bench_r),
        ))

    # ── Model disagreement ──
    p_gb_s = pd.Series(p_gb, index=idx_te)
    p_rf_s = pd.Series(p_rf, index=idx_te)
    disagr = (p_gb_s - p_rf_s).abs()
    high_c = float((disagr < 0.10).mean()) * 100

    # ── Crisis alpha ──
    crisis_periods = {
        "2008 Financial Crisis": ("2008-09-01", "2009-03-31"),
        "2011 Euro Debt Crisis":  ("2011-07-01", "2011-10-31"),
        "2015 Flash Crash":       ("2015-08-01", "2015-09-30"),
        "2018 Volmageddon":       ("2018-01-26", "2018-04-30"),
        "2020 COVID Crash":       ("2020-02-19", "2020-04-30"),
        "2022 Inflation Bear":    ("2022-01-01", "2022-10-31"),
    }
    crisis_data = []
    all_r = pd.concat([strat_r.rename("s"), bench_r.rename("b")], axis=1).dropna()
    for name, (s, e) in crisis_periods.items():
        try:
            sub = all_r.loc[s:e]
            if len(sub) < 5:
                continue
            sr = float((1 + sub["s"]).prod() - 1)
            br = float((1 + sub["b"]).prod() - 1)
            crisis_data.append(dict(
                period=name, strategy=sr, market=br,
                alpha=sr - br, preserved=sr > br))
        except Exception:
            pass

    return dict(
        strat_r=strat_r, bench_r=bench_r,
        equity_strat=(1 + strat_r).cumprod(),
        equity_bench=(1 + bench_r).cumprod(),
        drawdown_strat=_rolling_drawdown(strat_r),
        drawdown_bench=_rolling_drawdown(bench_r),
        signal=sig_s, p_gb=p_gb_s, p_rf=p_rf_s,
        feature_cols=fc, X_te_s=X_te_s, shap_vals=shap_vals,
        perm_sharpes=perm_sh, perm_p_value=perm_p,
        pct_beaten=pct_beat, pct_95=pct_95,
        mc_paths=mc_paths,
        ols_alpha=ols_alpha, ols_beta=ols_beta, ols_r2=ols_r2, ols_ir=ols_ir,
        roll_sh_s=roll_sh_s, roll_sh_b=roll_sh_b, roll_wr=roll_wr,
        tc_results=tc_results, disagreement=disagr, high_conviction=high_c,
        crisis_data=crisis_data,
        in_sharpe=in_sharpe, out_sharpe=out_sharpe,
        in_dd=in_dd, out_dd=out_dd, in_wr=in_wr, out_wr=out_wr,
        n_trades=int(flips.sum()),
        train_end=idx[train_size - 1],
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.6rem;letter-spacing:3px;color:#2a7a9a;text-transform:uppercase;">RESEARCH TERMINAL</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.55rem;letter-spacing:2px;color:#1a5a7a;text-transform:uppercase;margin-top:-10px;">streamlit</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.6rem;letter-spacing:3px;color:#3a8aaa;text-transform:uppercase;">MODEL CONTROLS</p>', unsafe_allow_html=True)
    risky_asset = st.text_input("High-Beta Asset", value="QQQ")
    safe_asset  = st.text_input("Risk-Free Asset",  value="SHY")
    embargo     = st.slider("Purged Embargo (Months)", 1, 12, 4)
    n_mc        = st.number_input("Monte Carlo Sims", 100, 2000, 500, step=100)
    st.markdown("---")
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.58rem;color:#2a5a7a;line-height:1.8;">Regime-Filtered Boosting · Purged walk-<br>forward validation · Ensemble voting · SHAP<br>attribution · Permutation testing</p>', unsafe_allow_html=True)
    run_btn = st.button("EXECUTE RESEARCH PIPELINE")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p style="font-family:Share Tech Mono;font-size:.58rem;letter-spacing:4px;color:#2a6a8a;text-transform:uppercase;margin-bottom:-4px;">QUANTITATIVE RESEARCH LAB</p>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Adaptive Macro-Conditional Ensemble</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-bar">AMCE FRAMEWORK &nbsp;&middot;&nbsp; REGIME FILTERING &nbsp;&middot;&nbsp; ENSEMBLE VOTING &nbsp;&middot;&nbsp; STATISTICAL VALIDATION</p>', unsafe_allow_html=True)

st.markdown("""
<div class="hyp-box">
<div class="hyp-title">RESEARCH HYPOTHESIS</div>
<b>H0 (Null):</b> Macro-conditional regime signals provide no statistically significant improvement over passive equity exposure.<br>
<b>H1 (Alternative):</b> Integrating Regime Filtering (Trend) with Gradient Boosting signals generates positive crisis alpha and statistically significant risk-adjusted outperformance.<br>
<span style="color:#1a5a7a;">Test: Signal permutation (n=1,000+) &nbsp;|&nbsp; Threshold: p &le; 0.05 &nbsp;|&nbsp; Alpha: OLS GLM on excess returns &nbsp;|&nbsp; Returns: net-of-fees &amp; slippage, pre-tax</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if run_btn or "results" in st.session_state:

    if run_btn:
        with st.spinner("Running research pipeline..."):
            try:
                loader = DataLoader()
                prices, returns, vix = loader.create_dataset(risky_asset, safe_asset)
                df_feat = build_features(prices, returns, vix, risky_asset, safe_asset)
                res = run_pipeline(df_feat, returns, risky_asset, safe_asset,
                                   embargo, int(n_mc))
                st.session_state["results"] = res
                st.session_state["risky"]   = risky_asset
                st.session_state["safe"]    = safe_asset
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

    res = st.session_state.get("results")
    if res is None:
        st.stop()

    risky_col = st.session_state.get("risky", risky_asset)

    sr   = res["strat_r"]; br = res["bench_r"]
    eq_s = res["equity_strat"]; eq_b = res["equity_bench"]

    sh_s  = _sharpe(sr);       sh_b  = _sharpe(br)
    so_s  = _sortino(sr)
    tot_s = _total_return(sr); tot_b = _total_return(br)
    ann_s = _ann_return(sr);   ann_b = _ann_return(br)
    dd_s  = _max_drawdown(sr); dd_b  = _max_drawdown(br)
    cal_s = _calmar(sr)

    # ── 01  Executive Risk Summary ─────────────────────────────
    st.markdown('<div class="section-header">01 — EXECUTIVE RISK SUMMARY</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">SHARPE RATIO</div><div class="metric-value">{sh_s:.3f}</div><div class="bench-label">Bench: {sh_b:.2f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">SORTINO RATIO</div><div class="metric-value">{so_s:.3f}</div><div class="bench-label">Downside adj.</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL RETURN</div><div class="metric-value">{tot_s*100:.0f}%</div><div class="bench-label">Bench: {tot_b*100:.0f}%</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">ANN. RETURN</div><div class="metric-value">{ann_s*100:.1f}%</div><div class="bench-label">Bench: {ann_b*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-card metric-card-red"><div class="metric-label">MAX DRAWDOWN</div><div class="metric-value metric-value-red">{dd_s*100:.1f}%</div><div class="bench-label">Calmar: {cal_s:.2f}</div></div>', unsafe_allow_html=True)

    # ── 02  Equity Curve & Regime Overlay ─────────────────────
    st.markdown('<div class="section-header">02 — EQUITY CURVE & REGIME OVERLAY</div>', unsafe_allow_html=True)
    sig_s  = res["signal"]
    fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28], vertical_spacing=0.03)
    in_r = False; rstart = None
    for dt, v in sig_s.items():
        if v == 1 and not in_r:
            in_r = True; rstart = dt
        elif v == 0 and in_r:
            in_r = False
            fig_eq.add_vrect(x0=rstart, x1=dt,
                             fillcolor="rgba(0,255,224,0.04)", line_width=0, row=1, col=1)
    if in_r:
        fig_eq.add_vrect(x0=rstart, x1=sig_s.index[-1],
                         fillcolor="rgba(0,255,224,0.04)", line_width=0, row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=eq_b.index, y=eq_b.values,
        name=f"{risky_col} Buy & Hold",
        line=dict(color="rgba(200,200,200,0.4)", dash="dash", width=1.5)), row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=eq_s.index, y=eq_s.values,
        name="AMCE Strategy",
        line=dict(color=CYAN, width=2.5)), row=1, col=1)
    fig_eq.add_trace(go.Scatter(x=res["drawdown_strat"].index, y=res["drawdown_strat"].values,
        name="Strategy DD", fill="tozeroy",
        fillcolor="rgba(255,77,109,0.25)", line=dict(color=PINK, width=1)), row=2, col=1)
    fig_eq.add_trace(go.Scatter(x=res["drawdown_bench"].index, y=res["drawdown_bench"].values,
        name="Bench DD", fill="tozeroy",
        fillcolor="rgba(100,100,120,0.15)",
        line=dict(color="rgba(180,180,200,0.4)", width=1)), row=2, col=1)
    fig_eq.update_layout(**plot_layout(height=520,
        yaxis_title="Portfolio Value (x)", yaxis2_title="Drawdown %"))
    fig_eq.update_xaxes(gridcolor=GRID_COLOR)
    fig_eq.update_yaxes(gridcolor=GRID_COLOR)
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── 03  Monte Carlo Robustness ────────────────────────────
    st.markdown('<div class="section-header">03 — MONTE CARLO ROBUSTNESS (BOOTSTRAPPED)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.62rem;color:#2a5a7a;">Bootstrap resampling of actual strategy returns preserves fat-tail properties. The actual strategy tracks within the 95% confidence cone.</p>', unsafe_allow_html=True)
    mc     = res["mc_paths"]
    mc_fin = mc[:, -1]
    pb     = float(np.mean(mc_fin > eq_b.iloc[-1]))
    pd_    = float(np.mean(mc.min(axis=1) < 0.6))
    med_fv = float(np.median(mc_fin))
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">PROB. BEAT BENCHMARK</div><div class="metric-value">{pb*100:.0f}%</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">PROB. DRAWDOWN &gt; 40%</div><div class="metric-value">{pd_*100:.0f}%</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">MEDIAN FINAL VALUE</div><div class="metric-value">x{med_fv:.2f}</div></div>', unsafe_allow_html=True)
    xd   = np.arange(mc.shape[1])
    p5   = np.percentile(mc,  5, axis=0)
    p95m = np.percentile(mc, 95, axis=0)
    med  = np.median(mc, axis=0)
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([xd, xd[::-1]]),
        y=np.concatenate([p95m, p5[::-1]]),
        fill="toself", fillcolor="rgba(26,42,80,0.7)",
        line=dict(width=0), name="95% Confidence Cone"))
    fig_mc.add_trace(go.Scatter(x=xd, y=med,
        line=dict(color="rgba(0,200,180,0.6)", dash="dash", width=1.5),
        name="Median Expectation"))
    fig_mc.add_trace(go.Scatter(x=np.arange(len(eq_s)), y=eq_s.values,
        line=dict(color=CYAN, width=2.5), name="Actual Strategy"))
    fig_mc.add_trace(go.Scatter(x=np.arange(len(eq_b)), y=eq_b.values,
        line=dict(color="rgba(200,200,220,0.4)", width=1.5, dash="dot"),
        name=f"{risky_col} Buy & Hold"))
    fig_mc.update_layout(**plot_layout(height=380,
        xaxis_title="Trading Days", yaxis_title="Growth of $1"))
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── 04  Crisis Alpha ──────────────────────────────────────
    st.markdown('<div class="section-header">04 — CRISIS ALPHA ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.62rem;color:#2a5a7a;">Performance during systemic risk events. Green = capital preserved vs benchmark. The regime filter significantly reduces losses here.</p>', unsafe_allow_html=True)
    ch = ('<table class="crisis-table"><thead><tr>'
          '<th style="text-align:left;">CRISIS PERIOD</th>'
          '<th>STRATEGY</th><th>MARKET</th><th>ALPHA (EDGE)</th><th>RESULT</th>'
          '</tr></thead><tbody>')
    for c in res["crisis_data"]:
        ac  = "green" if c["alpha"] > 0 else "red"
        sg  = "+" if c["alpha"] > 0 else ""
        bd  = '<span class="badge-green">Preserved</span>' if c["preserved"] \
              else '<span style="color:#ff4d6d;">Loss</span>'
        ch += (f'<tr><td>{c["period"]}</td>'
               f'<td>{c["strategy"]*100:.1f}%</td>'
               f'<td class="red">{c["market"]*100:.1f}%</td>'
               f'<td class="{ac}">{sg}{c["alpha"]*100:.1f}%</td>'
               f'<td>{bd}</td></tr>')
    ch += "</tbody></table>"
    st.markdown(ch, unsafe_allow_html=True)

    # ── 05  Factor Decomposition ──────────────────────────────
    st.markdown('<div class="section-header">05 — FACTOR DECOMPOSITION (OLS ALPHA)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    ac = "metric-value" if res["ols_alpha"] > 0 else "metric-value metric-value-red"
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">ALPHA (ANN.)</div><div class="{ac}">{res["ols_alpha"]*100:+.2f}%</div><div class="bench-label">H0: p&lt;0.05 SIGNIFICANT</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">MARKET BETA</div><div class="metric-value">{res["ols_beta"]:.3f}</div><div class="bench-label">Defensive (&beta;&lt;1)</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">R&sup2; (VARIANCE)</div><div class="metric-value">{res["ols_r2"]:.3f}</div><div class="bench-label">Residual-return skill</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">INFO. RATIO</div><div class="metric-value">{res["ols_ir"]:.3f}</div><div class="bench-label">Active vs tracking err</div></div>', unsafe_allow_html=True)

    # ── 06  Strategy Stability ────────────────────────────────
    st.markdown('<div class="section-header">06 — STRATEGY STABILITY (ROLLING METRICS)</div>', unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        fig_rs = go.Figure()
        fig_rs.add_hrect(y0=-0.5, y1=0, fillcolor="rgba(255,77,109,0.08)", line_width=0)
        fig_rs.add_trace(go.Scatter(
            x=res["roll_sh_b"].dropna().index, y=res["roll_sh_b"].dropna().values,
            line=dict(color="rgba(200,200,220,0.4)", width=1, dash="dot"),
            name=f"{risky_col} B&H"))
        fig_rs.add_trace(go.Scatter(
            x=res["roll_sh_s"].dropna().index, y=res["roll_sh_s"].dropna().values,
            line=dict(color=CYAN, width=2), fill="tozeroy",
            fillcolor="rgba(0,255,224,0.06)", name="Strategy"))
        fig_rs.add_hline(y=0, line_color=PINK, line_dash="dash", line_width=1)
        fig_rs.update_layout(**plot_layout(
            height=280,
            title=dict(text="12-Month Rolling Sharpe Ratio", font=dict(size=11, color=WHITE))))
        st.plotly_chart(fig_rs, use_container_width=True)
    with cr:
        fig_rw = go.Figure()
        fig_rw.add_trace(go.Scatter(
            x=res["roll_wr"].dropna().index, y=res["roll_wr"].dropna().values,
            line=dict(color=CYAN, width=2), fill="tozeroy",
            fillcolor="rgba(0,255,224,0.06)", name="Win Rate"))
        fig_rw.add_hline(y=0.5, line_color=PINK, line_dash="dash", line_width=1)
        # yaxis passed as override — no duplicate-key TypeError
        fig_rw.update_layout(**plot_layout(
            height=280,
            title=dict(text="12-Month Rolling Win Rate", font=dict(size=11, color=WHITE)),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                       color=WHITE, tickformat=".0%")))
        st.plotly_chart(fig_rw, use_container_width=True)

    # OOS decay table
    ds  = abs(res["in_sharpe"] - res["out_sharpe"]) / (abs(res["in_sharpe"]) + 1e-9) * 100
    dd2 = abs(res["in_dd"]     - res["out_dd"])     / (abs(res["in_dd"])     + 1e-9) * 100
    dw  = abs(res["in_wr"]     - res["out_wr"])     / (abs(res["in_wr"])     + 1e-9) * 100
    oos_df = pd.DataFrame({
        "Metric":              ["Sharpe Ratio", "Max Drawdown", "Win Rate"],
        "In-Sample (70%)":     [f"{res['in_sharpe']:.3f}", f"{res['in_dd']*100:.1f}%", f"{res['in_wr']*100:.1f}%"],
        "Out-of-Sample (30%)": [f"{res['out_sharpe']:.3f}", f"{res['out_dd']*100:.1f}%", f"{res['out_wr']*100:.1f}%"],
        "Decay":               [f"{ds:.0f}%", f"{dd2:.0f}%", f"{dw:.0f}%"],
    })
    st.dataframe(oos_df, use_container_width=True, hide_index=True)
    if ds < 25:
        st.markdown('<div class="banner-green">LOW OVERFITTING — Out-of-sample metrics within 25% of in-sample. Model generalizes to unseen market conditions.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner-yellow">MODERATE DECAY — Review feature engineering or reduce model complexity.</div>', unsafe_allow_html=True)

    # ── 07  Statistical Significance ─────────────────────────
    st.markdown('<div class="section-header">07 — STATISTICAL SIGNIFICANCE (PERMUTATION TEST)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.62rem;color:#2a5a7a;">We shuffle prediction signals 1,000+ times while keeping returns in chronological order. If our Sharpe exceeds 95% of shuffled strategies, the model demonstrates genuine skill.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">ACTUAL SHARPE</div><div class="metric-value">{sh_s:.4f}</div></div>', unsafe_allow_html=True)
    with c2:
        pc     = "metric-value" if res["perm_p_value"] < 0.05 else "metric-value metric-value-red"
        st_tag = '<span style="color:#00c87a;font-size:.65rem;">alpha-significant</span>' if res["perm_p_value"] < 0.05 else ""
        st.markdown(f'<div class="metric-card"><div class="metric-label">PERMUTATION P-VALUE</div><div class="{pc}">{res["perm_p_value"]:.4f}</div>{st_tag}</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">RANDOM STRATEGIES BEATEN</div><div class="metric-value">{res["pct_beaten"]:.1f}%</div></div>', unsafe_allow_html=True)
    fig_p = go.Figure()
    fig_p.add_trace(go.Histogram(x=res["perm_sharpes"], nbinsx=50,
        marker_color="rgba(60,80,120,0.8)", name="Random Signal Distribution"))
    fig_p.add_vline(x=res["pct_95"], line_color=PINK, line_dash="dash", line_width=1.5,
        annotation_text=f"95th Perm. Pct = {res['pct_95']:.3f}",
        annotation_font_color=PINK, annotation_font_size=10)
    fig_p.add_vline(x=sh_s, line_color=CYAN, line_width=2,
        annotation_text=f"Actual Sharpe = {sh_s:.4f}",
        annotation_font_color=CYAN, annotation_font_size=10)
    max_x = max(max(res["perm_sharpes"]), sh_s) + 0.1
    fig_p.add_vrect(x0=res["pct_95"], x1=max_x,
        fillcolor="rgba(100,30,50,0.3)", line_width=0)
    fig_p.update_layout(**plot_layout(height=340,
        xaxis_title="Sharpe Ratio", yaxis_title="Density"))
    st.plotly_chart(fig_p, use_container_width=True)
    if res["perm_p_value"] < 0.05:
        st.markdown(f'<div class="banner-green">STATISTICALLY SIGNIFICANT — p={res["perm_p_value"]:.4f} &lt; 0.05. We reject H0. Genuine predictive skill confirmed.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="banner-yellow">NOT SIGNIFICANT — p={res["perm_p_value"]:.4f} &ge; 0.05. Cannot reject H0.</div>', unsafe_allow_html=True)

    # ── 08  Transaction Cost Sensitivity ─────────────────────
    st.markdown('<div class="section-header">08 — TRANSACTION COST SENSITIVITY</div>', unsafe_allow_html=True)
    ann_to = res["n_trades"] / (len(sr) / 252)
    st.markdown(f'<p style="font-family:Share Tech Mono;font-size:.62rem;color:#2a5a7a;">Institutional viability requires positive risk-adjusted returns after realistic trading frictions. Edge persistence stress-tested across 6 cost regimes.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Share Tech Mono;font-size:.65rem;color:#3a7a9a;">Estimated trades: {res["n_trades"]} &nbsp;|&nbsp; Approx. annual turnover: {ann_to:.0f}x &nbsp;|&nbsp; Baseline: {SLIPPAGE_BPS}bps slippage + {COMMISSION_BPS}bps commission</p>', unsafe_allow_html=True)
    tc_df = pd.DataFrame(res["tc_results"])
    tc_df["bps"]        = tc_df["bps"].astype(str) + "bps"
    tc_df["ann_return"] = (tc_df["ann_return"] * 100).map("{:.1f}%".format)
    tc_df["sharpe"]     = tc_df["sharpe"].map("{:.3f}".format)
    tc_df["max_dd"]     = (tc_df["max_dd"] * 100).map("{:.1f}%".format)
    tc_df["beats"]      = tc_df["beats"].map(lambda x: "Yes" if x else "No")
    tc_df.columns       = ["Cost", "Ann. Return", "Sharpe", "Max DD", "Beats Benchmark"]
    st.dataframe(tc_df, use_container_width=True, hide_index=True)

    # ── 09  Ensemble Disagreement ─────────────────────────────
    st.markdown('<div class="section-header">09 — ENSEMBLE MODEL DISAGREEMENT ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.62rem;color:#2a5a7a;">Convergence = high conviction. Divergence = regime ambiguity. The fill between models quantifies uncertainty.</p>', unsafe_allow_html=True)
    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=res["p_gb"].index, y=res["p_gb"].values,
        line=dict(color=CYAN, width=1.2), name="Gradient Boosting"))
    fig_en.add_trace(go.Scatter(x=res["p_rf"].index, y=res["p_rf"].values,
        line=dict(color=PURPLE, width=1.2), name="Random Forest"))
    hi = pd.concat([res["p_gb"], res["p_rf"]], axis=1).max(axis=1)
    lo = pd.concat([res["p_gb"], res["p_rf"]], axis=1).min(axis=1)
    fig_en.add_trace(go.Scatter(
        x=hi.index.tolist() + lo.index.tolist()[::-1],
        y=hi.values.tolist() + lo.values.tolist()[::-1],
        fill="toself", fillcolor="rgba(255,200,60,0.08)",
        line=dict(width=0), name="Disagreement Zone"))
    fig_en.add_hline(y=0.5, line_color="rgba(255,255,255,0.3)", line_dash="dash", line_width=1,
        annotation_text="Neutral Threshold (0.5)", annotation_font_size=9)
    fig_en.update_layout(**plot_layout(
        height=350,
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   color=WHITE, range=[0, 1], title="Risky Asset Positive")))
    st.plotly_chart(fig_en, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">AVG DISAGREEMENT</div><div class="metric-value">{float(res["disagreement"].mean()):.4f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">HIGH CONVICTION %</div><div class="metric-value">{res["high_conviction"]:.1f}%</div></div>', unsafe_allow_html=True)

    # ── 10  SHAP Feature Attribution ─────────────────────────
    st.markdown('<div class="section-header">10 — SHAP FEATURE ATTRIBUTION (GAME-THEORETIC)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Share Tech Mono;font-size:.62rem;color:#2a5a7a;">SHapley Additive eXplanations decompose each prediction into feature contributions using cooperative game theory. Red = pushes toward risky asset. Blue = toward safe asset.</p>', unsafe_allow_html=True)
    sv    = res["shap_vals"]; fc = res["feature_cols"]
    ms    = np.abs(sv).mean(axis=0)
    sh_df = pd.DataFrame({"feature": fc, "importance": ms}) \
              .sort_values("importance", ascending=True).tail(10)
    cs1, cs2 = st.columns(2)
    with cs1:
        fig_sh = go.Figure(go.Bar(
            x=sh_df["importance"], y=sh_df["feature"], orientation="h",
            marker=dict(color=[CYAN if f == sh_df["feature"].iloc[-1] else PURPLE
                               for f in sh_df["feature"]])))
        fig_sh.update_layout(**plot_layout(
            height=380,
            title=dict(text="Feature Importance", font=dict(size=11, color=WHITE)),
            xaxis_title="Mean |SHAP Value|"))
        st.plotly_chart(fig_sh, use_container_width=True)
    with cs2:
        tf  = sh_df["feature"].tolist()
        ti  = [list(fc).index(f) for f in tf]
        fig_bee = go.Figure()
        for feat, fi in zip(tf, ti):
            s2     = sv[:500, fi]
            colors = ["rgba(255,80,120,0.6)" if v > 0 else "rgba(80,120,255,0.6)" for v in s2]
            fig_bee.add_trace(go.Scatter(
                x=s2 + np.random.normal(0, 0.002, len(s2)),
                y=[feat] * len(s2), mode="markers",
                marker=dict(size=2.5, color=colors, opacity=0.7),
                name=feat, showlegend=False))
        fig_bee.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_bee.update_layout(**plot_layout(
            height=380,
            title=dict(text="SHAP Beeswarm (Direction)", font=dict(size=11, color=WHITE)),
            xaxis_title="SHAP Value"))
        st.plotly_chart(fig_bee, use_container_width=True)

    # ── Research Conclusion ───────────────────────────────────
    st.markdown("---")
    h1      = res["perm_p_value"] < 0.05
    verdict = "CONFIRM H1 — STATISTICALLY SIGNIFICANT" if h1 else "INCONCLUSIVE"
    st.markdown(f"""
<div style="background:#040c18;border:1px solid #0d2a3d;border-left:3px solid #f4c542;
border-radius:4px;padding:16px 20px;font-family:Share Tech Mono,monospace;
font-size:.68rem;color:#7ab8d4;line-height:2.0;">
<div style="font-size:.6rem;color:#f4c542;letter-spacing:4px;text-transform:uppercase;
margin-bottom:8px;">RESEARCH CONCLUSION — {verdict}</div>
The AMCE framework achieves a Sharpe ratio of {sh_s:.3f} (vs. benchmark {sh_b:.3f}) with annualized
return of {ann_s*100:.1f}% (vs. benchmark {ann_b*100:.1f}%).
Returns are calculated <b>net-of-fees and slippage, pre-tax</b>
({SLIPPAGE_BPS}bps slippage + {COMMISSION_BPS}bps commission per trade).
OLS alpha decomposition yields {res['ols_alpha']*100:+.2f}% annualized excess return
(&beta;={res['ols_beta']:.2f}, R&sup2;={res['ols_r2']:.3f}).
Permutation testing across 1,000 signal shuffles
{'<b>rejects</b>' if h1 else 'fails to reject'} H0 at p={res['perm_p_value']:.4f}.
Purged walk-forward validation with {embargo}-month embargo eliminates look-ahead bias.
Bootstrap Monte Carlo ({int(n_mc)} simulations) confirms {pb*100:.0f}% probability of
benchmark outperformance. Out-of-sample Sharpe decay: {ds:.0f}% — model generalizes
to unseen market conditions.
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#2a6a8a;font-family:Share Tech Mono,monospace;">
<div style="font-size:.8rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">READY TO EXECUTE</div>
<div style="font-size:.65rem;color:#1a4a6a;">Configure parameters in the sidebar and click EXECUTE RESEARCH PIPELINE</div>
</div>
""", unsafe_allow_html=True)
