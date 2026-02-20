import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core_engine import InstitutionalQuantEngine

st.set_page_config(page_title="Systematic Macro Terminal", layout="wide", initial_sidebar_state="expanded")

# Institutional Dark CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #8b949e; font-family: 'Inter', sans-serif; }
    .metric-box { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 5px; border-top: 3px solid #00f2fe; }
    .metric-title { font-size: 0.75rem; color: #8b949e; font-weight: bold; letter-spacing: 1px; }
    .metric-value { font-size: 1.6rem; color: #c9d1d9; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ EXECUTION DESK")
risk_asset = st.sidebar.text_input("Risk Asset", "SPY")
safe_asset = st.sidebar.text_input("Safe Asset", "IEF")

st.sidebar.markdown("### 📉 FRICTIONS")
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
slip_bps = st.sidebar.slider("Slippage (bps)", 0, 20, 5)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.0, 40.0, 15.0) / 100.0

execute = st.sidebar.button("RUN OUT-OF-SAMPLE BACKTEST")

st.markdown("## 🌐 SYSTEMATIC MACRO RESEARCH TERMINAL")
st.markdown("Strict walk-forward out-of-sample validation. Friction and tax adjusted.")

if execute:
    with st.spinner("Executing Quant Pipeline..."):
        engine = InstitutionalQuantEngine(risk_asset, safe_asset)
        engine.fetch_data()
        engine.engineer_features()
        engine.walk_forward_validation()
        bt = engine.backtest_with_frictions(tc_bps, slip_bps, tax_rate)
        stats = engine.calculate_statistics()
        
        # 1. Executive Summary
        st.markdown("#### I. RISK & RETURN SUMMARY")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f"<div class='metric-box'><div class='metric-title'>OOS SHARPE</div><div class='metric-value'>{stats['Sharpe']:.2f}</div><div style='font-size: 0.7rem; color: #8b949e;'>95% CI: [{stats['Sharpe_95CI'][0]:.2f}, {stats['Sharpe_95CI'][1]:.2f}]</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><div class='metric-title'>CAGR (NET)</div><div class='metric-value'>{stats['CAGR']*100:.2f}%</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><div class='metric-title'>MAX DRAWDOWN</div><div class='metric-value'>{stats['Max_DD']*100:.2f}%</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-box'><div class='metric-title'>SKEWNESS</div><div class='metric-value'>{stats['Skew']:.2f}</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-box'><div class='metric-title'>TOTAL RETURN</div><div class='metric-value'>{stats['Total_Ret']*100:.0f}%</div></div>", unsafe_allow_html=True)
        
        st.write("")
        
        # 2. Equity Curve & Drawdown Profile
        st.markdown("#### II. PORTFOLIO TRAJECTORY")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        fig.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Strat'], name="Strategy (Net)", line=dict(color="#00f2fe", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Bench'], name="Benchmark", line=dict(color="#8b949e", dash="dot")), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=bt.index, y=bt['Strat_DD'], name="Drawdown", fill='tozeroy', line=dict(color="#ff3366")), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14", height=600, hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)
