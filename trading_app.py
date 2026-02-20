import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from core_engine import InstitutionalAMCE

st.set_page_config(page_title="AMCE Master Terminal", layout="wide", initial_sidebar_state="expanded")

# --- INSTITUTIONAL DARK CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    .metric-box { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; border-top: 3px solid #58a6ff; margin-bottom: 15px;}
    .metric-title { font-size: 0.75rem; color: #8b949e; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;}
    .metric-value { font-size: 1.6rem; color: #f0f6fc; font-family: 'Space Mono', monospace; font-weight: 700;}
    .metric-sub { font-size: 0.8rem; color: #3fb950; margin-top: 5px;}
    .metric-sub-negative { font-size: 0.8rem; color: #f85149; margin-top: 5px;}
    .section-header { font-family: 'Space Mono', monospace; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-top: 40px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h3 style='color: #58a6ff; font-family: monospace;'>⚙️ MASTER ENGINE</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

risk_asset = st.sidebar.text_input("High-Beta Asset", "QQQ")
safe_asset = st.sidebar.text_input("Risk-Free Asset", "SHY")

st.sidebar.markdown("### 🎛️ ML PARAMS")
train_window = st.sidebar.slider("Train Window (Days)", 252, 1260, 756, step=252)
step_size = st.sidebar.slider("Step Size (Days)", 21, 252, 63, step=21)

st.sidebar.markdown("### 🎲 MONTE CARLO")
mc_sims = st.sidebar.slider("Bootstrap Paths", 100, 1000, 500, step=100)

st.sidebar.markdown("### 📉 REAL WORLD FRICTIONS")
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
slip_bps = st.sidebar.slider("Market Slippage (bps)", 0, 20, 2)
tax_rate = st.sidebar.slider("Short-Term Tax Rate (%)", 0.0, 50.0, 0.0, step=5.0) / 100.0

execute = st.sidebar.button("COMPILE MASTER TERMINAL", use_container_width=True)

# --- STATS HELPER ---
def calc_stats(returns):
    cum_ret = (1 + returns).cumprod()
    cagr = (cum_ret.iloc[-1]) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0
    peak = cum_ret.cummax()
    dd = (cum_ret / peak) - 1
    return cagr, vol, sharpe, dd.min(), cum_ret, dd

# --- MAIN DASHBOARD ---
st.markdown("<h1 style='font-family: monospace; color: #58a6ff; margin-bottom: 0px;'>INSTITUTIONAL MACRO ENSEMBLE</h1>", unsafe_allow_html=True)
st.markdown("<p style='letter-spacing: 1px; color: #8b949e; font-size: 0.85rem;'>10-CHART VISUALIZATION SUITE • MONTE CARLO STRESS TESTING • 3D REGIME MAPPING</p>", unsafe_allow_html=True)

if execute:
    with st.spinner("Compiling ML Models, running 500-path Monte Carlo, and rendering 3D WebGL..."):
        # 1. RUN CORE ENGINE
        engine = InstitutionalAMCE(risk_asset, safe_asset)
        engine.fetch_data()
        engine.engineer_features()
        engine.purged_walk_forward_backtest(train_window=train_window, step_size=step_size)
        
        # --- NEW REPLACEMENT LINE ---
        bt = engine.simulate_portfolio(tc_bps=tc_bps, slip_bps=slip_bps, tax_rate=tax_rate)
        
        strat_cagr, strat_vol, strat_sharpe, strat_mdd, strat_cum, strat_dd = calc_stats(bt['Net_Ret'])
        
        # 2. EXECUTIVE METRICS
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.markdown(f"<div class='metric-box'><div class='metric-title'>OOS SHARPE</div><div class='metric-value'>{strat_sharpe:.2f}</div><div class='metric-sub'>Bench: {bench_sharpe:.2f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><div class='metric-title'>NET CAGR</div><div class='metric-value'>{strat_cagr*100:.2f}%</div><div class='{'metric-sub' if strat_cagr>bench_cagr else 'metric-sub-negative'}'>{strat_cagr*100 - bench_cagr*100:+.2f}% vs Bench</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><div class='metric-title'>MAX DRAWDOWN</div><div class='metric-value'>{strat_mdd*100:.2f}%</div><div class='metric-sub'>Bench: {bench_mdd*100:.2f}%</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-box'><div class='metric-title'>VOLATILITY</div><div class='metric-value'>{strat_vol*100:.2f}%</div><div class='metric-sub'>Bench: {bench_vol*100:.2f}%</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-box'><div class='metric-title'>CALMAR RATIO</div><div class='metric-value'>{strat_cagr/abs(strat_mdd):.2f}</div><div class='metric-sub'>Return / Risk</div></div>", unsafe_allow_html=True)
        c6.markdown(f"<div class='metric-box'><div class='metric-title'>AVG EXPOSURE</div><div class='metric-value'>{bt['Actual_Weight_Risk'].mean()*100:.0f}%</div><div class='metric-sub'>Capital Deployed</div></div>", unsafe_allow_html=True)

        # ==========================================
        # SECTION 1: PERFORMANCE & EXPOSURE
        # ==========================================
        st.markdown("<div class='section-header'>01 — Portfolio Trajectory & Dynamic Allocation</div>", unsafe_allow_html=True)
        
        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04)
        # Chart 1: Equity Curve
        fig1.add_trace(go.Scatter(x=bt.index, y=strat_cum, name="Strategy (Net)", line=dict(color="#58a6ff", width=2)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=bt.index, y=bench_cum, name="Benchmark", line=dict(color="#8b949e", width=1, dash="dot")), row=1, col=1)
        # Chart 2: Underwater Profile
        fig1.add_trace(go.Scatter(x=bt.index, y=strat_dd, name="Strategy DD", fill='tozeroy', line=dict(color="#f85149", width=1), fillcolor="rgba(248, 81, 73, 0.2)"), row=2, col=1)
        # Chart 3: Position Sizing
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['Actual_Weight_Risk'], name="Risk Exposure", fill='tozeroy', line=dict(color="#3fb950", width=1), fillcolor="rgba(63, 185, 80, 0.2)"), row=3, col=1)
        
        fig1.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=700, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
        fig1.update_yaxes(title_text="Cumulative Return", row=1, col=1, gridcolor="#30363d")
        fig1.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1, gridcolor="#30363d")
        fig1.update_yaxes(title_text="Equity Weight", tickformat=".0%", row=3, col=1, gridcolor="#30363d", range=[0, 1.1])
        st.plotly_chart(fig1, use_container_width=True)

        # ==========================================
        # SECTION 2: MONTE CARLO & STATISTICAL RIGOR
        # ==========================================
        st.markdown("<div class='section-header'>02 — Monte Carlo Simulation & p-Value (Statistical Edge)</div>", unsafe_allow_html=True)
        
        # Run MC Simulation
        daily_rets = bt['Net_Ret'].values
        mc_paths = []
        for _ in range(mc_sims):
            # Bootstrap daily returns with replacement
            sim_rets = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
            mc_paths.append((1 + sim_rets).cumprod())
            
        mc_paths = np.array(mc_paths)
        final_wealths = mc_paths[:, -1]
        bench_final = bench_cum.iloc[-1]
        
        # Calculate p-value: Probability strategy underperforms the benchmark final wealth
        p_value = np.sum(final_wealths < bench_final) / mc_sims

        col_mc1, col_mc2 = st.columns([1.5, 1])
        with col_mc1:
            st.markdown(f"<span style='font-size: 0.85rem; color: #8b949e;'>CHART 4: {mc_sims}-PATH BOOTSTRAPPED RETURN TRAJECTORIES</span>", unsafe_allow_html=True)
            fig_mc = go.Figure()
            # Plot only first 50 paths to save browser memory
            for i in range(min(50, mc_sims)):
                fig_mc.add_trace(go.Scatter(x=bt.index, y=mc_paths[i], mode='lines', line=dict(color='rgba(88, 166, 255, 0.05)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=bt.index, y=strat_cum, name="Actual Strategy", line=dict(color="#3fb950", width=3)))
            fig_mc.add_trace(go.Scatter(x=bt.index, y=bench_cum, name="Actual Benchmark", line=dict(color="#f85149", width=2, dash="dot")))
            fig_mc.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_mc, use_container_width=True)
            
        with col_mc2:
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>CHART 5: FINAL WEALTH DISTRIBUTION</span>", unsafe_allow_html=True)
            fig_hist = px.histogram(x=final_wealths, nbins=50, color_discrete_sequence=['#58a6ff'], opacity=0.7)
            fig_hist.add_vline(x=bench_final, line_width=3, line_dash="dash", line_color="#f85149", annotation_text="Benchmark Final", annotation_position="top right")
            fig_hist.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Final Cumulative Return (x)", yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Print p-value mathematically
            color = "#3fb950" if p_value < 0.05 else "#f85149"
            st.markdown(f"<div style='text-align: center; padding: 10px; border: 1px solid #30363d; border-radius: 5px;'><h4 style='margin:0; color:{color};'>p-value = {p_value:.4f}</h4><p style='margin:0; font-size: 0.8rem; color: #8b949e;'>Probability of underperforming benchmark by random chance.</p></div>", unsafe_allow_html=True)

        # ==========================================
        # SECTION 3: ML DIAGNOSTICS & 3D MAPPING
        # ==========================================
        st.markdown("<div class='section-header'>03 — 3D Regime Map & Model Disagreement</div>", unsafe_allow_html=True)
        col_ml1, col_ml2 = st.columns([1, 1])

        with col_ml1:
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>CHART 6: 3D FEATURE SPACE (VOL vs MOMENTUM vs AI CONFIDENCE)</span>", unsafe_allow_html=True)
            df_3d = engine.full_data.loc[bt.index].copy()
            df_3d['AI_Probability'] = bt['Prob']
            df_3d['Stance'] = np.where(df_3d['AI_Probability'] > 0.5, 'Risk-On', 'Risk-Off')
            df_3d = df_3d.iloc[::3] # Downsample for 3D performance
            
            fig_3d = px.scatter_3d(
                df_3d, x='Cross_Asset_Strength', y='Vol_1M', z='AI_Probability', color='Stance',
                color_discrete_map={'Risk-On': '#3fb950', 'Risk-Off': '#f85149'},
                opacity=0.8, size_max=4
            )
            fig_3d.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", margin=dict(l=0, r=0, b=0, t=0), height=450,
                scene=dict(xaxis_title='Cross-Asset Momentum', yaxis_title='1M Volatility', zaxis_title='Risk-On Prob'))
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_ml2:
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>CHART 7: ENSEMBLE DISAGREEMENT (LR vs RF vs GB VARIANCE)</span>", unsafe_allow_html=True)
            fig_dis = go.Figure()
            fig_dis.add_trace(go.Scatter(x=bt.index, y=engine.results['Model_Disagreement'].rolling(21).mean(), name="21D Avg Variance", fill='tozeroy', line=dict(color="#d2a8ff", width=2), fillcolor="rgba(210, 168, 255, 0.2)"))
            fig_dis.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=450, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_dis, use_container_width=True)

        # ==========================================
        # SECTION 4: TEMPORAL STABILITY & HEATMAPS
        # ==========================================
        st.markdown("<div class='section-header'>04 — Temporal Stability & Crisis Alpha Capture</div>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns([1, 1])
        
        with col_t1:
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>CHART 8: ROLLING 12-MONTH SHARPE RATIO</span>", unsafe_allow_html=True)
            roll_ret = bt['Net_Ret'].rolling(252).mean() * 252
            roll_vol = bt['Net_Ret'].rolling(252).std() * np.sqrt(252)
            roll_sharpe = roll_ret / roll_vol
            
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name="12M Sharpe", line=dict(color="#58a6ff")))
            fig_roll.add_hline(y=0, line_dash="dash", line_color="#8b949e")
            fig_roll.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_roll, use_container_width=True)

        with col_t2:
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>CHART 9: ANNUAL RETURN COMPARISON</span>", unsafe_allow_html=True)
            # Resample to annual
            strat_ann = bt['Net_Ret'].resample('Y').apply(lambda x: (1+x).prod() - 1)
            bench_ann = bt['Bench_Ret'].resample('Y').apply(lambda x: (1+x).prod() - 1)
            years = strat_ann.index.year
            
            fig_ann = go.Figure(data=[
                go.Bar(name='Strategy', x=years, y=strat_ann.values, marker_color="#58a6ff"),
                go.Bar(name='Benchmark', x=years, y=bench_ann.values, marker_color="#30363d")
            ])
            fig_ann.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=300, margin=dict(l=0, r=0, t=10, b=0), barmode='group')
            fig_ann.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_ann, use_container_width=True)

        # Chart 10: Monthly Heatmap
        st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>CHART 10: MONTHLY NET RETURN DISTRIBUTION</span>", unsafe_allow_html=True)
        monthly_returns = bt['Net_Ret'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        df_hm = pd.DataFrame({'Year': monthly_returns.index.year, 'Month': monthly_returns.index.month, 'Return': monthly_returns.values})
        hm_pivot = df_hm.pivot(index='Year', columns='Month', values='Return')
        hm_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_pivot.values, x=hm_pivot.columns, y=hm_pivot.index,
            colorscale=[[0.0, '#f85149'], [0.5, '#0d1117'], [1.0, '#3fb950']],
            zmid=0, text=np.round(hm_pivot.values*100, 1), texttemplate="%{text}%", showscale=False
        ))
        fig_hm.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=400, margin=dict(l=0, r=0, t=10, b=0))
        fig_hm.update_yaxes(autorange="reversed", type='category')
        st.plotly_chart(fig_hm, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px; border: 1px dashed #30363d; margin-top: 50px;'>
        <h2 style='color: #58a6ff; font-family: monospace;'>[ TERMINAL STANDBY ]</h2>
        <p style='color: #8b949e;'>Awaiting execution. Click 'Compile Master Terminal' to run the ensemble and render the 10-chart layout.</p>
    </div>
    """, unsafe_allow_html=True)
