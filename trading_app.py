import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from core_engine import InstitutionalAMCE

st.set_page_config(page_title="AMCE Research Terminal", layout="wide", initial_sidebar_state="expanded")

# --- INSTITUTIONAL QUANT CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&display=swap');

    .stApp { 
        background-color: #0B0F14; 
        color: #D1D5DB; 
        font-family: 'Source Sans 3', sans-serif; 
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* Metric Boxes */
    .metric-box { 
        background-color: #111827; 
        border: 1px solid #1F2937; 
        padding: 16px; 
        border-radius: 4px; 
        box-shadow: inset 0px 0px 10px rgba(0,0,0,0.2);
        margin-bottom: 16px;
    }
    .metric-title { 
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem; 
        color: #9CA3AF; 
        font-weight: 500; 
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value { 
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem; 
        color: #F3F4F6; 
        font-weight: 700;
        text-shadow: 0px 0px 8px rgba(255,255,255,0.1);
    }
    .metric-sub { 
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem; 
        color: #6B7280; 
        margin-top: 4px;
    }
    .metric-footnote {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.7rem;
        font-style: italic;
        color: #6B7280;
        margin-top: 6px;
        border-top: 1px solid #1F2937;
        padding-top: 4px;
    }
    .positive-val { color: #0FA47A; }
    .negative-val { color: #8B1E2D; }

    /* Section Headers */
    .section-header { 
        font-family: 'IBM Plex Sans', sans-serif; 
        color: #D1D5DB; 
        font-size: 1.1rem;
        font-weight: 500;
        border-bottom: 1px solid #1F2937; 
        padding-bottom: 6px; 
        margin-top: 48px; 
        margin-bottom: 16px; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
    }
    
    /* Small Sidebar Text */
    .sidebar-caption {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: -10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h3 style='color: #D1D5DB; font-size: 1.2rem; font-weight: 600; letter-spacing: 1px;'>RESEARCH CONFIGURATION</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color: #1F2937; margin-top: 0px;'>", unsafe_allow_html=True)

risk_asset = st.sidebar.text_input("High-Beta Asset", "QQQ")
safe_asset = st.sidebar.text_input("Risk-Free Asset", "SHY")

st.sidebar.markdown("<h4 style='color: #9CA3AF; font-size: 0.9rem; margin-top: 20px;'>MODEL SPECIFICATION</h4>", unsafe_allow_html=True)
train_window = st.sidebar.slider("Train Window (Days)", 252, 1260, 756, step=252)
st.sidebar.markdown("<div class='sidebar-caption'>Rolling walk-forward formation period.</div>", unsafe_allow_html=True)

step_size = st.sidebar.slider("Step Size (Days)", 21, 252, 63, step=21)
st.sidebar.markdown("<div class='sidebar-caption'>Out-of-sample holding period before retrain.</div>", unsafe_allow_html=True)

st.sidebar.markdown("<h4 style='color: #9CA3AF; font-size: 0.9rem; margin-top: 20px;'>STATISTICAL ENGINE</h4>", unsafe_allow_html=True)
mc_sims = st.sidebar.slider("Bootstrap Paths", 100, 1000, 500, step=100)
st.sidebar.markdown("<div class='sidebar-caption'>Iterations for p-value and CI generation.</div>", unsafe_allow_html=True)

st.sidebar.markdown("<h4 style='color: #9CA3AF; font-size: 0.9rem; margin-top: 20px;'>EXECUTION FRICTIONS</h4>", unsafe_allow_html=True)
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
st.sidebar.markdown("<div class='sidebar-caption'>Applied per turnover event.</div>", unsafe_allow_html=True)

slip_bps = st.sidebar.slider("Market Slippage (bps)", 0, 20, 2)
st.sidebar.markdown("<div class='sidebar-caption'>Modeled impact to bid/ask spread crossing.</div>", unsafe_allow_html=True)

tax_rate = st.sidebar.slider("Short-Term Tax Rate (%)", 0.0, 50.0, 0.0, step=5.0) / 100.0
st.sidebar.markdown("<div class='sidebar-caption'>Applied daily to positive pre-tax active returns.</div>", unsafe_allow_html=True)

execute = st.sidebar.button("INITIALIZE BACKTEST", use_container_width=True)

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
st.markdown("<h1 style='color: #E5E7EB; letter-spacing: 1.5px; margin-bottom: 0px;'>INSTITUTIONAL MACRO ENSEMBLE</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6B7280; font-size: 0.9rem;'>OOS WALK-FORWARD BACKTEST • MONTE CARLO INFERENCE • REGIME MAPPING</p>", unsafe_allow_html=True)

if execute:
    with st.spinner("Compiling ML Models and Bootstrap Distributions..."):
        # 1. RUN CORE ENGINE
        engine = InstitutionalAMCE(risk_asset, safe_asset)
        engine.fetch_data()
        engine.engineer_features()
        engine.purged_walk_forward_backtest(train_window=train_window, step_size=step_size)
        
        # THE FIX: Frictions are properly passed in
        bt = engine.simulate_portfolio(tc_bps=tc_bps, slip_bps=slip_bps, tax_rate=tax_rate)
        
        # THE FIX: Both strat and bench stats are calculated
        strat_cagr, strat_vol, strat_sharpe, strat_mdd, strat_cum, strat_dd = calc_stats(bt['Net_Ret'])
        bench_cagr, bench_vol, bench_sharpe, bench_mdd, bench_cum, bench_dd = calc_stats(bt['Bench_Ret'])
        
        # Run MC Simulation for CIs and p-values
        daily_rets = bt['Net_Ret'].values
        mc_sharpes = []
        mc_final = []
        for _ in range(mc_sims):
            sim_rets = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
            sim_cagr = ((1 + sim_rets).prod()) ** (252 / len(sim_rets)) - 1
            sim_vol = sim_rets.std() * np.sqrt(252)
            mc_sharpes.append(sim_cagr / sim_vol if sim_vol != 0 else 0)
            mc_final.append((1 + sim_rets).prod())
            
        ci_lower, ci_upper = np.percentile(mc_sharpes, [2.5, 97.5])
        bench_final = bench_cum.iloc[-1]
        p_val_wealth = np.sum(np.array(mc_final) < bench_final) / mc_sims
        if p_val_wealth == 0: p_val_wealth = 0.0001 # lower bound display
        
        # 2. EXECUTIVE METRICS
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        c1.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>SHARPE RATIO</div>
            <div class='metric-value'>{strat_sharpe:.2f}</div>
            <div class='metric-sub'>95% CI: [{ci_lower:.2f} – {ci_upper:.2f}]</div>
            <div class='metric-footnote'>Bench: {bench_sharpe:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        cagr_diff = strat_cagr - bench_cagr
        cagr_color = "positive-val" if cagr_diff > 0 else "negative-val"
        c2.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>NET CAGR</div>
            <div class='metric-value'>{strat_cagr*100:.1f}%</div>
            <div class='metric-sub {cagr_color}'>{cagr_diff*100:+.1f}% vs Bench</div>
            <div class='metric-footnote'>p-val: {p_val_wealth:.2e}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>MAX DRAWDOWN</div>
            <div class='metric-value'>{-strat_mdd*100:.1f}%</div>
            <div class='metric-sub'>Bench: {-bench_mdd*100:.1f}%</div>
            <div class='metric-footnote'>Worst-case trajectory</div>
        </div>
        """, unsafe_allow_html=True)
        
        c4.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>ANN. VOLATILITY</div>
            <div class='metric-value'>{strat_vol*100:.1f}%</div>
            <div class='metric-sub'>Bench: {bench_vol*100:.1f}%</div>
            <div class='metric-footnote'>Daily σ × √252</div>
        </div>
        """, unsafe_allow_html=True)
        
        calmar = strat_cagr/abs(strat_mdd) if strat_mdd != 0 else 0
        c5.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>CALMAR RATIO</div>
            <div class='metric-value'>{calmar:.2f}</div>
            <div class='metric-sub'>CAGR / Max DD</div>
            <div class='metric-footnote'>Tail-risk adjusted</div>
        </div>
        """, unsafe_allow_html=True)
        
        c6.markdown(f"""
        <div class='metric-box'>
            <div class='metric-title'>MEAN EXPOSURE</div>
            <div class='metric-value'>{bt['Actual_Weight_Risk'].mean()*100:.0f}%</div>
            <div class='metric-sub'>Capital in Risk Asset</div>
            <div class='metric-footnote'>Continuous scaling</div>
        </div>
        """, unsafe_allow_html=True)

        # ==========================================
        # SECTION 1: PERFORMANCE & EXPOSURE
        # ==========================================
        st.markdown("<div class='section-header'>01 — Portfolio Trajectory & Dynamic Allocation</div>", unsafe_allow_html=True)
        
        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04)
        
        # Deep teal for strategy, slate for bench
        fig1.add_trace(go.Scatter(x=bt.index, y=strat_cum, name="Net Strategy", line=dict(color="#1E6F7A", width=1.5)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=bt.index, y=bench_cum, name="Benchmark", line=dict(color="#4B5563", width=1, dash="dot")), row=1, col=1)
        
        # Deep red for DD
        fig1.add_trace(go.Scatter(x=bt.index, y=strat_dd, name="Drawdown", fill='tozeroy', line=dict(color="#8B1E2D", width=0.5), fillcolor="rgba(139, 30, 45, 0.2)"), row=2, col=1)
        
        # Emerald for exposure
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['Actual_Weight_Risk'], name="Risk Exp.", fill='tozeroy', line=dict(color="#0FA47A", width=0.5), fillcolor="rgba(15, 164, 122, 0.1)"), row=3, col=1)
        
        fig1.update_layout(
            template="plotly_dark", plot_bgcolor="#0B0F14", paper_bgcolor="#0B0F14", 
            height=650, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
            font=dict(family="JetBrains Mono", size=10, color="#9CA3AF")
        )
        fig1.update_xaxes(showgrid=True, gridcolor="#1F2937", gridwidth=0.5)
        fig1.update_yaxes(showgrid=True, gridcolor="#1F2937", gridwidth=0.5, row=1, col=1)
        fig1.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1, gridcolor="#1F2937")
        fig1.update_yaxes(title_text="Weight", tickformat=".0%", row=3, col=1, gridcolor="#1F2937", range=[0, 1.1])
        st.plotly_chart(fig1, use_container_width=True)

        # ==========================================
        # SECTION 2: RESEARCH EXPANDERS & MC
        # ==========================================
        st.markdown("<div class='section-header'>02 — Statistical Rigor & Methodology</div>", unsafe_allow_html=True)
        
        with st.expander("VALIDATION FRAMEWORK & LEAKAGE PREVENTION"):
            st.markdown("""
            **Purged Walk-Forward Backtest Methodology:**
            * The model utilizes a rolling training window to adapt to shifting market regimes without introducing lookahead bias.
            * Combinatorial Purged Cross-Validation (CPCV) principles are applied implicitly through strict temporal separation of train/test splits.
            * Feature construction relies purely on lagged observations $t-1$ to predict $t+1$.
            * **Frictions Applied:** Turnover is explicitly calculated. Executions cross the simulated spread (slippage), and active capital gains are taxed per the specified configuration before net alpha is recorded.
            """)
            
        with st.expander("MODEL SPECIFICATION & ENSEMBLE WEIGHTING"):
            st.markdown("""
            **Heterogeneous Meta-Estimator:**
            * The engine combines a Linear Classifier (Logistic Regression) with Non-Linear Tree-based methods (Random Forest, Gradient Boosting).
            * This prevents overfitting to single-regime dynamics (e.g., linear momentum vs. non-linear volatility clustering).
            * Final signal probability is the unweighted mean of the ensemble outputs: $P_{ensemble} = \frac{1}{n}\sum P_i$
            * Position sizing maps the probability output to a continuous exposure scale [0.0, 1.0], acting as a natural volatility targeter.
            """)

        col_mc1, col_mc2 = st.columns([1.5, 1])
        with col_mc1:
            st.markdown("<span style='font-family: IBM Plex Sans; font-size: 0.8rem; color: #9CA3AF;'>BOOTSTRAPPED RETURN TRAJECTORIES</span>", unsafe_allow_html=True)
            fig_mc = go.Figure()
            # Plot subset of paths
            mc_paths = []
            for _ in range(50):
                sim_rets = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
                mc_paths.append((1 + sim_rets).cumprod())
            
            mc_paths = np.array(mc_paths)
            for i in range(50):
                fig_mc.add_trace(go.Scatter(x=bt.index, y=mc_paths[i], mode='lines', line=dict(color='rgba(30, 111, 122, 0.05)'), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=bt.index, y=strat_cum, name="Actual Strategy", line=dict(color="#0FA47A", width=2)))
            fig_mc.add_trace(go.Scatter(x=bt.index, y=bench_cum, name="Benchmark", line=dict(color="#8B1E2D", width=1.5, dash="dot")))
            
            fig_mc.update_layout(template="plotly_dark", plot_bgcolor="#0B0F14", paper_bgcolor="#0B0F14", height=350, margin=dict(l=0, r=0, t=10, b=0), font=dict(family="JetBrains Mono", size=10))
            fig_mc.update_xaxes(showgrid=True, gridcolor="#1F2937")
            fig_mc.update_yaxes(showgrid=True, gridcolor="#1F2937")
            st.plotly_chart(fig_mc, use_container_width=True)
            
        with col_mc2:
            st.markdown("<span style='font-family: IBM Plex Sans; font-size: 0.8rem; color: #9CA3AF;'>FINAL WEALTH DISTRIBUTION</span>", unsafe_allow_html=True)
            fig_hist = px.histogram(x=mc_final, nbins=50, color_discrete_sequence=['#1E6F7A'], opacity=0.8)
            fig_hist.add_vline(x=bench_final, line_width=2, line_dash="dash", line_color="#8B1E2D", annotation_text="Bench Final", annotation_position="top right")
            fig_hist.update_layout(template="plotly_dark", plot_bgcolor="#0B0F14", paper_bgcolor="#0B0F14", height=350, margin=dict(l=0, r=0, t=10, b=0), font=dict(family="JetBrains Mono", size=10), xaxis_title="Wealth Multiplier", yaxis_title="Freq")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ==========================================
        # SECTION 3: 3D REGIME & DISAGREEMENT
        # ==========================================
        st.markdown("<div class='section-header'>03 — Feature Space & Model Variance</div>", unsafe_allow_html=True)
        col_ml1, col_ml2 = st.columns([1, 1])

        with col_ml1:
            st.markdown("<span style='font-family: IBM Plex Sans; font-size: 0.8rem; color: #9CA3AF;'>3D REGIME MAPPING (VOL vs MOMENTUM vs AI PROB)</span>", unsafe_allow_html=True)
            df_3d = engine.full_data.loc[bt.index].copy()
            df_3d['AI_Probability'] = bt['Prob']
            df_3d['Stance'] = np.where(df_3d['AI_Probability'] > 0.5, 'Risk-On', 'Risk-Off')
            df_3d = df_3d.iloc[::3] 
            
            fig_3d = px.scatter_3d(
                df_3d, x='Cross_Asset_Strength', y='Vol_1M', z='AI_Probability', color='Stance',
                color_discrete_map={'Risk-On': '#0FA47A', 'Risk-Off': '#8B1E2D'},
                opacity=0.7, size_max=3
            )
            fig_3d.update_layout(template="plotly_dark", paper_bgcolor="#0B0F14", margin=dict(l=0, r=0, b=0, t=0), height=400, font=dict(family="JetBrains Mono", size=10))
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_ml2:
            st.markdown("<span style='font-family: IBM Plex Sans; font-size: 0.8rem; color: #9CA3AF;'>ENSEMBLE PREDICTION VARIANCE</span>", unsafe_allow_html=True)
            fig_dis = go.Figure()
            fig_dis.add_trace(go.Scatter(x=bt.index, y=engine.results['Model_Disagreement'].rolling(21).mean(), fill='tozeroy', line=dict(color="#6B7280", width=1), fillcolor="rgba(107, 114, 128, 0.2)"))
            fig_dis.update_layout(template="plotly_dark", plot_bgcolor="#0B0F14", paper_bgcolor="#0B0F14", height=400, margin=dict(l=0, r=0, t=10, b=0), font=dict(family="JetBrains Mono", size=10))
            fig_dis.update_xaxes(showgrid=True, gridcolor="#1F2937")
            fig_dis.update_yaxes(showgrid=True, gridcolor="#1F2937")
            st.plotly_chart(fig_dis, use_container_width=True)

        # ==========================================
        # FOOTER / METADATA
        # ==========================================
        st.markdown("<hr style='border-color: #1F2937; margin-top: 40px;'>", unsafe_allow_html=True)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; color: #6B7280; font-family: JetBrains Mono; font-size: 0.7rem;'>
            <div>MODEL VERSION: 3.1.4-STABLE • SEED: 42</div>
            <div>LAST COMPILED: {current_time}</div>
            <div>FOR INTERNAL RESEARCH PURPOSES ONLY</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px; border: 1px solid #1F2937; margin-top: 50px; background-color: #111827; border-radius: 4px;'>
        <h2 style='color: #6B7280; font-family: "IBM Plex Sans", sans-serif; font-weight: 500; font-size: 1.2rem; letter-spacing: 1px;'>TERMINAL STANDBY</h2>
        <p style='color: #4B5563; font-family: "JetBrains Mono", monospace; font-size: 0.85rem;'>Awaiting initialization parameters. Click 'Initialize Backtest' to compile the ensemble.</p>
    </div>
    """, unsafe_allow_html=True)
