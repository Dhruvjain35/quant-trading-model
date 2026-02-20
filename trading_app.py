import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from core_engine import InstitutionalQuantEngine

st.set_page_config(page_title="Systematic Macro Terminal", layout="wide", initial_sidebar_state="expanded")

# --- INSTITUTIONAL DARK CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #8b949e; font-family: 'Inter', sans-serif; }
    .metric-box { background-color: #161b22; border: 1px solid #1f2937; padding: 15px; border-radius: 4px; border-top: 3px solid #00f2fe; margin-bottom: 10px;}
    .metric-title { font-size: 0.75rem; color: #8b949e; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;}
    .metric-value { font-size: 1.8rem; color: #e6edf3; font-family: 'Space Mono', monospace; font-weight: 700;}
    .metric-sub { font-size: 0.75rem; color: #2ea043; margin-top: 5px;}
    .section-header { font-family: 'Space Mono', monospace; color: #4facfe; border-bottom: 1px solid #1f2937; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h3 style='color: #00f2fe; font-family: monospace;'>⚙️ QUANT DESK V3</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

risk_asset = st.sidebar.text_input("High-Beta Asset", "QQQ")
safe_asset = st.sidebar.text_input("Risk-Free Asset", "SHY")

st.sidebar.markdown("### 📉 EXECUTION FRICTIONS")
tc_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
slip_bps = st.sidebar.slider("Slippage (bps)", 0, 20, 5)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.0, 40.0, 15.0) / 100.0

execute = st.sidebar.button("COMPILE & RUN OOS BACKTEST", use_container_width=True)

# --- MAIN DASHBOARD ---
st.markdown("<h1 style='font-family: monospace; color: #00f2fe; margin-bottom: 0px;'>MACRO-CONDITIONAL REGIME MODEL</h1>", unsafe_allow_html=True)
st.markdown("<p style='letter-spacing: 2px; color: #8b949e; font-size: 0.8rem;'>STRICT OUT-OF-SAMPLE VALIDATION • 3D VOLATILITY MAPPING • XG-BOOST ENSEMBLE</p>", unsafe_allow_html=True)

if execute:
    with st.spinner("Compiling institutional pipeline... (Extracting features, building 3D surfaces)"):
        # 1. Run the Engine
        engine = InstitutionalQuantEngine(risk_asset, safe_asset)
        engine.fetch_data()
        engine.engineer_features()
        engine.walk_forward_validation()
        bt = engine.backtest_with_frictions(tc_bps, slip_bps, tax_rate)
        stats = engine.calculate_statistics()
        
        # 2. Top Level Metrics (Visually Dense)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.markdown(f"<div class='metric-box'><div class='metric-title'>OOS SHARPE</div><div class='metric-value'>{stats['Sharpe']:.3f}</div><div class='metric-sub'>95% CI: [{stats['Sharpe_95CI'][0]:.2f}, {stats['Sharpe_95CI'][1]:.2f}]</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-box'><div class='metric-title'>NET CAGR</div><div class='metric-value'>{stats['CAGR']*100:.2f}%</div><div class='metric-sub'>Post-Tax & Frictions</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-box'><div class='metric-title'>MAX DRAWDOWN</div><div class='metric-value'>{stats['Max_DD']*100:.2f}%</div><div class='metric-sub'>Calmar: {stats['Calmar']:.2f}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-box'><div class='metric-title'>ANN. VOLATILITY</div><div class='metric-value'>{stats['Vol']*100:.2f}%</div><div class='metric-sub'>Daily Realized</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-box'><div class='metric-title'>SKEWNESS</div><div class='metric-value'>{stats['Skew']:.2f}</div><div class='metric-sub'>Tail Risk Adj.</div></div>", unsafe_allow_html=True)
        c6.markdown(f"<div class='metric-box'><div class='metric-title'>TOTAL RETURN</div><div class='metric-value'>{stats['Total_Ret']*100:.0f}%</div><div class='metric-sub'>vs Bench: {stats['Bench_Total']*100:.0f}%</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>01 — PORTFOLIO TRAJECTORY & UNDERWATER PROFILE</div>", unsafe_allow_html=True)
        
        # 3. Massive Main Chart (Equity + Drawdown)
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Strat'], name="Strategy (Net)", line=dict(color="#00f2fe", width=2)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['Cum_Bench'], name=f"{risk_asset} Benchmark", line=dict(color="#8b949e", width=1, dash="dot")), row=1, col=1)
        
        # Drawdown underwater
        fig1.add_trace(go.Scatter(x=bt.index, y=bt['Strat_DD'], name="Strategy DD", fill='tozeroy', line=dict(color="#ff3366", width=1)), row=2, col=1)
        
        fig1.update_layout(template="plotly_dark", plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14", height=500, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig1.update_yaxes(title_text="Cumulative Return (x)", row=1, col=1, gridcolor="#1f2937")
        fig1.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1, gridcolor="#1f2937")
        st.plotly_chart(fig1, use_container_width=True)

        # 4. 3D Volatility & Feature Space AND Rolling Metrics
        st.markdown("<div class='section-header'>02 — 3D REGIME MAPPING & ROLLING STABILITY</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            # The 3D Chart Request
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>XGBOOST FEATURE SPACE: VOLATILITY VS MOMENTUM VS SIGNAL PROBABILITY</span>", unsafe_allow_html=True)
            
            # Extract features for 3D plot
            df_3d = engine.X.copy()
            df_3d['XG_Probability'] = bt['Prob']
            df_3d['Position'] = bt['Target_Weight'].map({1.0: 'Risk-On (Equities)', 0.0: 'Risk-Off (Bonds)'})
            df_3d = df_3d.dropna().iloc[::5] # Sample every 5th day to keep 3D render fast
            
            fig_3d = px.scatter_3d(
                df_3d, x='Mom_3M', y='Vol_1M', z='XG_Probability', color='Position',
                color_discrete_map={'Risk-On (Equities)': '#00f2fe', 'Risk-Off (Bonds)': '#ff3366'},
                opacity=0.7, size_max=3
            )
            fig_3d.update_layout(
                template="plotly_dark", paper_bgcolor="#0b0e14", margin=dict(l=0, r=0, b=0, t=0), height=400,
                scene=dict(
                    xaxis_title='3M Momentum', yaxis_title='1M Realized Vol', zaxis_title='Model Confidence',
                    xaxis=dict(gridcolor='#1f2937', backgroundcolor='#0b0e14'),
                    yaxis=dict(gridcolor='#1f2937', backgroundcolor='#0b0e14'),
                    zaxis=dict(gridcolor='#1f2937', backgroundcolor='#0b0e14')
                ),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with col2:
            st.markdown("<span style='font-size: 0.85rem; color: #8b949e;'>ROLLING 12-MONTH SHARPE & VOLATILITY DYNAMICS</span>", unsafe_allow_html=True)
            roll_ret = bt['Post_Tax_Ret'].rolling(252).mean() * 252
            roll_vol = bt['Post_Tax_Ret'].rolling(252).std() * np.sqrt(252)
            roll_sharpe = roll_ret / roll_vol
            
            fig_roll = make_subplots(specs=[[{"secondary_y": True}]])
            fig_roll.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name="12M Sharpe", line=dict(color="#00f2fe")), secondary_y=False)
            fig_roll.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, name="12M Volatility", line=dict(color="#8a2be2", dash="dot")), secondary_y=True)
            
            fig_roll.update_layout(template="plotly_dark", plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14", height=400, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
            fig_roll.update_yaxes(title_text="Sharpe Ratio", gridcolor="#1f2937", secondary_y=False)
            fig_roll.update_yaxes(title_text="Volatility", showgrid=False, secondary_y=True)
            st.plotly_chart(fig_roll, use_container_width=True)

        # 5. Monthly Returns Heatmap
        st.markdown("<div class='section-header'>03 — MONTHLY RETURN DISTRIBUTION (CRISIS ALPHA CAPTURE)</div>", unsafe_allow_html=True)
        
        # Calculate Monthly Returns
        monthly_returns = bt['Post_Tax_Ret'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        df_hm = pd.DataFrame({'Year': monthly_returns.index.year, 'Month': monthly_returns.index.month, 'Return': monthly_returns.values})
        hm_pivot = df_hm.pivot(index='Year', columns='Month', values='Return')
        hm_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=hm_pivot.values, x=hm_pivot.columns, y=hm_pivot.index,
            colorscale=[[0.0, '#ff3366'], [0.5, '#0b0e14'], [1.0, '#00f2fe']],
            zmid=0, text=np.round(hm_pivot.values*100, 1), texttemplate="%{text}%", showscale=False
        ))
        fig_hm.update_layout(template="plotly_dark", plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14", height=400, margin=dict(l=0, r=0, t=10, b=0))
        fig_hm.update_yaxes(autorange="reversed", type='category')
        st.plotly_chart(fig_hm, use_container_width=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 100px; border: 1px dashed #1f2937; margin-top: 50px;'>
        <h2 style='color: #4facfe; font-family: monospace;'>[ SYSTEM IDLE ]</h2>
        <p style='color: #8b949e;'>Awaiting parameter confirmation. Click "Compile & Run OOS Backtest" to generate 3D models and performance analytics.</p>
    </div>
    """, unsafe_allow_html=True)
