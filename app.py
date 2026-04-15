"""
================================================================================
THE VISIONARY MODEL — Interactive Policy Console
================================================================================
A precision macroeconomic instrument connecting a continuous-time Neural ODE
physics engine with multi-agent policy optimization. Designed for institutional
research and strategic economic forecasting.

Run: streamlit run app.py
================================================================================
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="The Visionary Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DESIGN SYSTEM
# ==============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* ── Base Canvas ── */
    .stApp {
        background-color: #FAF9F6;
        font-family: 'DM Sans', sans-serif;
        color: #2A2A2A;
    }

    /* ── Masthead ── */
    .masthead {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 0.5rem;
    }
    .masthead-rule {
        width: 60px;
        height: 1px;
        background: #8B6914;
        margin: 0 auto 1.2rem auto;
    }
    .masthead-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3rem;
        font-weight: 600;
        color: #1C1C1C;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin: 0;
        line-height: 1.1;
    }
    .masthead-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 400;
        color: #9B9183;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.8rem;
    }
    .masthead-bottom-rule {
        width: 100%;
        height: 1px;
        background: linear-gradient(to right, transparent, #D4CFC4, transparent);
        margin-top: 1.5rem;
    }

    /* ── Diamond Ornament Separator ── */
    .ornament {
        text-align: center;
        color: #C4B99A;
        font-size: 0.7rem;
        letter-spacing: 8px;
        margin: 2rem 0;
    }

    /* ── Section Headers (Numbered) ── */
    .section-header {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1C1C1C;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #E8E4DD;
    }
    .section-number {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B6914;
        letter-spacing: 2px;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* ── Directive Block (replaces prompt box) ── */
    .directive-block {
        background: #F5F3EF;
        border-left: 3px solid #8B6914;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0 2rem 0;
        font-size: 0.9rem;
        color: #4A4A4A;
        line-height: 1.6;
    }
    .directive-block strong {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1rem;
        color: #1C1C1C;
        font-weight: 600;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #F5F3EF;
        border-right: 1px solid #E8E4DD;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-family: 'Cormorant Garamond', serif;
        color: #1C1C1C;
        font-size: 1.3rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #D4CFC4;
        padding-bottom: 0.8rem;
        margin-bottom: 1.2rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'DM Sans', sans-serif;
        color: #6B6358;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-status {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        color: #6B6358;
        padding: 0.25rem 0;
        border-bottom: 1px solid #EBE7E0;
    }
    .sidebar-status-ok { color: #2D5F5D; }
    .sidebar-status-err { color: #9B3B3B; }
    .sidebar-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #D4CFC4;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        color: #9B9183;
        line-height: 1.8;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background-color: #1C1C1C;
        color: #FAF9F6;
        border: none;
        border-radius: 0;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        padding: 0.85rem 2.5rem;
        transition: background-color 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #3A3A3A;
        color: #FAF9F6;
        box-shadow: none;
        transform: none;
    }
    .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: #6B6358;
        border: 1px solid #D4CFC4;
        border-radius: 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 1px;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        color: #4A4A4A;
        font-size: 0.88rem;
        letter-spacing: 0.5px;
    }

    /* ── Plotly ── */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }

    /* ── Metric overrides ── */
    [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 600;
        color: #1C1C1C;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #9B9183;
    }
    /* ── Metric delta text visibility ── */
    [data-testid="stMetricDelta"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        color: #1C1C1C !important;
        -webkit-text-fill-color: #1C1C1C !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #1C1C1C !important;
    }

    /* ── Expander header visibility on all states ── */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader:hover,
    .streamlit-expanderHeader:focus,
    .streamlit-expanderHeader:active,
    details summary,
    details summary:hover,
    details summary:focus,
    details summary:active,
    details[open] summary {
        color: #1C1C1C !important;
        -webkit-text-fill-color: #1C1C1C !important;
        background-color: transparent !important;
    }
    details summary span,
    details summary p {
        color: #1C1C1C !important;
        -webkit-text-fill-color: #1C1C1C !important;
    }

    /* ── Download / secondary button text visibility ── */
    .stDownloadButton > button,
    .stButton > button[kind="secondary"] {
        background-color: #FAF9F6 !important;
        color: #1C1C1C !important;
        -webkit-text-fill-color: #1C1C1C !important;
        border: 1px solid #D4CFC4 !important;
        border-radius: 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover,
    .stButton > button[kind="secondary"]:hover {
        background-color: #1C1C1C !important;
        color: #FAF9F6 !important;
        -webkit-text-fill-color: #FAF9F6 !important;
        border-color: #1C1C1C !important;
    }

    /* ── Main Footer ── */
    .main-footer {
        text-align: center;
        margin-top: 5rem;
        padding: 2rem 0 3rem 0;
        border-top: 1px solid #E8E4DD;
    }
    .main-footer .footer-brand {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1rem;
        font-weight: 600;
        color: #1C1C1C;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .main-footer .footer-detail {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        color: #9B9183;
        margin-top: 0.6rem;
        letter-spacing: 0.5px;
        line-height: 1.8;
    }

    /* ── Widget label visibility (Firefox) ── */
    .stCheckbox label,
    .stCheckbox label span,
    .stCheckbox label p,
    .stCheckbox span,
    .stTextInput label,
    .stTextInput label p,
    .stSlider label,
    .stSlider label p,
    .stSelectbox label,
    .stSelectbox label p,
    .stMultiSelect label,
    .stMultiSelect label p,
    .stNumberInput label,
    .stNumberInput label p {
        color: #1C1C1C !important;
        -webkit-text-fill-color: #1C1C1C !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CHART DESIGN SYSTEM
# ==============================================================================

CHART_LAYOUT = dict(
    template="simple_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=12, color="#4A4A4A"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=11, color="#6B6358"),
        orientation="h",
        yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    margin=dict(l=45, r=15, t=40, b=35),
    xaxis=dict(
        gridcolor="#EBE7E0", zeroline=False,
        title_font=dict(size=11, color="#9B9183"),
        tickfont=dict(size=10, color="#9B9183"),
        linecolor="#D4CFC4", linewidth=1,
    ),
    yaxis=dict(
        gridcolor="#EBE7E0", zeroline=False,
        title_font=dict(size=11, color="#9B9183"),
        tickfont=dict(size=10, color="#9B9183"),
        linecolor="#D4CFC4", linewidth=1,
    ),
)

COLORS = {
    "baseline": "#B8B0A2",
    "optimal": "#2D5F5D",
    "accent": "#8B6914",
    "danger": "#9B3B3B",
    "success": "#2D5F5D",
    "warning": "#A68B2D",
    "repo": "#5B7E8A",
    "fiscal": "#8B6914",
}


# ==============================================================================
# CHART BUILDERS
# ==============================================================================

def create_gdp_chart(baseline_df: pd.DataFrame, optimal_df: pd.DataFrame) -> go.Figure:
    """GDP comparison chart — baseline vs optimal trajectory."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=baseline_df.index, y=baseline_df["Real_GDP_Crore"] / 83e5,
        name="Baseline Projection", mode="lines",
        line=dict(color=COLORS["baseline"], width=1.5, dash="dot"),
        hovertemplate="Baseline: $%{y:.2f}T<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=optimal_df.index, y=optimal_df["Real_GDP_Crore"] / 83e5,
        name="Optimal Trajectory", mode="lines",
        line=dict(color=COLORS["optimal"], width=2.5),
        fill="tonexty", fillcolor="rgba(45,95,93,0.06)",
        hovertemplate="Optimal: $%{y:.2f}T<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Real GDP",
            font=dict(family="Cormorant Garamond", size=16, color="#1C1C1C"),
        ),
        yaxis_title="USD Trillions",
    )
    return fig


def create_inflation_chart(baseline_df: pd.DataFrame, optimal_df: pd.DataFrame,
                           max_inflation: float) -> go.Figure:
    """Inflation rate comparison with constraint line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=baseline_df.index, y=baseline_df["Inflation_Rate_Monthly_RBI"],
        name="Baseline", mode="lines",
        line=dict(color=COLORS["baseline"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=optimal_df.index, y=optimal_df["Inflation_Rate_Monthly_RBI"],
        name="Optimal", mode="lines",
        line=dict(color=COLORS["danger"], width=2.5),
    ))
    fig.add_hline(
        y=max_inflation, line_dash="dash", line_color=COLORS["warning"],
        annotation_text=f"Ceiling: {max_inflation}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#9B9183"),
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Inflation Rate",
            font=dict(family="Cormorant Garamond", size=16, color="#1C1C1C"),
        ),
        yaxis_title="%",
    )
    return fig


def create_unemployment_chart(baseline_df: pd.DataFrame, optimal_df: pd.DataFrame,
                              max_unemployment: float) -> go.Figure:
    """Unemployment rate comparison with constraint line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=baseline_df.index, y=baseline_df["Urban_Youth_Unemployment_Rate"],
        name="Baseline", mode="lines",
        line=dict(color=COLORS["baseline"], width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=optimal_df.index, y=optimal_df["Urban_Youth_Unemployment_Rate"],
        name="Optimal", mode="lines",
        line=dict(color=COLORS["accent"], width=2.5),
    ))
    fig.add_hline(
        y=max_unemployment, line_dash="dash", line_color=COLORS["warning"],
        annotation_text=f"Ceiling: {max_unemployment}%",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#9B9183"),
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Urban Youth Unemployment",
            font=dict(family="Cormorant Garamond", size=16, color="#1C1C1C"),
        ),
        yaxis_title="%",
    )
    return fig


def create_policy_chart(optimal_df: pd.DataFrame) -> go.Figure:
    """Step chart showing optimal policy lever values over time."""
    fig = go.Figure()

    if "Repo_Rate" in optimal_df.columns:
        fig.add_trace(go.Scatter(
            x=optimal_df.index, y=optimal_df["Repo_Rate"],
            name="Repo Rate", mode="lines",
            line=dict(color=COLORS["repo"], width=2.5, shape="hv"),
            hovertemplate="Repo: %{y:.2f}%<extra></extra>",
        ))

    if "Gross_Fiscal_Deficit_Percent_GDP" in optimal_df.columns:
        fig.add_trace(go.Scatter(
            x=optimal_df.index, y=optimal_df["Gross_Fiscal_Deficit_Percent_GDP"],
            name="Fiscal Deficit (% GDP)", mode="lines",
            line=dict(color=COLORS["fiscal"], width=2.5, shape="hv"),
            hovertemplate="Deficit: %{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="Policy Instruments",
            font=dict(family="Cormorant Garamond", size=16, color="#1C1C1C"),
        ),
        yaxis_title="%",
    )
    return fig


# ==============================================================================
# CACHED RESOURCE: Load the orchestrator once
# ==============================================================================

@st.cache_resource
def load_orchestrator(model_path: str, scaler_path: str, data_path: str,
                      api_key: str = None):
    """
    Load the VisionaryOrchestrator with the Neural ODE model + scaler.
    Cached so the heavy PyTorch model only loads once per session.
    """
    from visionary_agents import VisionaryOrchestrator
    return VisionaryOrchestrator(
        model_path=model_path,
        scaler_path=scaler_path,
        data_path=data_path,
        openai_api_key=api_key if api_key else None,
    )


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("# The Visionary Model")

    st.markdown("### System Dependencies")

    model_path = st.text_input(
        "Model Weights",
        value="visionary_ode_model.pth",
        help="Path to the trained .pth file",
    )
    scaler_path = st.text_input(
        "Scaler Configuration",
        value="visionary_scaler.pkl",
        help="Path to the fitted scaler .pkl file",
    )
    data_path = st.text_input(
        "Historical Dataset",
        value="Final_Visionary_Economy_Dataset_Prepared.csv",
        help="Path to the original CSV dataset",
    )

    st.markdown("### Language Model")

    mock_mode = st.checkbox("Run in offline mode (no API key)", value=True)

    api_key = ""
    if not mock_mode:
        api_key = st.text_input(
            "Hugging Face API Token",
            type="password",
            placeholder="hf_...",
            help="Required for the Strategist and Analyst agents",
        )

    st.markdown("### Diagnostics")

    model_found = os.path.exists(model_path)
    scaler_found = os.path.exists(scaler_path)
    data_found = os.path.exists(data_path)

    def _status_line(label, ok):
        cls = "sidebar-status-ok" if ok else "sidebar-status-err"
        tag = "Ready" if ok else "Missing"
        return f"<div class='sidebar-status'><span class='{cls}'>[{tag}]</span> {label}</div>"

    st.markdown(
        _status_line("Model weights", model_found)
        + _status_line("Scaler configuration", scaler_found)
        + _status_line("Trajectory dataset", data_found)
        + _status_line(
            "LLM routing" if not mock_mode else "LLM (offline mode)",
            mock_mode or bool(api_key),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Computation Budget")

    n_trials = st.slider(
        "Optuna trial count",
        min_value=50, max_value=2000, value=500, step=50,
        help="Higher count improves policy quality at the cost of runtime",
    )

    if "study" in st.session_state:
        st.markdown("### Convergence Log")
        if st.button("Show optimization history", use_container_width=True):
            st.session_state["show_opt_history"] = not st.session_state.get(
                "show_opt_history", False
            )

    st.markdown(
        "<div class='sidebar-footer'>"
        "The Visionary Model<br/>"
        "Build 1.0 / Neural ODE Kernel<br/>"
        "Continuous-Time Physics Engine"
        "</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# MASTHEAD
# ==============================================================================

st.markdown("""
<div class="masthead">
    <div class="masthead-rule"></div>
    <div class="masthead-title">The Visionary Model</div>
    <div class="masthead-subtitle">Continuous-Time Macroeconomic Orchestration</div>
    <div class="masthead-bottom-rule"></div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 01: Policy Configuration
# ==============================================================================

st.markdown(
    '<div class="section-header">'
    '<span class="section-number">01.</span>'
    'Target Configuration'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    target_gdp = st.slider(
        "Target Real GDP (USD Trillions)",
        min_value=2.0, max_value=15.0, value=7.0, step=0.1,
        format="$%.1fT",
    )

with col2:
    target_year = st.select_slider(
        "Target Year",
        options=list(range(2028, 2051)),
        value=2035,
    )

with col3:
    max_inflation = st.slider(
        "Inflation Ceiling",
        min_value=2.0, max_value=12.0, value=6.0, step=0.5,
        format="%.1f%%",
    )

with col4:
    max_unemployment = st.slider(
        "Unemployment Ceiling",
        min_value=5.0, max_value=40.0, value=25.0, step=1.0,
        format="%.0f%%",
    )

# --- Policy Lever Selection ---
available_levers = [
    "Repo_Rate",
    "Gross_Fiscal_Deficit_Percent_GDP",
    "Gross_Fixed_Capital_Formation_Percent_GDP",
    "GSec_10Y_Yield"
]

selected_levers = st.multiselect(
    "Authorized Policy Instruments",
    options=available_levers,
    default=["Repo_Rate", "Gross_Fiscal_Deficit_Percent_GDP", "Gross_Fixed_Capital_Formation_Percent_GDP"],
    help="Select which macroeconomic levers the optimizer is permitted to adjust.",
)

lever_string = ", ".join(selected_levers)

constructed_prompt = (
    f"Find a path to ${target_gdp:.1f} Trillion GDP by {target_year} "
    f"keeping inflation under {max_inflation:.1f}% "
    f"and unemployment under {max_unemployment:.0f}%. "
    f"Allowed levers: {lever_string}."
)

st.markdown(
    f"<div class='directive-block'>"
    f"<strong>Prompt</strong><br/>"
    f"{constructed_prompt}</div>",
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 02: Partial Observability (Advanced)
# ==============================================================================

with st.expander("Advanced: Partial Observability Initialization", expanded=False):
    st.markdown(
        "<div style='color: #6B6358; font-size: 0.88rem; margin-bottom: 1rem; line-height: 1.6;'>"
        "Inject observed real-world data into a future month to override the engine's "
        "imputed baseline. Leave fields blank to use the Neural ODE's own forward projection."
        "</div>",
        unsafe_allow_html=True,
    )

    use_partial_observability = st.checkbox("Enable partial observability", value=False)

    if use_partial_observability:
        target_month_input = st.text_input(
            "Target initialization month (YYYY-MM)", value="2026-03"
        )

        feature_names = []
        if scaler_found:
            try:
                import joblib
                feature_names = joblib.load(scaler_path).get("feature_names", [])
            except Exception:
                pass

        if not feature_names:
            st.warning(
                "Could not load features from scaler. Verify the scaler path."
            )

        st.markdown("##### State Vector Overrides")
        st.markdown(
            "<div style='color: #6B6358; font-size: 0.82rem; margin-bottom: 0.5rem;'>"
            "Select any variable and define its absolute value for time <b>t</b> and/or <b>t-1</b> "
            "to merge with the imputed baseline."
            "</div>",
            unsafe_allow_html=True,
        )

        override_df = pd.DataFrame(
            columns=["Parameter", "t_value", "t_minus_1_value"]
        )

        edited_override_df = st.data_editor(
            override_df,
            column_config={
                "Parameter": st.column_config.SelectboxColumn(
                    "Macroeconomic Variable",
                    options=feature_names,
                    required=True,
                ),
                "t_value": st.column_config.NumberColumn(
                    "Month t Value", required=False
                ),
                "t_minus_1_value": st.column_config.NumberColumn(
                    "Month t-1 Value", required=False
                ),
            },
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
        )

        user_overrides = {}
        for _, row in edited_override_df.dropna(subset=["Parameter"]).iterrows():
            feat = row["Parameter"]
            t_val = row["t_value"]
            tm1_val = row["t_minus_1_value"]

            if pd.notna(t_val) or pd.notna(tm1_val):
                user_overrides[feat] = {}
                if pd.notna(t_val):
                    user_overrides[feat]["t"] = float(t_val)
                if pd.notna(tm1_val):
                    user_overrides[feat]["t_minus_1"] = float(tm1_val)

    else:
        target_month_input = None
        user_overrides = None


# ==============================================================================
# EXECUTION
# ==============================================================================

all_files_ready = model_found and scaler_found and data_found
can_run = all_files_ready and (mock_mode or api_key)

if not all_files_ready:
    st.warning("One or more system dependencies are missing. Check the sidebar.")

st.markdown('<div class="ornament">&#9670;</div>', unsafe_allow_html=True)

run_button = st.button(
    "EXECUTE SIMULATION",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
)

if run_button:
    with st.status("Initializing simulation sequence...", expanded=True) as status:

        st.write("Loading continuous-time physics engine...")
        try:
            orchestrator = load_orchestrator(
                model_path=model_path,
                scaler_path=scaler_path,
                data_path=data_path,
                api_key=api_key if not mock_mode else None,
            )
        except Exception as e:
            st.error(f"Engine initialization failed: {e}")
            st.stop()

        st.write("Agent 1 / Strategist: Parsing quantitative directives...")
        time.sleep(0.3)

        st.write(
            f"Agents 2-3 / Optimizer + Architect: Conducting {n_trials} "
            f"trajectory integrations..."
        )

        # --- Live meter ---
        meter_text = st.empty()
        meter_bar = st.progress(0)

        def optuna_meter(study, trial):
            current_trial = len(study.trials)
            progress = min(current_trial / n_trials, 1.0)
            try:
                best_score = study.best_value
            except ValueError:
                best_score = 0.0

            meter_text.markdown(
                f"<div style='font-family: DM Sans, monospace; font-size: 0.88rem; "
                f"color: #6B6358; margin-bottom: 0.3rem;'>"
                f"Trial {current_trial} of {n_trials}"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Best score: {best_score:.2f}</div>",
                unsafe_allow_html=True,
            )
            meter_bar.progress(progress)

        try:
            report, study = orchestrator.run(
                user_prompt=constructed_prompt,
                n_trials=n_trials,
                progress_callback=optuna_meter,
                target_month=target_month_input,
                user_overrides=user_overrides,
            )
            st.write("Agent 4 / Analyst: Composing executive synthesis...")
            time.sleep(0.3)

            status.update(
                label="Simulation complete", state="complete", expanded=False
            )

        except Exception as e:
            status.update(label="Simulation interrupted", state="error")
            st.error(
                f"A numerical fault was encountered:\n\n```\n{e}\n```\n\n"
                f"This typically indicates ODE solver divergence or "
                f"scale incongruity in the initialization state.",
            )
            st.stop()

    # ==================================================================
    # Store results in session state
    # ==================================================================
    st.session_state["report"] = report
    st.session_state["study"] = study
    st.session_state["target_year"] = target_year
    st.session_state["target_gdp"] = target_gdp
    st.session_state["max_inflation"] = max_inflation
    st.session_state["max_unemployment"] = max_unemployment

    output_dir = os.path.dirname(model_path) or "."
    baseline_path = os.path.join(output_dir, "baseline_trajectory.csv")
    optimal_path = os.path.join(output_dir, "optimal_trajectory.csv")

    try:
        baseline_df = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
        optimal_df = pd.read_csv(optimal_path, index_col=0, parse_dates=True)
        st.session_state["baseline_df"] = baseline_df
        st.session_state["optimal_df"] = optimal_df
    except Exception as e:
        st.warning(f"Could not resolve trajectory outputs: {e}")


# ==============================================================================
# RESULTS (persisted via session state)
# ==============================================================================

if "report" in st.session_state:
    report = st.session_state["report"]
    baseline_df = st.session_state.get("baseline_df")
    optimal_df = st.session_state.get("optimal_df")
    tgt_gdp = st.session_state.get("target_gdp", 10.0)
    tgt_year = st.session_state.get("target_year", 2035)
    max_infl = st.session_state.get("max_inflation", 6.0)
    max_unemp = st.session_state.get("max_unemployment", 25.0)

    # ── Section 02: Performance Index ──
    if baseline_df is not None and optimal_df is not None:
        st.markdown(
            '<div class="section-header">'
            '<span class="section-number">02.</span>'
            'Performance Index'
            '</div>',
            unsafe_allow_html=True,
        )

        REAL_TO_NOMINAL_MULTIPLIER = 2.15

        baseline_target_year_df = baseline_df[
            baseline_df.index.year == tgt_year
        ]
        baseline_gdp_t = (
            baseline_target_year_df["Real_GDP_Crore"].sum() / 83e5
        )
        baseline_nom_gdp_t = baseline_gdp_t * REAL_TO_NOMINAL_MULTIPLIER

        optimal_target_year_df = optimal_df[
            optimal_df.index.year == tgt_year
        ]
        optimal_gdp_t = (
            optimal_target_year_df["Real_GDP_Crore"].sum() / 83e5
        )
        optimal_nom_gdp_t = optimal_gdp_t * REAL_TO_NOMINAL_MULTIPLIER

        gdp_delta = optimal_gdp_t - baseline_gdp_t

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Baseline Real GDP",
                f"${baseline_gdp_t:.2f}T",
                help="Inflation-adjusted physical volume (Base 2011)",
            )
            st.markdown(
                f"<div style='color:#9B9183; font-size:0.8rem; margin-top:-12px;'>"
                f"Nominal: <b>${baseline_nom_gdp_t:.2f}T</b></div>",
                unsafe_allow_html=True,
            )

        with m2:
            st.metric(
                "Optimal Real GDP",
                f"${optimal_gdp_t:.2f}T",
                delta=(
                    f"+${gdp_delta:.2f}T"
                    if gdp_delta >= 0
                    else f"${gdp_delta:.2f}T"
                ),
                help="Inflation-adjusted physical volume (Base 2011)",
            )
            st.markdown(
                f"<div style='color:#2D5F5D; font-size:0.8rem; margin-top:-12px;'>"
                f"Nominal: <b>${optimal_nom_gdp_t:.2f}T</b></div>",
                unsafe_allow_html=True,
            )

        with m3:
            avg_infl = optimal_df["Inflation_Rate_Monthly_RBI"].mean()
            infl_status = (
                "Within bounds" if avg_infl < max_infl else "Exceeds ceiling"
            )
            st.metric(
                "Avg. Inflation",
                f"{avg_infl:.1f}%",
                delta=f"{infl_status} ({max_infl}%)",
                delta_color="off",
            )

        with m4:
            avg_unemp = optimal_df["Urban_Youth_Unemployment_Rate"].mean()
            unemp_status = (
                "Within bounds" if avg_unemp < max_unemp else "Exceeds ceiling"
            )
            st.metric(
                "Avg. Unemployment",
                f"{avg_unemp:.1f}%",
                delta=f"{unemp_status} ({max_unemp}%)",
                delta_color="off",
            )

        # ── Section 03: Trajectory Analysis ──
        st.markdown(
            '<div class="section-header">'
            '<span class="section-number">03.</span>'
            'Trajectory Analysis'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("show_opt_history") and "study" in st.session_state:
            from optuna.visualization import plot_optimization_history

            fig_opt = plot_optimization_history(st.session_state["study"])
            fig_opt.update_layout(
                title=dict(
                    text="Convergence History",
                    font=dict(family="Cormorant Garamond", size=16, color="#1C1C1C"),
                ),
                plot_bgcolor="#FAF9F6",
                paper_bgcolor="#FAF9F6",
                font=dict(color="#6B6358", family="DM Sans"),
                xaxis=dict(
                    title_font=dict(color="#1C1C1C"),
                    tickfont=dict(color="#1C1C1C"),
                    gridcolor="#EBE7E0",
                    zerolinecolor="#D4CFC4"
                ),
                yaxis=dict(
                    title_font=dict(color="#1C1C1C"),
                    tickfont=dict(color="#1C1C1C"),
                    gridcolor="#EBE7E0",
                    zerolinecolor="#D4CFC4"
                )
            )
            st.plotly_chart(fig_opt, use_container_width=True)

        chart_row1_left, chart_row1_right = st.columns(2)

        with chart_row1_left:
            st.plotly_chart(
                create_gdp_chart(baseline_df, optimal_df),
                use_container_width=True,
                key="gdp_chart",
            )

        with chart_row1_right:
            st.plotly_chart(
                create_inflation_chart(baseline_df, optimal_df, max_infl),
                use_container_width=True,
                key="inflation_chart",
            )

        chart_row2_left, chart_row2_right = st.columns(2)

        with chart_row2_left:
            st.plotly_chart(
                create_unemployment_chart(baseline_df, optimal_df, max_unemp),
                use_container_width=True,
                key="unemployment_chart",
            )

        with chart_row2_right:
            st.plotly_chart(
                create_policy_chart(optimal_df),
                use_container_width=True,
                key="policy_chart",
            )

        # ── Downloads ──
        st.markdown('<div class="ornament">&#9670;</div>', unsafe_allow_html=True)

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "Export Baseline (.csv)",
                baseline_df.to_csv(),
                file_name="baseline_trajectory.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Export Optimal (.csv)",
                optimal_df.to_csv(),
                file_name="optimal_trajectory.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "Export Report (.md)",
                report,
                file_name="visionary_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ── Section 04: Executive Report ──
    st.markdown(
        '<div class="section-header">'
        '<span class="section-number">04.</span>'
        'Policy Report'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Review complete analytical report", expanded=True):
        st.markdown(report)

    # ── Footer ──
    st.markdown("""
    <div class="main-footer">
        <div class="footer-brand">The Visionary Model</div>
        <div class="footer-detail">
            Continuous-Time Neural ODE Physics Engine<br/>
            Multi-Agent Policy Optimization with Qwen2.5-72B Synthesis<br/>
            Engineered for institutional macroeconomic research
        </div>
    </div>
    """, unsafe_allow_html=True)
