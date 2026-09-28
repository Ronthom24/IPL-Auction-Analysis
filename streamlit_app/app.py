"""
IPL Auction Intelligence Dashboard
Real, independently-sourced IPL auction + player data (Cricsheet + published
auction records) -> Random Forest price model + SHAP explainability.

No sold/unsold classifier: published auction results only list players who
were bought, so there is no real "unsold" ground truth to train one on.
See docs/DATA_QUALITY.md.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os, shap

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Auction Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
INK       = "#12172B"
SLATE     = "#4A5170"
PAPER     = "#F6F7FB"
CARD      = "#FFFFFF"
BORDER    = "#E4E7F1"
ACCENT    = "#3A5CE0"
ACCENT_2  = "#0FB5AE"
POSITIVE  = "#1E9E6B"
NEGATIVE  = "#D6455B"
AMBER     = "#E0982A"

ROLE_COLORS = {
    "Batsman": "#3A5CE0",
    "Bowler": "#D6455B",
    "All-Rounder": "#E0982A",
    "Wicketkeeper-Batsman": "#7A5CF0",
}

FRANCHISE_COLORS = {
    "Chennai Super Kings": "#F2C438",
    "Mumbai Indians": "#1D5EC4",
    "Royal Challengers Bengaluru": "#D6303F",
    "Kolkata Knight Riders": "#5B3A8E",
    "Delhi Capitals": "#2A6FDB",
    "Punjab Kings": "#DC1F35",
    "Rajasthan Royals": "#E23FA0",
    "Sunrisers Hyderabad": "#E8791A",
    "Gujarat Titans": "#1B2A4A",
    "Lucknow Super Giants": "#38A6B5",
    "Deccan Chargers": "#5C6B7A",
    "Gujarat Lions": "#E8641A",
    "Pune Warriors India": "#7A4FA3",
    "Kochi Tuskers Kerala": "#E8734F",
    "Rising Pune Supergiants": "#B5222E",
}

FONT = "'Segoe UI', -apple-system, Helvetica, Arial, sans-serif"

PLOTLY_LAYOUT = dict(
    font=dict(family=FONT, color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def style_fig(fig, title=None, height=340):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color=INK), x=0))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ─── LOAD ASSETS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf        = joblib.load(f"{MODEL_DIR}/price_predictor.pkl")
    explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
    le_role   = joblib.load(f"{MODEL_DIR}/le_role.pkl")
    le_nat    = joblib.load(f"{MODEL_DIR}/le_nat.pkl")
    return rf, explainer, le_role, le_nat


@st.cache_data
def load_data():
    df      = pd.read_csv(f"{MODEL_DIR}/processed_data.csv")
    top_val = pd.read_csv(f"{MODEL_DIR}/top_value_players.csv")
    fran    = pd.read_csv(f"{MODEL_DIR}/franchise_efficiency.csv")

    rename_map = {
        "playername": "player_name",
        "battingavg": "batting_avg",
        "battingsr": "batting_sr",
        "runsscored": "runs_scored",
        "economyrate": "economy_rate",
        "basepricecr": "base_price_cr",
        "soldpricecr": "sold_price_cr",
        "performancescore": "performance_score",
        "valueindex": "value_index",
        "bowlingavg": "bowling_avg",
        "bowlingsr": "bowling_sr",
    }
    df      = df.rename(columns=rename_map)
    top_val = top_val.rename(columns=rename_map)

    with open(f"{MODEL_DIR}/feature_names.json") as f:
        features = json.load(f)
    with open(f"{MODEL_DIR}/regression_metrics.json") as f:
        metrics = json.load(f)
    with open(f"{MODEL_DIR}/summary_stats.json") as f:
        stats = json.load(f)
    return df, top_val, fran, features, metrics, stats


rf, explainer, le_role, le_nat = load_models()
df, top_val, fran_eff, FEATURES, reg_metrics, stats = load_data()

ROLES      = sorted([r for r in df['role'].unique() if pd.notna(r)])
NATIONS    = sorted([n for n in df['nationality'].unique() if pd.notna(n)])
FRANCHISES = sorted(df['current_franchise'].unique())
YEARS      = sorted(df['year'].unique())

df["display_franchise"] = df["current_franchise"]
for role in ROLES:
    ROLE_COLORS.setdefault(role, SLATE)
for fr in FRANCHISES:
    FRANCHISE_COLORS.setdefault(fr, SLATE)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    html, body, [class*="css"] {{ font-family: {FONT}; }}
    .stApp {{ background: {PAPER}; }}
    #MainMenu, footer {{ visibility: hidden; }}

    section[data-testid="stSidebar"] {{
        background: {INK};
    }}
    section[data-testid="stSidebar"] * {{ color: #E8EAF6 !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.92rem; }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); }}

    h1, h2, h3 {{ color: {INK} !important; font-weight: 700 !important; }}
    h1 {{ letter-spacing: -0.02em; }}

    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 2px rgba(18,23,43,0.04);
    }}
    div[data-testid="stMetricLabel"] {{ color: {SLATE} !important; font-size: 0.8rem !important; }}
    div[data-testid="stMetricValue"] {{ color: {INK} !important; }}

    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(18,23,43,0.05);
        margin-bottom: 14px;
    }}
    .callout {{
        border-radius: 12px;
        padding: 14px 18px;
        margin: 10px 0 18px 0;
        font-size: 0.94rem;
        line-height: 1.5;
    }}
    .callout-info    {{ background: #EEF2FD; border-left: 4px solid {ACCENT}; color: {INK}; }}
    .callout-success {{ background: #EAF7F1; border-left: 4px solid {POSITIVE}; color: {INK}; }}
    .callout-warn    {{ background: #FDF3E4; border-left: 4px solid {AMBER}; color: {INK}; }}
    .callout b {{ color: {INK}; }}

    .pill {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
        background: #EEF2FD; color: {ACCENT}; margin-right: 6px;
    }}

    div[data-testid="stForm"] {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 18px 22px;
    }}
    .stButton>button, .stFormSubmitButton>button {{
        background: {ACCENT}; color: white; border-radius: 8px; border: none;
        font-weight: 600;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{ background: #2C48C4; }}

    div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


def callout(kind, title, body):
    st.markdown(
        f'<div class="callout callout-{kind}"><b>{title}</b><br>{body}</div>',
        unsafe_allow_html=True,
    )


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 IPL Auction Intelligence")
    st.caption("Real auction data + real ball-by-ball stats — no synthetic numbers")
    st.divider()

    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 Player Analysis",
        "💡 Value Finder",
        "🤖 Price Predictor",
        "🏢 Franchise Insights",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.82rem; line-height:1.7; opacity:0.85;">
    <b>Dataset</b><br>
    {stats['total_records']} auction records · {stats['unique_players']} players<br>
    {stats['years'][0]}–{stats['years'][-1]} · {len(FRANCHISES)} franchises<br><br>
    <b>Model</b><br>
    Random Forest · headline R² {reg_metrics['headline_r2_grouped_cv_mean']:.2f}<br>
    (5-fold CV, grouped by player)
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("Sold/unsold classification isn't offered — real auction records only list who was bought, so there's no real negative class to train on.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Overview")
    st.caption("Ten seasons of IPL auction activity, from real published records.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Auction Records", f"{stats['total_records']:,}")
    c2.metric("Unique Players", f"{stats['unique_players']:,}")
    c3.metric("Avg Sale Price", f"₹{stats['avg_price_overall']} Cr")
    c4.metric("Highest Bid", f"₹{stats['max_price']} Cr")

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Auction Price Trend")
        yearly = df.groupby('year')['sold_price_cr'].agg(['mean', 'median']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['mean'].round(2),
                                  mode='lines+markers', name='Mean',
                                  line=dict(color=ACCENT, width=2.5), marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['median'].round(2),
                                  mode='lines+markers', name='Median',
                                  line=dict(color=ACCENT_2, width=2, dash='dash'), marker=dict(size=6)))
        fig.update_layout(yaxis_title="₹ Crore", xaxis_title="")
        st.plotly_chart(style_fig(fig), width='stretch')

    with col2:
        st.markdown("#### Price Distribution by Role")
        fig = go.Figure()
        for role in ROLES:
            data = df[df['role'] == role]['sold_price_cr']
            fig.add_trace(go.Histogram(x=data, name=role, opacity=0.75,
                                        marker_color=ROLE_COLORS[role], nbinsx=20))
        fig.update_layout(barmode='overlay', xaxis_title="Sold Price (₹ Cr)", yaxis_title="Count")
        st.plotly_chart(style_fig(fig), width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Franchise Total Spend by Year")
        pivot = df.groupby(['year', 'display_franchise'])['sold_price_cr'].sum().reset_index()
        top6 = fran_eff.sort_values('total_spend', ascending=False)['franchise'].head(6).tolist()
        pivot_top = pivot[pivot['display_franchise'].isin(top6)]
        fig = px.line(pivot_top, x='year', y='sold_price_cr', color='display_franchise',
                       markers=True, color_discrete_map=FRANCHISE_COLORS,
                       labels={'sold_price_cr': 'Total Spend (₹ Cr)', 'year': '', 'display_franchise': ''})
        st.plotly_chart(style_fig(fig), width='stretch')

    with col4:
        st.markdown("#### Nationality Mix")
        nat_counts = df['nationality'].value_counts().reset_index()
        nat_counts.columns = ['nationality', 'count']
        fig = px.pie(nat_counts, names='nationality', values='count', hole=0.55,
                      color_discrete_sequence=[ACCENT, ACCENT_2, SLATE])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(style_fig(fig), width='stretch')

    st.markdown("#### Notes")
    best_value_team = fran_eff.sort_values('avg_value_index', ascending=False).iloc[0]
    biggest_spender = fran_eff.sort_values('total_spend', ascending=False).iloc[0]
    first_yr, last_yr = df['year'].min(), df['year'].max()
    p0 = df[df['year'] == first_yr]['sold_price_cr'].mean()
    p1 = df[df['year'] == last_yr]['sold_price_cr'].mean()
    growth = (p1 / p0 - 1) * 100 if p0 else 0

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        callout("success", "Best value franchise",
                f"{best_value_team['franchise']} gets the most performance per ₹ crore spent "
                f"(avg value index: {best_value_team['avg_value_index']:.1f}).")
    with ic2:
        callout("info", "Biggest spender",
                f"{biggest_spender['franchise']} — ₹{biggest_spender['total_spend']:.0f} Cr total, "
                f"₹{biggest_spender['avg_price']:.1f} Cr avg per player.")
    with ic3:
        callout("warn", "Price drift",
                f"Avg price moved {growth:+.0f}% from {first_yr} to {last_yr}. "
                f"Small samples in early/late years make year-on-year swings noisy — treat as a trend, not a precise rate.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Player Analysis":
    st.title("Player Analysis")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_role = st.selectbox("Role", ["All"] + ROLES)
    with col_f2:
        sel_nat = st.selectbox("Nationality", ["All"] + NATIONS)
    with col_f3:
        year_range = st.select_slider("Year range", options=YEARS, value=(YEARS[0], YEARS[-1]))

    fdf = df.copy()
    if sel_role != "All": fdf = fdf[fdf['role'] == sel_role]
    if sel_nat != "All":  fdf = fdf[fdf['nationality'] == sel_nat]
    fdf = fdf[(fdf['year'] >= year_range[0]) & (fdf['year'] <= year_range[1])]

    st.caption(f"{len(fdf)} auction records match these filters")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Performance Score vs Price")
        fig = px.scatter(fdf, x='performance_score', y='sold_price_cr',
                          color='role', size='career_matches',
                          hover_data=['player_name', 'year', 'display_franchise'],
                          color_discrete_map=ROLE_COLORS,
                          labels={'sold_price_cr': 'Sold Price (₹ Cr)', 'performance_score': 'Performance Score'})
        st.plotly_chart(style_fig(fig), width='stretch')

    with col2:
        st.markdown("#### Career Matches vs Price")
        fig = px.density_heatmap(fdf, x='career_matches', y='sold_price_cr',
                                  nbinsx=15, nbinsy=15,
                                  color_continuous_scale=[[0, PAPER], [0.5, ACCENT_2], [1, ACCENT]],
                                  labels={'sold_price_cr': 'Sold Price (₹ Cr)', 'career_matches': 'Career Matches'})
        st.plotly_chart(style_fig(fig), width='stretch')

    st.markdown("#### Batting Strike Rate vs Price (non-bowlers)")
    non_bowl = fdf[(fdf['role'] != 'Bowler') & fdf['batting_sr'].notna()]
    if len(non_bowl) > 3:
        fig = px.scatter(non_bowl, x='batting_sr', y='sold_price_cr',
                          color='role', trendline='ols', hover_data=['player_name', 'year'],
                          color_discrete_map=ROLE_COLORS,
                          labels={'sold_price_cr': 'Price (₹ Cr)', 'batting_sr': 'Batting Strike Rate'})
        st.plotly_chart(style_fig(fig, height=320), width='stretch')
    else:
        st.info("Not enough rows with a recorded strike rate for this filter combination.")

    st.markdown("#### Player Records")
    display_cols = ['player_name', 'role', 'nationality', 'year', 'career_matches', 'batting_avg',
                     'batting_sr', 'wickets', 'economy_rate', 'base_price_cr', 'sold_price_cr',
                     'display_franchise', 'value_index']
    st.dataframe(
        fdf[display_cols].rename(columns={'display_franchise': 'franchise'})
           .sort_values('sold_price_cr', ascending=False).reset_index(drop=True),
        width='stretch', height=320,
    )
    st.caption("Blank batting/bowling cells are real — that player has no recorded innings/overs of that kind, not a data error.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VALUE FINDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Value Finder":
    st.title("Value Finder")
    st.caption("Ranks players by performance score per ₹ crore spent. Read the note below before treating this as a discovery tool.")

    callout("warn", "How to read this",
            "Value index rewards a decent performance score at a low price — so it's structurally "
            "biased toward players bought near the ₹0.2–0.5 Cr base-price floor, not necessarily "
            "toward genuinely mispriced stars. Treat the list as a ranking heuristic, not a verified "
            "'undervalued player' detector — there's no outcome data (e.g. next-season performance) "
            "checking whether these picks actually paid off.")

    col1, col2 = st.columns([1, 2])
    with col1:
        budget = st.slider("Max price per player (₹ Cr)", 0.1, 15.0, 8.0, 0.1)
        role_filter = st.multiselect("Roles", ROLES, default=ROLES)
        nat_filter = st.multiselect("Nationality", NATIONS, default=NATIONS)
        top_n = st.slider("Show top N", 5, 20, 10)

    filtered = df[
        (df['sold_price_cr'] <= budget) &
        (df['role'].isin(role_filter)) &
        (df['nationality'].isin(nat_filter)) &
        (df['value_index'].notna())
    ].copy()

    top_players = filtered.nlargest(top_n, 'value_index').reset_index(drop=True)

    with col2:
        st.markdown(f"#### Top {top_n} value picks (≤ ₹{budget} Cr)")
        if len(top_players) == 0:
            st.warning("No players match these filters — try widening the budget or roles.")
        else:
            fig = go.Figure(go.Bar(
                y=top_players['player_name'], x=top_players['value_index'], orientation='h',
                marker_color=[ROLE_COLORS[r] for r in top_players['role']],
                text=[f"₹{p:.1f}Cr" for p in top_players['sold_price_cr']],
                textposition='inside',
            ))
            fig.update_layout(xaxis_title="Value Index", yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, height=380), width='stretch')

    if len(top_players) > 0:
        st.markdown("#### Budget Optimizer — greedy XI within a cap")
        st.caption("Greedy by value index within role quotas and a total cap — a fast approximation, not a globally optimal solver (no ILP/knapsack solve).")
        cap = st.number_input("Total squad budget (₹ Cr)", value=80.0, min_value=10.0, max_value=200.0)

        pool = filtered.sort_values('value_index', ascending=False).copy()
        selected, total_spend = [], 0.0
        role_counts = {r: 0 for r in ROLES}
        role_limits = {"Batsman": 5, "Bowler": 5, "All-Rounder": 4, "Wicketkeeper-Batsman": 2}

        for _, row in pool.iterrows():
            if len(selected) >= 11: break
            limit = role_limits.get(row['role'], 3)
            if total_spend + row['sold_price_cr'] <= cap and role_counts[row['role']] < limit:
                selected.append(row)
                total_spend += row['sold_price_cr']
                role_counts[row['role']] += 1

        if selected:
            sel_df = pd.DataFrame(selected)[
                ['player_name', 'role', 'nationality', 'sold_price_cr', 'performance_score', 'value_index']
            ].reset_index(drop=True)
            sel_df.index += 1

            c1, c2, c3 = st.columns(3)
            c1.metric("Players Selected", len(selected))
            c2.metric("Total Spend", f"₹{total_spend:.1f} Cr")
            c3.metric("Budget Remaining", f"₹{cap - total_spend:.1f} Cr")
            st.dataframe(sel_df, width='stretch')
        else:
            st.info("No combination fits this budget and role quota — try raising the cap.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRICE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Price Predictor":
    st.title("Price Predictor")

    callout("info", "Model",
            f"Random Forest Regressor · headline R² (5-fold CV, grouped by player) = "
            f"<b>{reg_metrics['headline_r2_grouped_cv_mean']:.2f}</b> · "
            f"holdout MAE ≈ ₹{reg_metrics['holdout_mae_cr']:.2f} Cr. "
            f"This explains roughly a third of price variance — auction prices carry real noise "
            f"(bidding wars, franchise-specific needs, hype) that stats alone can't capture. "
            f"Treat outputs as a rough anchor, not a quote.")

    with st.form("predictor_form"):
        st.markdown("##### Player Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            role = st.selectbox("Role", ROLES)
            nationality = st.selectbox("Nationality", NATIONS)
            year = st.selectbox("Auction Year", YEARS[::-1])
            career_matches = st.slider("Career Matches (at auction time)", 0, 250, 30)
            base_price = st.selectbox("Base Price (₹ Cr)", [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])

        with c2:
            st.markdown("**Batting**")
            no_batting = st.checkbox("No batting record (recognised bowler)", value=(role == "Bowler"))
            bat_avg = st.slider("Batting Average", 5.0, 65.0, 28.0, 0.5, disabled=no_batting)
            bat_sr = st.slider("Batting Strike Rate", 80.0, 210.0, 130.0, 1.0, disabled=no_batting)
            runs = st.slider("Career Runs", 0, 6000, 500, 10, disabled=no_batting)
            fours = st.slider("Career Fours", 0, 600, 40, disabled=no_batting)
            sixes = st.slider("Career Sixes", 0, 400, 20, disabled=no_batting)

        with c3:
            st.markdown("**Bowling**")
            no_bowling = st.checkbox("No bowling record (specialist batter)", value=(role in ("Batsman", "Wicketkeeper-Batsman")))
            wickets = st.slider("Career Wickets", 0, 400, 20, disabled=no_bowling)
            economy = st.slider("Economy Rate", 5.0, 12.0, 8.0, 0.1, disabled=no_bowling)
            bowl_avg = st.slider("Bowling Average", 12.0, 55.0, 28.0, 0.5, disabled=no_bowling)
            bowl_sr = st.slider("Bowling Strike Rate", 8.0, 35.0, 20.0, 0.5, disabled=no_bowling)
            catches = st.slider("Career Catches", 0, 150, 15)
            stumpings = st.slider("Career Stumpings (keepers)", 0, 80, 0)

        submitted = st.form_submit_button("Predict Auction Price", width='stretch')

    if submitted:
        def minmax_lookup(col, val):
            lo, hi = df[col].min(skipna=True), df[col].max(skipna=True)
            if pd.isna(val) or hi == lo: return np.nan
            return (val - lo) / (hi - lo)

        bat_avg_v = np.nan if no_batting else bat_avg
        bat_sr_v = np.nan if no_batting else bat_sr
        wickets_v = 0 if no_bowling else wickets
        economy_v = np.nan if no_bowling else economy
        bowl_avg_v = np.nan if no_bowling else bowl_avg
        bowl_sr_v = np.nan if no_bowling else bowl_sr

        norm_avg = minmax_lookup('batting_avg', bat_avg_v)
        norm_sr = minmax_lookup('batting_sr', bat_sr_v)
        norm_wkts = minmax_lookup('wickets', wickets_v)
        norm_econ_inv = minmax_lookup('economy_rate', -economy_v if not no_bowling else np.nan)
        norm_stump = minmax_lookup('value_index', stumpings)  # placeholder scale, keeper bump is small regardless

        def nanmean(*vals):
            vals = [v for v in vals if not pd.isna(v)]
            return float(np.mean(vals)) if vals else np.nan

        batting_component = nanmean(norm_avg, norm_sr)
        bowling_component = nanmean(norm_wkts, norm_econ_inv)

        if role == "Batsman":
            perf = batting_component
        elif role == "Bowler":
            perf = bowling_component
        elif role == "All-Rounder":
            perf = nanmean(batting_component, bowling_component) if not (pd.isna(batting_component) or pd.isna(bowling_component)) else nanmean(batting_component, bowling_component)
        else:
            perf = nanmean(0.8 * batting_component if not pd.isna(batting_component) else np.nan,
                            0.2 * min(1.0, stumpings / 40))
        perf = (perf * 100) if not pd.isna(perf) else 0.0

        role_enc = int(le_role.transform([role])[0])
        nat_enc = int(le_nat.transform([nationality])[0])

        input_data = pd.DataFrame([{
            'role_enc': role_enc, 'nationality_enc': nat_enc, 'year': year,
            'career_matches': career_matches,
            'battingavg': bat_avg_v, 'battingsr': bat_sr_v, 'runsscored': (0 if no_batting else runs),
            'fours': (0 if no_batting else fours), 'sixes': (0 if no_batting else sixes),
            'wickets': wickets_v, 'economyrate': economy_v,
            'bowlingavg': bowl_avg_v, 'bowlingsr': bowl_sr_v,
            'catches': catches, 'stumpings': stumpings,
            'basepricecr': base_price,
            'battingavg_missing': int(no_batting), 'battingsr_missing': int(no_batting),
            'economyrate_missing': int(no_bowling), 'bowlingavg_missing': int(no_bowling),
        }])[FEATURES]

        predicted_price = float(rf.predict(input_data)[0])
        predicted_price = max(base_price, predicted_price)

        st.divider()
        st.subheader("Prediction")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Price", f"₹{predicted_price:.2f} Cr")
        r2.metric("Base Price", f"₹{base_price:.2f} Cr")
        r3.metric("Performance Score", f"{perf:.1f}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_price,
            gauge={
                'axis': {'range': [0, 25], 'ticksuffix': 'Cr'},
                'bar': {'color': ACCENT},
                'steps': [
                    {'range': [0, 5], 'color': '#EAF7F1'},
                    {'range': [5, 12], 'color': '#FDF3E4'},
                    {'range': [12, 25], 'color': '#FBEAEE'},
                ],
            },
            title={'text': "Predicted Auction Price (₹ Cr)"}
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### Why this price? (SHAP)")
        shap_vals = explainer.shap_values(input_data)[0]
        feat_names_display = {
            'role_enc': 'Role', 'nationality_enc': 'Nationality', 'year': 'Year',
            'career_matches': 'Career Matches', 'battingavg': 'Batting Avg', 'battingsr': 'Batting SR',
            'runsscored': 'Runs', 'fours': 'Fours', 'sixes': 'Sixes', 'wickets': 'Wickets',
            'economyrate': 'Economy', 'bowlingavg': 'Bowling Avg', 'bowlingsr': 'Bowling SR',
            'catches': 'Catches', 'stumpings': 'Stumpings', 'basepricecr': 'Base Price',
            'battingavg_missing': 'No Batting Record', 'battingsr_missing': 'No Batting Record (SR)',
            'economyrate_missing': 'No Bowling Record', 'bowlingavg_missing': 'No Bowling Record (Avg)',
        }
        shap_df = pd.DataFrame({
            'Feature': [feat_names_display.get(f, f) for f in FEATURES],
            'SHAP Value': shap_vals,
        }).sort_values('SHAP Value', key=abs, ascending=False).head(10)

        fig = go.Figure(go.Bar(
            y=shap_df['Feature'][::-1], x=shap_df['SHAP Value'][::-1], orientation='h',
            marker_color=[POSITIVE if v > 0 else NEGATIVE for v in shap_df['SHAP Value'][::-1]],
            text=[f"+₹{v:.2f}Cr" if v > 0 else f"₹{v:.2f}Cr" for v in shap_df['SHAP Value'][::-1]],
            textposition='outside',
        ))
        fig.add_vline(x=0, line_color=BORDER, line_width=1)
        fig.update_layout(xaxis_title="Price Impact (₹ Cr)", margin=dict(l=10, r=70, t=10, b=10))
        st.plotly_chart(style_fig(fig, height=340), width='stretch')

        top_pos = shap_df[shap_df['SHAP Value'] > 0].iloc[0] if (shap_df['SHAP Value'] > 0).any() else None
        top_neg = shap_df[shap_df['SHAP Value'] < 0].iloc[0] if (shap_df['SHAP Value'] < 0).any() else None
        explanation = f"The model predicts <b>₹{predicted_price:.2f} Cr</b>. "
        if top_pos is not None:
            explanation += f"Biggest upward driver: <b>{top_pos['Feature']}</b> (+₹{top_pos['SHAP Value']:.2f} Cr). "
        if top_neg is not None:
            explanation += f"Biggest downward driver: <b>{top_neg['Feature']}</b> (₹{top_neg['SHAP Value']:.2f} Cr)."
        callout("info", "In plain English", explanation)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FRANCHISE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Franchise Insights":
    st.title("Franchise Insights")
    st.caption("Renamed franchises (Delhi Daredevils→Capitals, Kings XI→Punjab Kings, RCB Bangalore→Bengaluru) are merged under their current identity.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Spend vs Avg Performance")
        fig = px.scatter(fran_eff, x='total_spend', y='avg_performance',
                          size='players_bought', color='avg_value_index',
                          hover_data=['franchise'], text='franchise',
                          color_continuous_scale=[[0, "#D6455B"], [0.5, "#E0982A"], [1, "#1E9E6B"]],
                          labels={'total_spend': 'Total Spend (₹ Cr)', 'avg_performance': 'Avg Performance', 'avg_value_index': 'Value Index'})
        fig.update_traces(textposition='top center', textfont_size=9)
        st.plotly_chart(style_fig(fig), width='stretch')

    with col2:
        st.markdown("#### Efficiency Ranking (avg value index)")
        eff_sorted = fran_eff.sort_values('avg_value_index', ascending=True)
        colors = [FRANCHISE_COLORS.get(f, SLATE) for f in eff_sorted['franchise']]
        fig = go.Figure(go.Bar(
            y=eff_sorted['franchise'], x=eff_sorted['avg_value_index'], orientation='h',
            marker_color=colors, text=[f"{v:.1f}" for v in eff_sorted['avg_value_index']], textposition='inside',
        ))
        fig.update_layout(xaxis_title="Avg Value Index")
        st.plotly_chart(style_fig(fig), width='stretch')

    st.markdown("#### Spend Over Time")
    yearly_fran = df.groupby(['year', 'display_franchise'])['sold_price_cr'].sum().reset_index()
    fig = px.area(yearly_fran, x='year', y='sold_price_cr', color='display_franchise',
                   color_discrete_map=FRANCHISE_COLORS,
                   labels={'sold_price_cr': 'Total Spend (₹ Cr)', 'year': '', 'display_franchise': ''})
    st.plotly_chart(style_fig(fig, height=360), width='stretch')

    st.markdown("#### Franchise Table")
    display_fran = fran_eff.rename(columns={
        'franchise': 'Franchise', 'total_spend': 'Total Spend (₹ Cr)',
        'avg_price': 'Avg Price (₹ Cr)', 'avg_value_index': 'Avg Value Index',
        'players_bought': 'Players Bought', 'avg_performance': 'Avg Performance Score'
    }).sort_values('Avg Value Index', ascending=False).reset_index(drop=True)
    display_fran.index += 1
    st.dataframe(display_fran, width='stretch')

    best = display_fran.iloc[0]
    worst = display_fran.iloc[-1]
    callout("success", "Reading this table",
            f"{best['Franchise']} leads on value efficiency ({best['Avg Value Index']:.1f}), "
            f"{worst['Franchise']} trails ({worst['Avg Value Index']:.1f}). Value index is a "
            f"performance-per-crore heuristic from this project's own scoring formula, not an "
            f"external or validated efficiency metric — read it as directional, not definitive.")
