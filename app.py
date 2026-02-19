# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ED Flow Monitor", layout="wide")

# ---- Light styling to avoid “default Streamlit look” ----
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
      .kpi-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 14px 16px;
        background: white;
        box-shadow: 0 1px 10px rgba(0,0,0,0.04);
      }
      .muted {color: rgba(0,0,0,0.55); font-size: 0.9rem;}
      .section-title {margin-top: 0.5rem; margin-bottom: 0.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Load data ----
@st.cache_data
def load_data(path: str = "ed_visits.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["arrival_datetime"])
    df["arrival_date"] = pd.to_datetime(df["arrival_datetime"]).dt.date
    return df

df = load_data()

# ---- Header ----
st.title("ED Flow Monitor: Wait Times, LOS, and LWBS")
st.write(
    "For ED operations leaders to monitor crowding and identify drivers of delays (wait time, length of stay, and LWBS)."
)

# ---- Sidebar filters ----
st.sidebar.header("Filters")

min_date = df["arrival_date"].min()
max_date = df["arrival_date"].max()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Normalize date_range output
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

triage_options = sorted(df["triage_level"].unique().tolist())
triage_sel = st.sidebar.multiselect("Triage level", triage_options, default=triage_options)

complaint_options = sorted(df["chief_complaint"].unique().tolist())
complaint_sel = st.sidebar.multiselect("Chief complaint", complaint_options, default=complaint_options)

mode_options = sorted(df["arrival_mode"].unique().tolist())
mode_sel = st.sidebar.multiselect("Arrival mode", mode_options, default=mode_options)

pod_options = sorted(df["pod"].unique().tolist())
pod_sel = st.sidebar.multiselect("Pod", pod_options, default=pod_options)

metric_choice = st.sidebar.selectbox(
    "Primary metric to emphasize",
    ["door_to_provider_min", "length_of_stay_min", "bed_occupancy_pct"],
    index=0,
)

# ---- Apply filters ----
mask = (
    (df["arrival_date"] >= start_date)
    & (df["arrival_date"] <= end_date)
    & (df["triage_level"].isin(triage_sel))
    & (df["chief_complaint"].isin(complaint_sel))
    & (df["arrival_mode"].isin(mode_sel))
    & (df["pod"].isin(pod_sel))
)
dff = df.loc[mask].copy()

# ---- KPI calculations ----
def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")

total_visits = len(dff)
avg_dtp = safe_mean(dff["door_to_provider_min"])
avg_los = safe_mean(dff["length_of_stay_min"])
avg_occ = safe_mean(dff["bed_occupancy_pct"])
lwbs_rate = (dff["disposition"].eq("Left Without Being Seen").mean() * 100) if total_visits else 0.0
admit_rate = (dff["disposition"].eq("Admitted").mean() * 100) if total_visits else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.markdown(f"<div class='kpi-card'><div class='muted'>Visits</div><div style='font-size:1.6rem;'>{total_visits:,}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><div class='muted'>Avg Door→Provider (min)</div><div style='font-size:1.6rem;'>{avg_dtp:,.1f}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-card'><div class='muted'>Avg LOS (min)</div><div style='font-size:1.6rem;'>{avg_los:,.1f}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-card'><div class='muted'>Avg Occupancy (%)</div><div style='font-size:1.6rem;'>{avg_occ:,.1f}</div></div>", unsafe_allow_html=True)
k5.markdown(f"<div class='kpi-card'><div class='muted'>LWBS Rate</div><div style='font-size:1.6rem;'>{lwbs_rate:,.1f}%</div></div>", unsafe_allow_html=True)
k6.markdown(f"<div class='kpi-card'><div class='muted'>Admission Rate</div><div style='font-size:1.6rem;'>{admit_rate:,.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("")

# ---- Plotly theme ----
px.defaults.template = "plotly_white"

# ---- Charts (4+ distinct types) ----
c1, c2 = st.columns([1.25, 1])

# 1) Line chart: trend by day (distinct question: trend)
trend = (
    dff.groupby("arrival_date", as_index=False)
       .agg(
           avg_door_to_provider=("door_to_provider_min", "mean"),
           avg_los=("length_of_stay_min", "mean"),
           avg_occ=("bed_occupancy_pct", "mean"),
           visits=("visit_id", "count"),
       )
       .sort_values("arrival_date")
)

y_map = {
    "door_to_provider_min": ("avg_door_to_provider", "Avg Door→Provider (min)"),
    "length_of_stay_min": ("avg_los", "Avg LOS (min)"),
    "bed_occupancy_pct": ("avg_occ", "Avg Occupancy (%)"),
}
y_col, y_label = y_map[metric_choice]

fig_line = px.line(
    trend,
    x="arrival_date",
    y=y_col,
    markers=True,
    title=f"Daily Trend: {y_label}",
)
fig_line.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
c1.plotly_chart(fig_line, use_container_width=True)

# 2) Bar chart: comparison by chief complaint (distinct question: which categories drive LOS/waits)
metric_for_bar = st.selectbox(
    "Bar metric (comparison)",
    ["door_to_provider_min", "length_of_stay_min", "bed_occupancy_pct"],
    index=1,
)
bar_df = (
    dff.groupby("chief_complaint", as_index=False)
       .agg(value=(metric_for_bar, "mean"), visits=("visit_id", "count"))
       .sort_values("value", ascending=False)
)

fig_bar = px.bar(
    bar_df,
    x="chief_complaint",
    y="value",
    hover_data=["visits"],
    title=f"Comparison by Chief Complaint: Avg {metric_for_bar}",
)
fig_bar.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
c2.plotly_chart(fig_bar, use_container_width=True)

c3, c4 = st.columns([1, 1])

# 3) Scatter: relationship occupancy vs wait (distinct question: is crowding linked to delays)
sample = dff.sample(min(len(dff), 2500), random_state=7) if len(dff) else dff
fig_scatter = px.scatter(
    sample,
    x="bed_occupancy_pct",
    y="door_to_provider_min",
    color="triage_level",
    title="Crowding Relationship: Occupancy vs Door→Provider",
    opacity=0.65,
    hover_data=["chief_complaint", "arrival_mode", "pod", "disposition"],
)
fig_scatter.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
c3.plotly_chart(fig_scatter, use_container_width=True)

# 4) Histogram: distribution (distinct question: outliers vs broad shift)
fig_hist = px.histogram(
    dff,
    x="length_of_stay_min",
    nbins=40,
    title="LOS Distribution (min)",
)
fig_hist.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
c4.plotly_chart(fig_hist, use_container_width=True)

# Optional 5th chart: heatmap of mean wait by day/hour (nice for staffing)
st.markdown("### Staffing Signal: When are waits highest?")
heat = (
    dff.assign(dow=pd.to_datetime(dff["arrival_datetime"]).dt.day_name())
       .groupby(["dow", "hour"], as_index=False)
       .agg(mean_wait=("door_to_provider_min", "mean"), visits=("visit_id", "count"))
)

# Order days
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat["dow"] = pd.Categorical(heat["dow"], categories=day_order, ordered=True)
heat = heat.sort_values(["dow", "hour"])

pivot = heat.pivot(index="dow", columns="hour", values="mean_wait")
fig_heat = px.imshow(
    pivot,
    aspect="auto",
    title="Mean Door→Provider (min) by Day of Week × Hour",
)
fig_heat.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
st.plotly_chart(fig_heat, use_container_width=True)

# ---- Insight callouts (helps presentation) ----
st.markdown("### Quick Insights (auto-generated)")
if len(dff) >= 50:
    # occupancy threshold insight
    high_occ = dff[dff["bed_occupancy_pct"] >= 85]
    low_occ = dff[dff["bed_occupancy_pct"] < 85]
    if len(high_occ) > 30 and len(low_occ) > 30:
        delta = high_occ["door_to_provider_min"].mean() - low_occ["door_to_provider_min"].mean()
        st.write(f"- When **occupancy ≥ 85%**, average door→provider is **{delta:,.1f} minutes higher** than when occupancy < 85%.")
    # flu wave insight if present
    if "flu_wave_flag" in dff.columns and dff["flu_wave_flag"].mean() > 0.05:
        fw = dff[dff["flu_wave_flag"] == 1]["door_to_provider_min"].mean()
        nf = dff[dff["flu_wave_flag"] == 0]["door_to_provider_min"].mean()
        st.write(f"- During the **flu-wave window**, average door→provider is **{(fw - nf):,.1f} minutes higher** than outside the window.")
else:
    st.write("- Not enough filtered data to compute robust insights. Try widening filters.")
