"""Model health dashboard.

    uv run forecast dashboard          # or: uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from forecast.config import PROJECT_ROOT, load_config
from forecast.monitoring import views
from forecast.simulation.replay import DEFAULT_WORKDIR, load_meta

# Palette (dataviz reference instance): categorical slots in fixed order + chart ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK_2, MUTED = "#52514e", "#898781"
EVIDENTLY_THRESHOLD = 0.1  # Evidently's default for normed Wasserstein distance

st.set_page_config(page_title="Forecast Model Health", page_icon="📈", layout="wide")


# ---------- data ----------

WORKSPACES = {
    "Simulation (sim/)": DEFAULT_WORKDIR / "config.yaml",
    "Production (configs/config.yaml)": PROJECT_ROOT / "configs" / "config.yaml",
}
if os.environ.get("FORECAST_CONFIG"):  # point the dashboard at any workspace
    WORKSPACES = {"Custom (FORECAST_CONFIG)": Path(os.environ["FORECAST_CONFIG"]), **WORKSPACES}


@st.cache_data(ttl=60, show_spinner=False)
def load(config_path: str):
    cfg = load_config(config_path)
    runs = views.monitoring_runs(cfg)
    scored = views.scored_forecasts(cfg)
    try:
        from forecast.registry import list_versions
        versions = list_versions(cfg)
    except Exception:  # registry not created yet
        versions = pd.DataFrame()
    return cfg, runs, scored, versions, views.series_ids(cfg)


# ---------- chart helpers ----------

def hover_time_chart(long: pd.DataFrame, x: str, y: str, series: str, colors: dict[str, str],
                     y_title: str, dashes: dict[str, list[int]] | None = None,
                     fmt: str = ".3f", axis_fmt: str | None = None) -> alt.LayerChart:
    """Multi-line time chart with a nearest-x crosshair and a tooltip listing every series."""
    domain = list(colors)
    # One series needs no legend box: the chart title names it.
    legend = (alt.Legend(orient="top", title=None, symbolStrokeWidth=3) if len(domain) > 1
              else None)
    color = alt.Color(f"{series}:N", scale=alt.Scale(domain=domain, range=list(colors.values())),
                      legend=legend)
    base = alt.Chart(long).encode(x=alt.X(f"{x}:T", title=None),
                                  y=alt.Y(f"{y}:Q", title=y_title,
                                          axis=alt.Axis(format=axis_fmt) if axis_fmt
                                          else alt.Undefined),
                                  color=color)
    dash = dashes or {}
    lines = base.mark_line(strokeWidth=2).encode(
        strokeDash=alt.StrokeDash(f"{series}:N", legend=None, scale=alt.Scale(
            domain=domain, range=[dash.get(s, [1, 0]) for s in domain])))

    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False,
                                clear="pointerout")
    rule = (alt.Chart(long).transform_pivot(series, value=y, groupby=[x])
            .mark_rule(color=MUTED, strokeWidth=1)
            .encode(x=f"{x}:T", opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
                    tooltip=[alt.Tooltip(f"{x}:T", title="date")]
                    + [alt.Tooltip(f"{s}:Q", format=fmt) for s in domain])
            .add_params(hover))
    points = base.mark_point(filled=True, size=64).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return alt.layer(lines, rule, points)


def event_layers(changes: pd.DataFrame, shock: dict | None, end: pd.Timestamp,
                 label_y: float = 0) -> list:
    layers = []
    if shock:
        band = pd.DataFrame({"start": [pd.Timestamp(shock["date"])], "end": [end],
                             "label": [shock_label(shock)]})
        layers.append(alt.Chart(band).mark_rect(color=MUTED, opacity=0.10)
                      .encode(x="start:T", x2="end:T"))
        # Label sits at the bottom of the band (`label_y`, the zero line by default), clear
        # of the threshold lines up top.
        # (Anchored in data space: a pixel position at the bottom edge shrinks the plot.)
        layers.append(alt.Chart(band).mark_text(align="left", baseline="bottom", dx=4, dy=-4,
                                                color=INK_2, fontSize=11)
                      .encode(x="start:T", y=alt.datum(label_y), text="label:N"))
    if not changes.empty:
        ev = changes.assign(label="▲ v" + changes["model_version"].astype(str))
        layers.append(alt.Chart(ev).mark_rule(color=INK_2, strokeDash=[2, 3], strokeWidth=1)
                      .encode(x="as_of:T", tooltip=[
                          alt.Tooltip("as_of:T", title="new champion serving"),
                          alt.Tooltip("model_version:N", title="version"),
                          alt.Tooltip("model_data_end:T", title="trained up to")]))
        layers.append(alt.Chart(ev).mark_text(align="left", baseline="bottom", dx=3,
                                              color=INK_2, fontSize=11)
                      .encode(x="as_of:T", y=alt.value(0), text="label:N"))
    return layers


def shock_label(shock: dict) -> str:
    parts = []
    if shock.get("demand_factor", 1) != 1:
        parts.append(f"demand ×{shock['demand_factor']:g}")
    if shock.get("price_factor", 1) != 1:
        parts.append(f"price ×{shock['price_factor']:g}")
    scope = f" ({shock['dept']})" if shock.get("dept") else ""
    return "shock: " + ", ".join(parts) + scope if parts else "shock"


def decision_badge(decision: str) -> str:
    return "✅ OK" if decision == "ok" else "🔁 RETRAIN"


# ---------- page ----------

with st.sidebar:
    st.header("Workspace")
    available = {k: v for k, v in WORKSPACES.items() if v.exists()}
    if not available:
        st.error("No config found. Run `forecast simulate` or set up configs/config.yaml.")
        st.stop()
    choice = st.radio("Show data from", list(available), label_visibility="collapsed")
    config_path = available[choice]
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()

    meta = load_meta(config_path.parent)  # None unless this is a replay workspace
    if meta:
        st.subheader("Replay")
        st.markdown(f"- start **{meta['start']}**, {meta['days']} days\n"
                    f"- model **{meta['model']}**\n"
                    f"- weekly retrain: **{'on' if meta['weekly_retrain'] else 'off'}**")
        if meta.get("shock"):
            st.markdown(f"- {shock_label(meta['shock'])} from **{meta['shock']['date']}**")

cfg, runs, scored, versions, ids = load(str(config_path))
shock = meta.get("shock") if meta else None

st.title("Forecast model health")
m = cfg.monitoring
bias_rule = (f"live bias is past ±{m.max_abs_bias:.0%}, "
             if m.max_abs_bias is not None else "")
st.caption(f"Registry model `{cfg.registry.model_name}` · daily store × item unit sales · "
           f"retrain if live WAPE > {m.max_wape_ratio:g}× backtest, {bias_rule}"
           f"`{m.target_column}` drifts, ≥ {m.drift_share_threshold:.0%} of columns drift, "
           f"or the model is > {m.max_model_age_days} days old")

if runs.empty:
    st.info("No monitoring runs yet. Run `uv run forecast simulate` for a demo, or "
            "`uv run forecast pipeline monitor`.")
    st.stop()

latest = runs.iloc[-1]
changes = views.version_changes(runs)
end = runs["as_of"].max()

# KPI row
k = st.columns(6)
k[0].metric("Serving", f"v{latest['model_version']}", help=f"model type: {latest['model_type']}")
if pd.notna(latest["live_wape"]):
    k[1].metric("Live WAPE", f"{latest['live_wape']:.3f}",
                f"{latest['live_wape'] - latest['cv_wape']:+.3f} vs backtest",
                delta_color="inverse")
    k[2].metric("Live bias", f"{latest['live_bias']:+.1%}",
                help="+ = over-forecasting, − = under-forecasting, as a share of actual units")
else:
    k[1].metric("Live WAPE", "n/a", help="no forecasts cover the last 7 days")
    k[2].metric("Live bias", "n/a", help="no forecasts cover the last 7 days")
k[3].metric("Drifting", "n/a" if pd.isna(latest["drift_share"])
            else f"{latest['drift_share']:.0%}")
k[4].metric("Age", f"{int(latest['model_age_days'])}d")
k[5].metric("Decision", decision_badge(latest["decision"]),
            help=f"{len(changes)} model changes during this period")
if latest["reasons"]:
    st.warning("**Retrain triggered on " + str(latest["as_of"].date()) + ":** "
               + "; ".join(latest["reasons"]))

acc, bias_col = st.columns(2)

# 1. Live accuracy
with acc:
    st.subheader("Live accuracy vs. the retrain limit")
    wape = runs[["as_of", "live_wape", "cv_wape", "wape_limit"]].rename(columns={
        "live_wape": "live WAPE", "cv_wape": "backtest WAPE", "wape_limit": "retrain limit"})
    wape_long = wape.melt("as_of", var_name="series", value_name="wape").dropna()
    chart = hover_time_chart(
        wape_long, "as_of", "wape", "series",
        {"live WAPE": BLUE, "backtest WAPE": INK_2, "retrain limit": MUTED},
        "WAPE (lower is better)", dashes={"retrain limit": [5, 4]},
    )
    st.altair_chart(alt.layer(*event_layers(changes, shock, end), chart)
                    .properties(height=300), use_container_width=True)
    st.caption("Scores the forecasts actually served over the trailing 7 days. ▲ marks the "
               "day a new champion started serving; the shaded band is the injected shock.")

# 2. Live bias
with bias_col:
    st.subheader("Live bias vs. the retrain band")
    bias = (runs[["as_of", "live_bias"]].dropna()
            .assign(series="live bias").rename(columns={"live_bias": "bias"}))
    limit = m.max_abs_bias
    # Pad the y-range so the shock label (bottom) and the ▲ markers (top) have room clear
    # of the bias line and the limit lines.
    lo = min(bias["bias"].min(), -(limit or 0)) - 0.05 if not bias.empty else -0.05
    hi = max(bias["bias"].max(), limit or 0) + 0.05 if not bias.empty else 0.05
    layers = event_layers(changes, shock, end, label_y=lo)
    layers.append(alt.Chart(pd.DataFrame({"bias": [lo, hi]})).mark_point(opacity=0)
                  .encode(y="bias:Q"))
    if limit is not None:
        lim = pd.DataFrame({"bias": [-limit, limit], "as_of": [end, end],
                            "label": [f"−{limit:.0%} limit", f"+{limit:.0%} limit"]})
        layers.append(alt.Chart(lim).mark_rule(color=MUTED, strokeDash=[5, 4], strokeWidth=1.5)
                      .encode(y="bias:Q"))
        # Right-aligned just under each line: clear of the ▲ markers along the top edge.
        layers.append(alt.Chart(lim).mark_text(align="right", baseline="top", dx=-2, dy=3,
                                               color=INK_2, fontSize=11)
                      .encode(x="as_of:T", y="bias:Q", text="label:N"))
    layers.append(alt.Chart(pd.DataFrame({"bias": [0]}))
                  .mark_rule(color=INK_2, strokeWidth=1, opacity=0.5).encode(y="bias:Q"))
    chart = hover_time_chart(bias, "as_of", "bias", "series", {"live bias": BLUE},
                             "bias (+ over, − under)", fmt="+.1%", axis_fmt="+.0%")
    st.altair_chart(alt.layer(*layers, chart).properties(height=300),
                    use_container_width=True)
    st.caption("(forecast − actual) ÷ actual units over the same 7 days. A level shift shows "
               "up here within days, long before WAPE or drift; dashed lines are the "
               "retrain limit.")

left, right = st.columns(2)

# 3. Drift
with left:
    st.subheader("Data drift vs. training window")
    drift = views.drift_long(runs)
    if drift.empty:
        st.info("No drift checks yet (needs at least one day of new data).")
    else:
        thr = drift[["as_of"]].drop_duplicates().assign(column="drift threshold",
                                                        score=EVIDENTLY_THRESHOLD)
        drift = pd.concat([drift, thr], ignore_index=True)
        chart = hover_time_chart(
            drift, "as_of", "score", "column",
            {"sales": BLUE, "sell_price": ORANGE, "price_change_7": AQUA,
             "drift threshold": MUTED},
            "Wasserstein distance (normed)", dashes={"drift threshold": [5, 4]},
        )
        st.altair_chart(alt.layer(*event_layers(changes, shock, end), chart)
                        .properties(height=280), use_container_width=True)

# 4. Totals
with right:
    st.subheader("Total units per day: actual vs. forecast")
    totals = views.daily_totals(scored)
    if totals.empty:
        st.info("No forecasts have actuals to compare against yet.")
    else:
        tl = totals.melt("date", var_name="series", value_name="units")
        chart = hover_time_chart(tl, "date", "units", "series",
                                 {"actual": BLUE, "forecast": ORANGE}, "units / day", fmt=",.0f")
        st.altair_chart(alt.layer(*event_layers(changes, shock, end), chart)
                        .properties(height=280), use_container_width=True)

# 5. One series
st.subheader("Drill down: one series")
if ids:
    sid = st.selectbox("Series", ids, index=0)
    sv = views.series_view(cfg, sid, since=runs["as_of"].min() - pd.Timedelta(days=28))
    lines = sorted(sv["line"].unique(), key=lambda s: (s != "actual", s))
    forecast_lines = [s for s in lines if s != "actual"]
    # Actual in slot 1; forecasts in a single hue so versions don't compete with the actual.
    colors = {"actual": BLUE, **{s: ORANGE for s in forecast_lines}}
    chart = (alt.Chart(sv).mark_line(strokeWidth=2)
             .encode(x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="units / day"),
                     color=alt.Color("line:N", scale=alt.Scale(domain=list(colors),
                                     range=list(colors.values())),
                                     legend=alt.Legend(orient="top", title=None)),
                     detail="line:N",
                     opacity=alt.condition(alt.datum.line == "actual", alt.value(1),
                                           alt.value(0.75)),
                     tooltip=[alt.Tooltip("date:T"), alt.Tooltip("line:N", title="series"),
                              alt.Tooltip("value:Q", format=".2f")]))
    st.altair_chart(chart.properties(height=260), use_container_width=True)
    st.caption("Each orange segment is one model version's 28-day batch forecast.")

# 6. Tables
tab_runs, tab_versions, tab_report = st.tabs(["Monitoring log", "Registry versions",
                                              "Evidently report"])
with tab_runs:
    table = runs.assign(decision=runs["decision"].map(decision_badge),
                        reasons=runs["reasons"].map("; ".join),
                        drifted=runs["drifted_columns"].fillna("[]"))
    st.dataframe(
        table[["as_of", "model_version", "model_age_days", "n_points", "live_wape", "cv_wape",
               "wape_ratio", "live_bias", "drift_share", "drifted", "decision", "reasons"]]
        .sort_values("as_of", ascending=False),
        hide_index=True, use_container_width=True,
        column_config={
            "as_of": st.column_config.DateColumn("day"),
            "live_wape": st.column_config.NumberColumn("live WAPE", format="%.3f"),
            "cv_wape": st.column_config.NumberColumn("backtest WAPE", format="%.3f"),
            "wape_ratio": st.column_config.NumberColumn("ratio", format="%.2f×"),
            "live_bias": st.column_config.NumberColumn("live bias", format="percent"),
            "drift_share": st.column_config.NumberColumn("drift share", format="%.2f"),
        },
    )
with tab_versions:
    if versions.empty:
        st.info("No registered versions.")
    else:
        st.dataframe(versions.sort_values("version", ascending=False), hide_index=True,
                     use_container_width=True)
with tab_report:
    reports = runs.dropna(subset=["report_path"])
    reports = reports[reports["report_path"].map(lambda p: Path(p).exists())]
    if reports.empty:
        st.info("No Evidently reports yet.")
    else:
        day = st.selectbox("Day", reports["as_of"].dt.date.astype(str).tolist()[::-1])
        path = Path(reports.loc[reports["as_of"].dt.date.astype(str) == day,
                                "report_path"].iloc[0])
        components.html(path.read_text(encoding="utf-8"), height=900, scrolling=True)
