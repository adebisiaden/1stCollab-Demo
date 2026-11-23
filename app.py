"""Creator Brand Health — Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from brand_health.llm import available_providers, summarize
from brand_health.metrics import (
    FIT_WEIGHTS, creator_profile, load_brief, load_posts, roster_table, weekly_posts,
)
from brand_health.report import COMPONENT_LABELS, render_report

ROOT = Path(__file__).parent
SAMPLE_CSV = ROOT / "data" / "sample_creator_posts.csv"
BRIEF_JSON = ROOT / "data" / "brand_brief.json"
SERIES = "#2a78d6"
REF = "#8a8984"

st.set_page_config(page_title="Creator Brand Health", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Sidebar: data + brief + LLM
# --------------------------------------------------------------------------- #
@st.cache_data
def _load(source_bytes: bytes | None) -> pd.DataFrame:
    if source_bytes is None:
        return load_posts(SAMPLE_CSV)
    from io import BytesIO
    return load_posts(BytesIO(source_bytes))


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


with st.sidebar:
    st.header("1 · Data")
    upload = st.file_uploader("Creator post metrics CSV", type="csv",
                              help="Leave empty to use the bundled sample.")
    try:
        df = _load(upload.getvalue() if upload else None)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.caption(f"{len(df):,} posts · {df.creator_handle.nunique()} creators"
               + ("" if upload else " (sample data)"))

    st.header("2 · Brand brief")
    default = load_brief(BRIEF_JSON)
    with st.expander("Edit brief", expanded=False):
        brief = {
            "brand_name": st.text_input("Brand", default["brand_name"]),
            "campaign": st.text_input("Campaign", default["campaign"]),
            "product": st.text_input("Product", default["product"]),
            "target_categories": _csv_list(st.text_input(
                "Target categories", ", ".join(default["target_categories"]))),
            "keywords": _csv_list(st.text_area("Keywords", ", ".join(default["keywords"]))),
            "target_age_brackets": st.multiselect(
                "Target age brackets", ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"],
                default["target_age_brackets"]),
            "target_countries": _csv_list(st.text_input(
                "Target countries (ISO)", ", ".join(default["target_countries"]))),
            "preferred_platforms": st.multiselect(
                "Preferred platforms", ["instagram", "tiktok", "youtube"],
                default["preferred_platforms"]),
            "min_engagement_rate": st.slider(
                "Min engagement rate", 0.0, 0.10, default["min_engagement_rate"], 0.005,
                format="%.3f"),
            "desired_posts_per_week": st.slider(
                "Desired posts / week", 1, 14, default["desired_posts_per_week"]),
            "excluded_terms": _csv_list(st.text_input(
                "Excluded terms (brand safety)", ", ".join(default["excluded_terms"]))),
            "tone": st.text_input("Tone", default["tone"]),
        }
    st.caption(f"**{brief['brand_name']}** — {brief['campaign']}")

    st.header("3 · Summary model")
    providers = available_providers()
    provider = st.selectbox("LLM provider", providers,
                            help="Set ANTHROPIC_API_KEY or GEMINI_API_KEY to enable an LLM.")
    if providers == ["Rule-based (offline)"]:
        st.caption("No API key found — using the offline rule-based summary.")

brief_hash = hashlib.md5(json.dumps(brief, sort_keys=True).encode()).hexdigest()[:8]

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("Creator Brand Health")
st.caption("Engagement, posting cadence and audience fit against a brand brief, "
           "summarised into a one-page report.")

roster = roster_table(df, brief)
tab_roster, tab_creator, tab_report = st.tabs(["Roster", "Creator deep dive", "One-page report"])

with tab_roster:
    top = roster.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Creators analysed", len(roster))
    c2.metric("Strong fits", int((roster.fit_label == "Strong fit").sum()))
    c3.metric("Top fit", f"@{top.creator}", f"{top.fit_score:.0f}/100", delta_color="off")
    c4.metric("Creators with safety flags", int((roster.flagged_posts > 0).sum()))

    chart = (
        alt.Chart(roster)
        .mark_bar(color=SERIES, cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("fit_score:Q", title="Audience-fit score", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("creator:N", sort="-x", title=None),
            tooltip=["creator", "platform", "category",
                     alt.Tooltip("fit_score:Q", format=".1f"), "fit_label",
                     alt.Tooltip("median_er:Q", format=".1%", title="median ER"),
                     alt.Tooltip("posts_per_week:Q", format=".1f"), "flagged_posts"],
        )
        .properties(height=36 * len(roster))
    )
    rules = alt.Chart(pd.DataFrame({"x": [55, 75]})).mark_rule(
        color=REF, strokeDash=[3, 3]).encode(x="x:Q")
    st.altair_chart(chart + rules, width="stretch")
    st.caption("Dashed lines mark the moderate (55) and strong (75) fit thresholds.")

    st.dataframe(
        roster, hide_index=True, width="stretch",
        column_config={
            "followers": st.column_config.NumberColumn(format="%d"),
            "median_er": st.column_config.NumberColumn("median ER", format="percent"),
            "posts_per_week": st.column_config.NumberColumn("posts/wk", format="%.1f"),
            "consistency": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "fit_score": st.column_config.ProgressColumn("fit score", min_value=0, max_value=100,
                                                         format="%.0f"),
        },
    )

creator = st.sidebar.selectbox("Creator for deep dive / report", roster.creator.tolist())
posts = df[df.creator_handle == creator]
profile = creator_profile(posts, brief)
weekly = weekly_posts(posts)
eng, cad, fit = profile["engagement"], profile["cadence"], profile["fit"]

with tab_creator:
    st.subheader(f"@{creator}")
    st.caption(f"{profile['platform']} · {profile['content_category']} · audience "
               f"{profile['audience_top_age']}, {profile['audience_top_country']} · "
               f"{profile['date_range'][0]} to {profile['date_range'][1]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fit score", f"{fit['score']:.0f}/100", profile["fit_label"], delta_color="off")
    trend = eng["engagement_trend_pct"]
    c2.metric("Median engagement", f"{eng['median_engagement_rate']:.2%}",
              None if trend is None else f"{trend:+.0%} trend")
    c3.metric("Posts / week", f"{cad['posts_per_week']:.1f}",
              f"target {brief['desired_posts_per_week']}", delta_color="off")
    c4.metric("Consistency", f"{cad['consistency_score']:.2f}" if cad["consistency_score"] else "n/a",
              f"longest gap {cad['longest_gap_days']:.0f}d" if cad["longest_gap_days"] else None,
              delta_color="off")

    left, right = st.columns(2)
    with left:
        st.markdown("**Engagement rate per post**")
        line = alt.Chart(posts).mark_line(color=SERIES, strokeWidth=2).encode(
            x=alt.X("posted_at:T", title=None),
            y=alt.Y("engagement_rate:Q", title=None, axis=alt.Axis(format="%")),
        )
        pts = alt.Chart(posts).mark_circle(color=SERIES, size=60, opacity=0).encode(
            x="posted_at:T", y="engagement_rate:Q",
            tooltip=[alt.Tooltip("posted_at:T", title="posted"), "post_id", "caption",
                     alt.Tooltip("engagement_rate:Q", format=".2%", title="ER"),
                     "likes", "comments"],
        )
        ref = alt.Chart(pd.DataFrame({"y": [brief["min_engagement_rate"]]})).mark_rule(
            color=REF, strokeDash=[3, 3]).encode(y="y:Q")
        st.altair_chart(line + pts + ref, width="stretch")
        st.caption(f"Dashed line: brief minimum ({brief['min_engagement_rate']:.1%}).")
    with right:
        st.markdown("**Posts per week**")
        bars = alt.Chart(weekly).mark_bar(color=SERIES, cornerRadiusEnd=4).encode(
            x=alt.X("week:T", title=None),
            y=alt.Y("posts:Q", title=None),
            tooltip=[alt.Tooltip("week:T", title="week of"), "posts",
                     alt.Tooltip("median_engagement_rate:Q", format=".2%", title="median ER")],
        )
        tgt = alt.Chart(pd.DataFrame({"y": [brief["desired_posts_per_week"]]})).mark_rule(
            color=REF, strokeDash=[3, 3]).encode(y="y:Q")
        st.altair_chart(bars + tgt, width="stretch")
        st.caption(f"Dashed line: brief target ({brief['desired_posts_per_week']}/week).")

    st.markdown("**Audience-fit breakdown**")
    breakdown = pd.DataFrame([
        {"component": COMPONENT_LABELS[k], "match (0–1)": fit["components"][k],
         "points": fit["components"][k] * w * 100, "max points": w * 100}
        for k, w in FIT_WEIGHTS.items()
    ])
    st.dataframe(breakdown, hide_index=True, width="stretch", column_config={
        "match (0–1)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "points": st.column_config.NumberColumn(format="%.1f"),
        "max points": st.column_config.NumberColumn(format="%.0f"),
    })
    if fit["flagged_posts"]:
        st.warning(f"⚠️ Brand safety: {fit['flagged_posts']} posts mention excluded terms "
                   f"({', '.join(fit['flagged_terms'])}) — −{fit['safety_penalty']:.1f} pts.")
    with st.expander("Posts"):
        st.dataframe(posts.drop(columns=["engagement_rate_views"]), hide_index=True,
                     width="stretch")

with tab_report:
    key = f"summary::{creator}::{provider}::{brief_hash}"
    rank = int(roster.index[roster.creator == creator][0]) + 1
    roster_ctx = {"rank_by_fit": rank, "creators_in_roster": len(roster),
                  "roster_median_engagement_rate": float(roster.median_er.median())}

    col_a, col_b = st.columns([1, 3])
    with col_a:
        generate = st.button("Generate summary", type="primary", width="stretch")
    with col_b:
        st.caption(f"Summary for **@{creator}** using **{provider}**. "
                   "Only the computed metrics and the brief are sent to the model.")

    if generate or (key not in st.session_state and provider.startswith("Rule")):
        with st.spinner(f"Writing summary with {provider}…"):
            try:
                st.session_state[key] = summarize(profile, brief, provider, roster_ctx)
            except Exception as e:  # surface API errors without crashing the app
                st.error(f"{provider} failed: {e}. Falling back to the rule-based summary.")
                st.session_state[key] = summarize(profile, brief, "Rule-based (offline)")

    if key in st.session_state:
        summary_md, source = st.session_state[key]
        html_report = render_report(profile, brief, posts, weekly, summary_md, source, roster)
        st.download_button("⬇️ Download report (HTML, print to PDF)", html_report,
                           file_name=f"brand_health_{creator}.html", mime="text/html")
        # Safe to iframe: every CSV / brief / LLM value is html-escaped in render_report.
        st.iframe(html_report, height=1250)
    else:
        st.info("Click **Generate summary** to build the one-page report.")
