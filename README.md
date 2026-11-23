# Creator Brand Health

Score creators against a brand brief using their public post metrics, then produce a
one-page brand health report with an LLM-written summary.

**Stack:** Python, pandas, Streamlit, Altair, Claude (`claude-opus-5-5`) or Gemini (`gemini-2.5-flash`).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # or GEMINI_API_KEY=...  (optional)
streamlit run app.py
```

Without an API key the app still works: it uses a deterministic, rule-based summary.

## What it computes

| Metric | Definition |
|---|---|
| **Engagement rate** | (likes + comments + shares + saves) ÷ followers, per post. Reported as median (robust to viral spikes), mean, by views, and the trend (second half vs. first half of the period). |
| **Posting cadence** | Posts/week over the observed window, median and longest gap between posts, and a **consistency score** = 1 / (1 + coefficient of variation of gaps). 1.0 = perfectly regular. |
| **Audience-fit score (0–100)** | Weighted match against the brief: category 25, keyword relevance 20 (share of posts using brief keywords), audience age 15, region 10, platform 10, engagement vs. brief minimum 10, cadence vs. target 10. A **brand-safety penalty** of up to −25 scales with the share of posts mentioning excluded terms. ≥75 strong, ≥55 moderate, else weak. |

Weights live in `FIT_WEIGHTS` in [brand_health/metrics.py](brand_health/metrics.py).

## The report

The **One-page report** tab sends only the computed metrics and the brief (no raw posts) to
the selected LLM, which returns a Verdict / Strengths / Risks / Next step summary grounded in
the numbers. The page includes KPI tiles, the fit breakdown, engagement and cadence charts,
and a roster comparison. Download it as HTML and print to PDF; it's sized for one letter page.

Claude calls use `claude-opus-5-5` at low effort with server-side refusal fallback enabled
(`fallbacks="default"`). If an API call fails, the app shows the error and falls back to the
rule-based summary.

## Project layout

```
app.py                         Streamlit UI (Roster · Creator deep dive · One-page report)
brand_health/metrics.py        Loading, engagement, cadence, audience-fit, roster
brand_health/llm.py            Claude / Gemini / rule-based summaries
brand_health/report.py         One-page HTML report (inline SVG charts, light/dark, print CSS)
data/sample_creator_posts.csv  421 synthetic posts from 8 fictional creators (90 days)
data/brand_brief.json          Mock brand brief (fictional SPF brand)
scripts/generate_sample_data.py Regenerates the sample CSV (seeded)
tests/test_metrics.py          Unit and end-to-end tests:  pytest -q
```

## Using your own data

Upload a CSV in the sidebar. Required columns: `creator_handle, platform, posted_at,
followers, likes, comments`. Optional: `views, shares, saves, caption, hashtags,
content_category, audience_top_age, audience_top_country, format, post_id`. Missing optional
columns score as "no match" in the fit components that need them.

Audience age and country are usually not public. In the sample they stand in for
creator-provided media-kit data or third-party estimates.
