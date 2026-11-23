"""Render a one-page, print-ready HTML brand health report."""
from __future__ import annotations

import html
import re
from datetime import date

import pandas as pd

from .metrics import FIT_WEIGHTS

PLATFORM_NAMES = {"tiktok": "TikTok", "youtube": "YouTube", "instagram": "Instagram"}

COMPONENT_LABELS = {
    "category": "Category match", "keywords": "Keyword relevance", "age": "Audience age",
    "region": "Audience region", "platform": "Platform", "engagement": "Engagement vs. min",
    "cadence": "Cadence vs. target",
}


def _md_to_html(md: str) -> str:
    """Minimal Markdown (bold, italics, bullets, paragraphs) to HTML."""
    out, in_list = [], False
    for raw in md.splitlines():
        line = html.escape(raw.strip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"(?<!\*)\*(?!\s)(.+?)\*", r"<em>\1</em>", line)
        if re.match(r"^[-•*]\s+", raw.strip()):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{re.sub(r'^[-•*]\s+', '', line)}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line:
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _line_svg(posts: pd.DataFrame, min_er: float, w=340, h=120) -> str:
    """Engagement rate per post over time, with the brief minimum as a reference line."""
    pad_l, pad_r, pad_t, pad_b = 34, 8, 8, 18
    t = posts["posted_at"].astype("int64").to_numpy()
    y = posts["engagement_rate"].to_numpy()
    if len(t) < 2:
        return ""
    ymax = max(y.max(), min_er) * 1.1
    sx = lambda v: pad_l + (v - t.min()) / (t.max() - t.min()) * (w - pad_l - pad_r)
    sy = lambda v: pad_t + (1 - v / ymax) * (h - pad_t - pad_b)
    pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(t, y))
    ref = sy(min_er)
    ticks = "".join(
        f'<text x="{pad_l - 4}" y="{sy(v) + 3:.1f}" text-anchor="end" class="ax">{v:.0%}</text>'
        f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>'
        for v in (0, ymax / 2, ymax * 0.95)
    )
    d0, d1 = posts["posted_at"].min(), posts["posted_at"].max()
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="Engagement rate per post over time">
  {ticks}
  <line x1="{pad_l}" x2="{w - pad_r}" y1="{ref:.1f}" y2="{ref:.1f}" class="ref"/>
  <polyline points="{pts}" class="series"/>
  <text x="{pad_l}" y="{h - 4}" class="ax">{d0:%b %d}</text>
  <text x="{w - pad_r}" y="{h - 4}" text-anchor="end" class="ax">{d1:%b %d}</text>
</svg>"""


def _bars_svg(weekly: pd.DataFrame, target: float, w=340, h=120) -> str:
    """Posts per week bars with the brief's target cadence as a reference line."""
    pad_l, pad_r, pad_t, pad_b = 22, 8, 8, 18
    n = len(weekly)
    if n == 0:
        return ""
    ymax = max(weekly["posts"].max(), target) * 1.15
    slot = (w - pad_l - pad_r) / n
    bw = max(slot - 2, 1)  # 2px gap between adjacent bars
    sy = lambda v: pad_t + (1 - v / ymax) * (h - pad_t - pad_b)
    base = sy(0)
    bars = []
    for i, v in enumerate(weekly["posts"]):
        if v <= 0:
            continue
        x, top = pad_l + i * slot, sy(v)
        r = min(4, bw / 2, base - top)
        # rounded data-end, square baseline
        bars.append(
            f'<path d="M{x:.1f},{base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
            f'H{x + bw - r:.1f} Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} V{base:.1f} Z" class="bar">'
            f'<title>Week of {weekly["week"].iloc[i]:%b %d}: {int(v)} posts</title></path>')
    ref = sy(target)
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="Posts per week">
  <line x1="{pad_l}" x2="{w - pad_r}" y1="{base:.1f}" y2="{base:.1f}" class="grid"/>
  <text x="{pad_l - 4}" y="{base + 3:.1f}" text-anchor="end" class="ax">0</text>
  <text x="{pad_l - 4}" y="{sy(ymax / 1.15) + 3:.1f}" text-anchor="end" class="ax">{ymax / 1.15:.0f}</text>
  {''.join(bars)}
  <line x1="{pad_l}" x2="{w - pad_r}" y1="{ref:.1f}" y2="{ref:.1f}" class="ref"/>
  <text x="{pad_l}" y="{h - 4}" class="ax">{weekly['week'].iloc[0]:%b %d}</text>
  <text x="{w - pad_r}" y="{h - 4}" text-anchor="end" class="ax">{weekly['week'].iloc[-1]:%b %d}</text>
</svg>"""


def _fit_rows(fit: dict) -> str:
    rows = []
    for key, weight in FIT_WEIGHTS.items():
        v = fit["components"].get(key, 0)
        rows.append(
            f'<tr><td>{COMPONENT_LABELS[key]}</td>'
            f'<td class="meter"><span style="width:{v * 100:.0f}%"></span></td>'
            f'<td class="num">{v * weight * 100:.1f} / {weight * 100:.0f}</td></tr>')
    if fit["safety_penalty"]:
        rows.append(f'<tr class="penalty"><td>Brand-safety penalty</td><td>'
                    f'{fit["flagged_posts"]} flagged posts ({html.escape(", ".join(fit["flagged_terms"]))})'
                    f'</td><td class="num">−{fit["safety_penalty"]:.1f}</td></tr>')
    return "\n".join(rows)


def render_report(profile: dict, brief: dict, posts: pd.DataFrame, weekly: pd.DataFrame,
                  summary_md: str, summary_source: str, roster: pd.DataFrame | None = None) -> str:
    e, c, f = profile["engagement"], profile["cadence"], profile["fit"]
    esc = html.escape
    trend = e.get("engagement_trend_pct")
    trend_txt = "n/a" if trend is None else f"{trend:+.0%}"
    rank_txt = ""
    if roster is not None and len(roster) > 1:
        rank = int(roster.index[roster["creator"] == profile["creator_handle"]][0]) + 1
        rank_txt = f"Ranked #{rank} of {len(roster)} creators analysed"

    kpis = [
        ("Fit score", f"{f['score']:.0f}<small>/100</small>", profile["fit_label"]),
        ("Median engagement", f"{e['median_engagement_rate']:.1%}",
         f"Avg {e['avg_engagement_rate']:.1%} · trend {trend_txt}"),
        ("Posting cadence", f"{c['posts_per_week']:.1f}<small>/wk</small>",
         f"Median gap {c['median_gap_days']:.1f}d · longest {c['longest_gap_days']:.0f}d"
         if c["median_gap_days"] is not None else ""),
        ("Followers", f"{e['followers']:,}", f"{e['posts']} posts analysed"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="lbl">{l}</div><div class="val">{v}</div>'
        f'<div class="sub">{esc(s)}</div></div>' for l, v, s in kpis)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brand Health Report</title>
<style>
:root {{
  --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink-2:#52514e; --muted:#8a8984;
  --rule:#e4e3df; --series:#2a78d6; --ref:#8a8984; --warn-bg:#fff4e0; --warn-ink:#8a4b00;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --surface:#1a1a19; --card:#222220; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#8f8e86;
  --rule:#3a3936; --series:#3987e5; --ref:#8f8e86; --warn-bg:#3a2a12; --warn-ink:#f2c27a; }} }}
:root[data-theme="dark"] {{
  --surface:#1a1a19; --card:#222220; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#8f8e86;
  --rule:#3a3936; --series:#3987e5; --ref:#8f8e86; --warn-bg:#3a2a12; --warn-ink:#f2c27a; }}
@page {{ size: letter; margin: 0.45in; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font:13px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.page {{ max-width:8.1in; margin:0 auto; padding:24px 16px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px;
  border-bottom:2px solid var(--ink); padding-bottom:10px; margin-bottom:14px; flex-wrap:wrap; }}
h1 {{ font-size:22px; margin:0; letter-spacing:-0.01em; }}
h2 {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--ink-2); margin:0 0 6px; }}
.meta {{ color:var(--ink-2); font-size:12px; }}
.meta strong {{ color:var(--ink); }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px; }}
.kpi {{ background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:10px 12px; }}
.kpi .lbl {{ font-size:11px; color:var(--ink-2); }}
.kpi .val {{ font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; }}
.kpi .val small {{ font-size:13px; color:var(--ink-2); font-weight:500; }}
.kpi .sub {{ font-size:11px; color:var(--muted); }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }}
.card {{ background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:12px; }}
svg {{ width:100%; height:auto; display:block; }}
svg .series {{ fill:none; stroke:var(--series); stroke-width:2; stroke-linejoin:round; }}
svg .bar {{ fill:var(--series); }}
svg .grid {{ stroke:var(--rule); stroke-width:1; }}
svg .ref {{ stroke:var(--ref); stroke-width:1; stroke-dasharray:3 3; }}
svg .ax {{ fill:var(--muted); font-size:9px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
td, th {{ padding:4px 6px; border-bottom:1px solid var(--rule); text-align:left; }}
th {{ color:var(--ink-2); font-weight:600; font-size:11px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.meter {{ width:40%; }}
.meter span {{ display:block; height:8px; border-radius:0 4px 4px 0; background:var(--series); min-width:1px; }}
.penalty td {{ color:var(--warn-ink); background:var(--warn-bg); }}
.summary p {{ margin:0 0 6px; }} .summary ul {{ margin:0 0 8px; padding-left:18px; }}
.summary li {{ margin-bottom:2px; }}
.cap {{ font-size:10px; color:var(--muted); margin:-4px 0 4px; }}
.src {{ font-size:10px; color:var(--muted); margin-top:4px; }}
tr.me td {{ font-weight:650; }}
footer {{ font-size:10px; color:var(--muted); margin-top:10px; }}
@media (max-width:640px) {{ .kpis {{ grid-template-columns:1fr 1fr; }} .grid2 {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:#fff; }} .page {{ padding:0; }} }}
</style></head>
<body><div class="page">
<header>
  <div><h2>Brand health report</h2>
    <h1>@{esc(profile['creator_handle'])} × {esc(brief.get('brand_name', 'Brand'))}</h1>
    <div class="meta">{esc(brief.get('campaign', ''))}</div></div>
  <div class="meta" style="text-align:right">
    <strong>{esc(PLATFORM_NAMES.get(profile['platform'], profile['platform'].title()))}</strong> · {esc(profile['content_category'])} ·
    audience {esc(profile['audience_top_age'])}, {esc(profile['audience_top_country'])}<br>
    {profile['date_range'][0]} to {profile['date_range'][1]}<br>{rank_txt}</div>
</header>

<section class="kpis">{kpi_html}</section>

<section class="grid2">
  <div class="card summary"><h2>Analyst summary</h2>{_md_to_html(summary_md)}
    <div class="src">Generated by {esc(summary_source)} from the metrics on this page.</div></div>
  <div class="card"><h2>Audience-fit breakdown</h2>
    <table>{_fit_rows(f)}
      <tr><td><strong>Total</strong></td><td></td><td class="num"><strong>{f['score']:.1f} / 100</strong></td></tr>
    </table></div>
</section>

<section class="grid2">
  <div class="card"><h2>Engagement rate per post</h2><div class="cap">Dashed line: brief minimum ({brief.get('min_engagement_rate', 0.03):.0%})</div>{_line_svg(posts, brief.get('min_engagement_rate', 0.03))}</div>
  <div class="card"><h2>Posts per week</h2><div class="cap">Dashed line: brief target ({brief.get('desired_posts_per_week', 3):g}/week)</div>{_bars_svg(weekly, brief.get('desired_posts_per_week', 3))}</div>
</section>

{_roster_html(roster, profile['creator_handle'])}

<footer>Engagement rate = (likes + comments + shares + saves) ÷ followers. Fit score weights:
{', '.join(f'{COMPONENT_LABELS[k].lower()} {v:.0%}' for k, v in FIT_WEIGHTS.items())}; brand-safety
penalty up to 25 pts. Data: public post metrics (sample). Generated {date.today().isoformat()}.</footer>
</div></body></html>"""


def _roster_html(roster: pd.DataFrame | None, handle: str) -> str:
    if roster is None or len(roster) < 2:
        return ""
    top = roster.head(6)
    if handle not in set(top["creator"]):
        top = pd.concat([top, roster[roster["creator"] == handle]])
    rows = "".join(
        f'<tr class="{"me" if r.creator == handle else ""}"><td>@{html.escape(r.creator)}</td>'
        f'<td>{html.escape(r.platform)}</td><td class="num">{r.followers:,}</td>'
        f'<td class="num">{r.median_er:.1%}</td><td class="num">{r.posts_per_week:.1f}</td>'
        f'<td class="num">{r.fit_score:.0f}</td><td>{html.escape(r.fit_label)}</td></tr>'
        for r in top.itertuples())
    return f"""<section class="card"><h2>Roster comparison (by fit score)</h2>
<table><tr><th>Creator</th><th>Platform</th><th class="num">Followers</th><th class="num">Median ER</th>
<th class="num">Posts/wk</th><th class="num">Fit</th><th>Label</th></tr>{rows}</table></section>"""
