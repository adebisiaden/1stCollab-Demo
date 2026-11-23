"""Engagement, cadence and audience-fit metrics for creator post data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "creator_handle", "platform", "posted_at", "followers",
    "likes", "comments",
}
OPTIONAL_NUMERIC = ["views", "shares", "saves"]
OPTIONAL_TEXT = ["caption", "hashtags", "content_category",
                 "audience_top_age", "audience_top_country", "format", "post_id"]

# Weights for the audience-fit components (sum to 1.0).
FIT_WEIGHTS = {
    "category": 0.25,
    "keywords": 0.20,
    "age": 0.15,
    "region": 0.10,
    "platform": 0.10,
    "engagement": 0.10,
    "cadence": 0.10,
}
MAX_SAFETY_PENALTY = 25.0  # points removed if every post hits an excluded term


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_posts(source) -> pd.DataFrame:
    """Load and normalise a creator-posts CSV (path or file-like)."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    for col in OPTIONAL_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
    for col in OPTIONAL_TEXT:
        if col not in df.columns:
            df[col] = ""
    for col in ["followers", "likes", "comments", *OPTIONAL_NUMERIC]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in OPTIONAL_TEXT:
        df[col] = df[col].fillna("").astype(str)

    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
    df = df.dropna(subset=["posted_at", "followers"])
    df = df[df["followers"] > 0].copy()
    df["platform"] = df["platform"].str.lower().str.strip()
    return add_post_metrics(df)


def load_brief(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def add_post_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["interactions"] = df[["likes", "comments", "shares", "saves"]].fillna(0).sum(axis=1)
    # Engagement rate by followers is the industry-standard comparable metric.
    df["engagement_rate"] = df["interactions"] / df["followers"]
    views = df["views"].where(df["views"] > 0)
    df["engagement_rate_views"] = df["interactions"] / views
    return df.sort_values(["creator_handle", "posted_at"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Creator-level metrics
# --------------------------------------------------------------------------- #
def engagement_summary(posts: pd.DataFrame) -> dict:
    er = posts["engagement_rate"]
    half = len(posts) // 2
    trend = None
    if half >= 3:
        early, late = er.iloc[:half].median(), er.iloc[half:].median()
        trend = (late - early) / early if early else None
    return {
        "posts": int(len(posts)),
        "followers": int(posts["followers"].iloc[-1]),
        "avg_engagement_rate": float(er.mean()),
        "median_engagement_rate": float(er.median()),
        "avg_engagement_rate_views": _nan_to_none(posts["engagement_rate_views"].mean()),
        "avg_likes": float(posts["likes"].mean()),
        "avg_comments": float(posts["comments"].mean()),
        "comment_to_like_ratio": float(posts["comments"].sum() / max(posts["likes"].sum(), 1)),
        "engagement_trend_pct": _nan_to_none(trend),
        "top_post_engagement_rate": float(er.max()),
    }


def cadence_summary(posts: pd.DataFrame) -> dict:
    ts = posts["posted_at"].sort_values()
    if len(ts) < 2:
        return {"posts_per_week": float(len(ts)), "median_gap_days": None,
                "longest_gap_days": None, "consistency_score": None,
                "busiest_weekday": ts.dt.day_name().iloc[0] if len(ts) else None,
                "window_days": 0.0}
    gaps = ts.diff().dropna().dt.total_seconds() / 86400
    window_days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400, 7.0)
    cv = gaps.std() / gaps.mean() if gaps.mean() else 0
    return {
        "posts_per_week": float(len(ts) / window_days * 7),
        "median_gap_days": float(gaps.median()),
        "longest_gap_days": float(gaps.max()),
        # 1.0 = perfectly regular gaps; falls toward 0 as gaps get erratic.
        "consistency_score": float(1 / (1 + cv)),
        "busiest_weekday": ts.dt.day_name().mode().iloc[0],
        "window_days": float(window_days),
    }


def weekly_posts(posts: pd.DataFrame) -> pd.DataFrame:
    s = posts.set_index("posted_at").resample("W-MON")["engagement_rate"]
    out = pd.DataFrame({"posts": s.count(), "median_engagement_rate": s.median()})
    return out.reset_index().rename(columns={"posted_at": "week"})


# --------------------------------------------------------------------------- #
# Audience fit
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    score: float
    components: dict = field(default_factory=dict)   # name -> 0..1
    safety_penalty: float = 0.0
    flagged_posts: int = 0
    flagged_terms: list = field(default_factory=list)
    keyword_hits: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _contains_any(text: str, terms: list[str]) -> list[str]:
    text = text.lower()
    return [t for t in terms if re.search(rf"\b{re.escape(t.lower())}", text)]


def _mode(series: pd.Series) -> str:
    s = series[series.str.strip() != ""]
    return s.mode().iloc[0] if len(s) else ""


def audience_fit(posts: pd.DataFrame, brief: dict,
                 engagement: dict | None = None, cadence: dict | None = None) -> FitResult:
    engagement = engagement or engagement_summary(posts)
    cadence = cadence or cadence_summary(posts)
    text = (posts["caption"] + " " + posts["hashtags"].str.replace("#", " ", regex=False))

    comps, notes = {}, []

    category = _mode(posts["content_category"]).lower()
    targets = [c.lower() for c in brief.get("target_categories", [])]
    comps["category"] = 1.0 if category in targets else 0.0
    notes.append(f"Primary category '{category or 'unknown'}' "
                 f"{'matches' if comps['category'] else 'is outside'} brief categories.")

    keywords = brief.get("keywords", [])
    hits_per_post = text.apply(lambda t: _contains_any(t, keywords))
    kw_share = float((hits_per_post.str.len() > 0).mean()) if len(posts) else 0.0
    comps["keywords"] = kw_share
    keyword_hits = sorted({k for hits in hits_per_post for k in hits})

    age = _mode(posts["audience_top_age"])
    comps["age"] = 1.0 if age in brief.get("target_age_brackets", []) else 0.0
    country = _mode(posts["audience_top_country"]).upper()
    comps["region"] = 1.0 if country in [c.upper() for c in brief.get("target_countries", [])] else 0.0
    platform = _mode(posts["platform"])
    comps["platform"] = 1.0 if platform in brief.get("preferred_platforms", []) else 0.0

    min_er = brief.get("min_engagement_rate") or 0.03
    comps["engagement"] = min(engagement["median_engagement_rate"] / min_er, 1.0)

    desired = brief.get("desired_posts_per_week") or 3
    ppw = cadence["posts_per_week"]
    comps["cadence"] = max(0.0, 1 - abs(ppw - desired) / desired)

    base = 100 * sum(FIT_WEIGHTS[k] * comps[k] for k in FIT_WEIGHTS)

    excluded = brief.get("excluded_terms", [])
    flags = text.apply(lambda t: _contains_any(t, excluded))
    flagged = int((flags.str.len() > 0).sum())
    penalty = MAX_SAFETY_PENALTY * flagged / len(posts) if len(posts) else 0.0
    flagged_terms = sorted({t for f in flags for t in f})
    if flagged:
        notes.append(f"{flagged} post(s) mention excluded terms: {', '.join(flagged_terms)}.")

    return FitResult(
        score=round(max(0.0, base - penalty), 1),
        components={k: round(v, 3) for k, v in comps.items()},
        safety_penalty=round(penalty, 1),
        flagged_posts=flagged,
        flagged_terms=flagged_terms,
        keyword_hits=keyword_hits,
        notes=notes,
    )


def fit_label(score: float) -> str:
    if score >= 75:
        return "Strong fit"
    if score >= 55:
        return "Moderate fit"
    return "Weak fit"


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #
def creator_profile(posts: pd.DataFrame, brief: dict) -> dict:
    eng = engagement_summary(posts)
    cad = cadence_summary(posts)
    fit = audience_fit(posts, brief, eng, cad)
    return {
        "creator_handle": posts["creator_handle"].iloc[0],
        "platform": _mode(posts["platform"]),
        "content_category": _mode(posts["content_category"]),
        "audience_top_age": _mode(posts["audience_top_age"]),
        "audience_top_country": _mode(posts["audience_top_country"]),
        "date_range": (posts["posted_at"].min().date().isoformat(),
                       posts["posted_at"].max().date().isoformat()),
        "engagement": eng,
        "cadence": cad,
        "fit": fit.to_dict(),
        "fit_label": fit_label(fit.score),
    }


def roster_table(df: pd.DataFrame, brief: dict) -> pd.DataFrame:
    rows = []
    for handle, posts in df.groupby("creator_handle"):
        p = creator_profile(posts, brief)
        rows.append({
            "creator": handle,
            "platform": p["platform"],
            "category": p["content_category"],
            "followers": p["engagement"]["followers"],
            "posts": p["engagement"]["posts"],
            "median_er": p["engagement"]["median_engagement_rate"],
            "posts_per_week": p["cadence"]["posts_per_week"],
            "consistency": p["cadence"]["consistency_score"],
            "fit_score": p["fit"]["score"],
            "fit_label": p["fit_label"],
            "flagged_posts": p["fit"]["flagged_posts"],
        })
    return pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
