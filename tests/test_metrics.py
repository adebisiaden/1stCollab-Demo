from io import StringIO

import pandas as pd
import pytest

from brand_health.llm import rule_based_summary
from brand_health.metrics import (
    audience_fit, cadence_summary, creator_profile, engagement_summary, load_posts, roster_table,
)
from brand_health.report import render_report
from brand_health.metrics import weekly_posts

BRIEF = {
    "target_categories": ["skincare"], "keywords": ["spf"],
    "target_age_brackets": ["18-24"], "target_countries": ["US"],
    "preferred_platforms": ["tiktok"], "min_engagement_rate": 0.05,
    "desired_posts_per_week": 7, "excluded_terms": ["giveaway"],
}


def _csv(rows):
    header = ("creator_handle,platform,posted_at,followers,likes,comments,shares,saves,"
              "caption,hashtags,content_category,audience_top_age,audience_top_country\n")
    return load_posts(StringIO(header + "\n".join(rows)))


@pytest.fixture
def daily():
    rows = [f"a,tiktok,2026-01-{d:02d} 10:00,1000,40,5,3,2,spf tip,#spf,skincare,18-24,US"
            for d in range(1, 15)]
    return _csv(rows)


def test_engagement_rate(daily):
    assert daily["engagement_rate"].iloc[0] == pytest.approx(50 / 1000)
    assert engagement_summary(daily)["median_engagement_rate"] == pytest.approx(0.05)


def test_cadence_daily(daily):
    c = cadence_summary(daily)
    assert c["median_gap_days"] == pytest.approx(1.0)
    assert c["posts_per_week"] == pytest.approx(14 / 13 * 7)
    assert c["consistency_score"] == pytest.approx(1.0)


def test_perfect_fit(daily):
    fit = audience_fit(daily, BRIEF)
    assert fit.safety_penalty == 0
    assert fit.score > 98


def test_safety_penalty_and_mismatch():
    rows = [f"b,youtube,2026-01-{d:02d} 10:00,1000,5,1,0,0,huge giveaway,#tech,tech,35-44,UK"
            for d in range(1, 8)]
    fit = audience_fit(_csv(rows), BRIEF)
    assert fit.flagged_posts == 7
    assert fit.safety_penalty == pytest.approx(25.0)
    assert fit.score < 20


def test_missing_required_column():
    with pytest.raises(ValueError, match="missing required"):
        load_posts(StringIO("creator_handle,likes\na,1\n"))


def test_sample_end_to_end():
    df = load_posts("data/sample_creator_posts.csv")
    roster = roster_table(df, BRIEF)
    assert len(roster) == df.creator_handle.nunique()
    posts = df[df.creator_handle == roster.creator.iloc[0]]
    profile = creator_profile(posts, BRIEF)
    html = render_report(profile, BRIEF, posts, weekly_posts(posts),
                         rule_based_summary(profile, BRIEF), "test", roster)
    assert "<svg" in html and profile["creator_handle"] in html
