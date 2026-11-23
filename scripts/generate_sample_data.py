"""Generate a synthetic CSV of public creator post metrics.

All creators, handles and captions are fictional. Run:
    python scripts/generate_sample_data.py
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "data" / "sample_creator_posts.csv"
END_DATE = datetime(2026, 11, 15)
DAYS = 90

# handle, platform, followers, niche, audience age bracket, top country,
# posts/week, base engagement rate, caption themes
CREATORS = [
    ("glowwithmaya", "instagram", 184_000, "skincare", "18-24", "US", 4.0, 0.048,
     ["spf", "skincare routine", "clean beauty", "dewy skin", "serum"]),
    ("dermdiaries", "tiktok", 612_000, "skincare", "25-34", "US", 6.5, 0.062,
     ["dermatologist tips", "acne", "retinol", "skin barrier", "spf"]),
    ("minimal.marcus", "instagram", 92_000, "lifestyle", "25-34", "UK", 2.5, 0.035,
     ["morning routine", "minimalism", "self care", "grooming", "coffee"]),
    ("fitfuelfiona", "tiktok", 340_000, "fitness", "18-24", "US", 5.0, 0.071,
     ["workout", "protein", "gym fit", "sweat proof makeup", "recovery"]),
    ("sunnyside.beauty", "youtube", 1_150_000, "beauty", "25-34", "CA", 1.5, 0.029,
     ["makeup tutorial", "spf", "drugstore haul", "skincare routine", "glow"]),
    ("techtomlin", "youtube", 760_000, "tech", "25-34", "US", 2.0, 0.031,
     ["gadget review", "unboxing", "smartphone", "desk setup", "giveaway"]),
    ("gourmet.gia", "instagram", 228_000, "food", "35-44", "US", 3.0, 0.041,
     ["recipe", "meal prep", "brunch", "wine pairing", "sourdough"]),
    ("budgetbeautybex", "tiktok", 88_000, "beauty", "18-24", "UK", 8.0, 0.089,
     ["drugstore dupes", "clean beauty", "spf", "get ready with me", "crypto giveaway"]),
]

HASHTAGS = {
    "skincare": ["#skincare", "#spf", "#skintok", "#cleanbeauty", "#glowingskin"],
    "beauty": ["#makeup", "#beauty", "#grwm", "#spf", "#drugstorebeauty"],
    "lifestyle": ["#selfcare", "#morningroutine", "#minimalism", "#wellness"],
    "fitness": ["#fitness", "#gymtok", "#workout", "#wellness", "#selfcare"],
    "tech": ["#tech", "#gadgets", "#unboxing", "#setup"],
    "food": ["#recipe", "#foodie", "#mealprep", "#brunch"],
}

FORMATS = {"instagram": ["reel", "carousel", "image"], "tiktok": ["video"], "youtube": ["video", "short"]}


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for (handle, platform, followers, niche, age, country, ppw, base_er, themes) in CREATORS:
        start = END_DATE - timedelta(days=DAYS)
        t = start + timedelta(hours=rng.uniform(0, 48))
        n = 0
        while t < END_DATE:
            n += 1
            er = max(0.004, rng.gauss(base_er, base_er * 0.35))
            # occasional viral spike
            if rng.random() < 0.06:
                er *= rng.uniform(2.0, 3.5)
            interactions = followers * er
            likes = int(interactions * rng.uniform(0.78, 0.86))
            comments = int(interactions * rng.uniform(0.03, 0.06))
            shares = int(interactions * rng.uniform(0.04, 0.09))
            saves = max(0, int(interactions) - likes - comments - shares)
            views = int(followers * rng.uniform(0.25, 1.6)) if platform != "instagram" or rng.random() < 0.7 else ""
            theme = rng.choice(themes)
            tags = rng.sample(HASHTAGS[niche], k=min(3, len(HASHTAGS[niche])))
            rows.append({
                "creator_handle": handle,
                "platform": platform,
                "post_id": f"{handle[:4]}-{n:03d}",
                "posted_at": t.strftime("%Y-%m-%d %H:%M"),
                "format": rng.choice(FORMATS[platform]),
                "followers": followers + int(rng.gauss(0, followers * 0.002)) + n * int(followers * 0.0005),
                "views": views,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
                "caption": f"New post: {theme} ✨",
                "hashtags": " ".join(tags),
                "content_category": niche,
                "audience_top_age": age,
                "audience_top_country": country,
            })
            # irregular gaps around the creator's average cadence
            gap_days = max(0.2, rng.expovariate(ppw / 7))
            t += timedelta(days=gap_days)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} posts for {df.creator_handle.nunique()} creators to {OUT}")


if __name__ == "__main__":
    main()
