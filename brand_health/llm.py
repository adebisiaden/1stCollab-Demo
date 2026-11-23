"""LLM summary for the brand health report (Claude by default, Gemini optional).

Falls back to a deterministic rule-based summary when no API key is configured,
so the app always produces a report.
"""
from __future__ import annotations

import json
import os

CLAUDE_MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an influencer-marketing analyst writing for a brand partnerships manager.
You receive computed metrics for one creator and a brand brief. Write a concise brand health
summary for a one-page report.

Format (Markdown, under 220 words):
**Verdict:** one sentence with a clear recommendation (pursue / pursue with conditions / pass).
**Strengths:** 2-3 bullets.
**Risks:** 2-3 bullets (brand safety, audience mismatch, cadence, engagement trend).
**Next step:** one sentence.

Ground every claim in the numbers provided; quote them (e.g. "4.1% median engagement").
Engagement rates are fractions in the data: present them as percentages.
Do not invent metrics, demographics or history that are not in the input."""


def build_prompt(profile: dict, brief: dict, roster_context: dict | None = None) -> str:
    payload = {"brand_brief": brief, "creator_metrics": profile}
    if roster_context:
        payload["roster_context"] = roster_context
    return ("Here is the data as JSON. Write the brand health summary.\n\n"
            + json.dumps(payload, indent=2, default=str))


def available_providers() -> list[str]:
    providers = []
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        providers.append("Claude")
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        providers.append("Gemini")
    providers.append("Rule-based (offline)")
    return providers


def summarize(profile: dict, brief: dict, provider: str = "Claude",
              roster_context: dict | None = None) -> tuple[str, str]:
    """Return (markdown_summary, source_label)."""
    prompt = build_prompt(profile, brief, roster_context)
    if provider == "Claude":
        return _claude(prompt), f"Claude ({CLAUDE_MODEL})"
    if provider == "Gemini":
        return _gemini(prompt), f"Gemini ({GEMINI_MODEL})"
    return rule_based_summary(profile, brief), "Rule-based (no LLM)"


def _claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            # Server-side fallback: if a safety classifier declines, the API
            # re-runs the request on a fallback model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError("Claude API key was rejected. Check ANTHROPIC_API_KEY.") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError("Claude rate limit hit. Try again in a moment.") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("Could not reach the Claude API.") from e

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to summarise this input.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={response.stop_reason}).")
    return text


def _gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    if not response.text:
        raise RuntimeError("Gemini returned no text.")
    return response.text.strip()


def rule_based_summary(profile: dict, brief: dict) -> str:
    """Deterministic summary used when no LLM is configured."""
    eng, cad, fit = profile["engagement"], profile["cadence"], profile["fit"]
    comps = fit["components"]
    er = eng["median_engagement_rate"]
    min_er = brief.get("min_engagement_rate", 0.03)

    if fit["score"] >= 75 and fit["flagged_posts"] == 0:
        verdict = "Pursue: strong audience fit with no brand-safety flags."
    elif fit["score"] >= 55:
        verdict = "Pursue with conditions: solid fit, but address the risks below first."
    else:
        verdict = "Pass for this campaign: audience fit is too weak."

    strengths, risks = [], []
    if er >= min_er:
        strengths.append(f"{er:.1%} median engagement beats the {min_er:.1%} brief minimum.")
    else:
        risks.append(f"{er:.1%} median engagement is below the {min_er:.1%} brief minimum.")
    if comps.get("category"):
        strengths.append(f"Content category ({profile['content_category']}) is on-brief.")
    else:
        risks.append(f"Content category ({profile['content_category']}) is off-brief.")
    if comps.get("keywords", 0) >= 0.5:
        strengths.append(f"{comps['keywords']:.0%} of posts use brief keywords "
                         f"({', '.join(fit['keyword_hits'][:4])}).")
    if not comps.get("region"):
        risks.append(f"Top audience country ({profile['audience_top_country']}) is outside the target markets.")
    if not comps.get("age"):
        risks.append(f"Top audience age ({profile['audience_top_age']}) is outside the target brackets.")
    if fit["flagged_posts"]:
        risks.append(f"{fit['flagged_posts']} posts mention excluded terms "
                     f"({', '.join(fit['flagged_terms'])}).")
    if comps.get("cadence", 1) < 0.5:
        risks.append(f"Posts {cad['posts_per_week']:.1f}x/week vs. the brief's "
                     f"{brief.get('desired_posts_per_week')}x/week target.")
    trend = eng.get("engagement_trend_pct")
    if trend is not None and trend < -0.15:
        risks.append(f"Engagement is down {abs(trend):.0%} over the period.")

    lines = [f"**Verdict:** {verdict}", "", "**Strengths:**"]
    lines += [f"- {s}" for s in (strengths[:3] or ["None stand out against this brief."])]
    lines += ["", "**Risks:**"]
    lines += [f"- {r}" for r in (risks[:3] or ["No major risks detected."])]
    nxt = ("Request audience demographics screenshots and a content-safety review before outreach."
           if fit["flagged_posts"] or not comps.get("region")
           else "Reach out with the campaign brief and request a rate card.")
    lines += ["", f"**Next step:** {nxt}"]
    return "\n".join(lines)
