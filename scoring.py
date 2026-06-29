"""
scoring.py — Intelligent Candidate Discovery & Ranking
======================================================

All the *judgment* in the ranker lives here, separated from orchestration so it
can be read, reasoned about, and defended on its own.

Design philosophy (straight from the JD):
  - The structured labels lie. `current_title` does not match what people
    actually did in ~67% of profiles. So we score on what a candidate DID
    (career-history descriptions + summary), not on titles or skill labels.
  - Skill lists are gamed. We never let a raw skill list inflate a score; skill
    claims are trust-discounted by endorsements + duration.
  - Behavioral signals decide *availability*. A perfect-on-paper candidate who
    is inactive / unresponsive is not actually hireable -> down-weight.
  - The dataset has traps (keyword stuffers, honeypots, CV/speech-only). We
    penalize the detectable ones so they don't pollute the top 100.

Every number below is a deliberate, explainable choice. Tune the WEIGHTS dict
to change behavior; nothing is hidden.
"""

from __future__ import annotations
import re
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# Sentinel handling: -1 and {} mean "missing", NOT "zero/bad".
# ---------------------------------------------------------------------------

def clean_signal(value: Any, missing_sentinels=(-1, -1.0)) -> float | None:
    """Return None when a numeric signal is a missing-sentinel, else the value."""
    if value is None:
        return None
    if value in missing_sentinels:
        return None
    return value


# ---------------------------------------------------------------------------
# Text construction for semantic matching.
# We embed the EVIDENCE of what someone did, not the labels they claim.
# ---------------------------------------------------------------------------

def candidate_text(rec: dict) -> str:
    """Build the text blob used for semantic similarity.

    Includes: headline + summary + every career-history description (the real
    signal). Career *titles* are included because they appear inside the history
    narrative, but the top-level current_title is intentionally NOT trusted as a
    standalone label. Skill *names* are deliberately excluded so keyword-stuffers
    gain nothing from a padded skills list.
    """
    p = rec.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for job in rec.get("career_history", []):
        # title gives mild context; description carries the weight
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(t for t in parts if t).strip()


# ---------------------------------------------------------------------------
# Domain vocabularies used by the rule layer.
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mphasis", "dxc", "ltimindtree",
    "larsen", "mindtree",  # services-heavy; counted only if ENTIRE career is here
}

# JD: production retrieval / ranking / IR / NLP is what matters.
IR_NLP_TERMS = [
    "retrieval", "ranking", "recommendation", "recommender", "search",
    "embedding", "embeddings", "vector", "semantic", "information retrieval",
    "nlp", "natural language", "bm25", "elasticsearch", "opensearch", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "llm", "language model",
    "fine-tun", "rag", "ndcg", "learning to rank", "relevance",
]
# JD explicitly de-prioritizes these when NOT paired with NLP/IR.
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection", "opencv",
    "speech recognition", "tts", "asr", "robotics", "gans", "segmentation",
]
# Tier-1 Indian cities / target locations from the JD.
TARGET_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "gurgaon",
    "gurugram", "bengaluru", "bangalore",
]
NON_TECH_TITLE_TERMS = [
    "marketing", "sales", "operations manager", "customer support",
    "support manager", "account manager", "hr ", "human resources", "recruiter",
    "brand", "content writer",
]


def _text_lower(rec: dict) -> str:
    return candidate_text(rec).lower()


# ---------------------------------------------------------------------------
# Rule-layer components. Each returns an additive adjustment in score-space
# (roughly [-0.3, +0.15]) plus optional notes for the reasoning string.
# ---------------------------------------------------------------------------

def experience_fit(rec: dict) -> float:
    """Soft band around the JD's 5-9 years. Flexible, not a hard gate."""
    yoe = rec.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        return 0.10
    if 4 <= yoe < 5 or 9 < yoe <= 11:
        return 0.04
    if yoe < 2:
        return -0.10           # too junior for a senior founding role
    if yoe > 15:
        return -0.06           # likely over-senior / "moved into architecture"
    return 0.0


def product_vs_services(rec: dict) -> tuple[float, str | None]:
    """JD rejects ENTIRE-career consulting; rewards product-company signal."""
    companies = [
        (j.get("company") or "").lower() for j in rec.get("career_history", [])
    ]
    if not companies:
        return 0.0, None
    is_consulting = [
        any(f in c for f in CONSULTING_FIRMS) for c in companies if c
    ]
    if is_consulting and all(is_consulting):
        return -0.20, "career entirely at services/consulting firms (JD disqualifier)"
    return 0.0, None


def ir_vs_cv(rec: dict) -> tuple[float, str | None]:
    """Reward IR/NLP evidence; gently penalize CV/speech/robotics WITHOUT it."""
    t = _text_lower(rec)
    has_ir = any(term in t for term in IR_NLP_TERMS)
    has_cv = any(term in t for term in CV_SPEECH_ROBOTICS_TERMS)
    if has_cv and not has_ir:
        return -0.18, "primarily CV/speech/robotics, little NLP/IR exposure"
    if has_ir:
        return 0.08, None
    return 0.0, None


def keyword_stuffing_penalty(rec: dict) -> tuple[float, str | None]:
    """Classic stuffer: many high-proficiency skills with ~no evidence.

    Evidence = endorsements + months of use. A wall of 'advanced/expert' skills
    with zero endorsements and zero duration is a stuffing signal.
    """
    skills = rec.get("skills", [])
    if not skills:
        return 0.0, None
    high = [s for s in skills if s.get("proficiency") in ("advanced", "expert")]
    if len(high) < 5:
        return 0.0, None
    no_evidence = [
        s for s in high
        if (s.get("endorsements", 0) or 0) == 0
        and (s.get("duration_months", 0) or 0) == 0
    ]
    ratio = len(no_evidence) / len(high)
    if ratio >= 0.6:
        return -0.15, "many high-proficiency skills with no endorsements/usage (stuffing)"
    return 0.0, None


def honeypot_flags(rec: dict) -> tuple[float, str | None]:
    """Detect subtly-impossible profiles (forced to relevance 0 in ground truth).

    We don't have company founding dates, so we use internal contradictions:
      - 'expert' in a skill claimed for 0 months
      - a single role longer than the person's entire stated experience
      - total tenure wildly exceeding years_of_experience
    Ranking honeypots high triggers a >10% disqualifier, so we penalize hard.
    """
    flags = 0
    skills = rec.get("skills", [])
    if any(
        s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0
        for s in skills
    ):
        flags += 1

    yoe_months = (rec.get("profile", {}).get("years_of_experience", 0) or 0) * 12
    hist = rec.get("career_history", [])
    durations = [j.get("duration_months", 0) or 0 for j in hist]
    if yoe_months > 0:
        if any(d > yoe_months + 12 for d in durations):
            flags += 1                       # one job longer than whole career
        if sum(durations) > yoe_months * 2.0 + 24:
            flags += 1                       # impossible overlapping tenure

    if flags >= 2:
        return -0.60, "profile has internal impossibilities (likely honeypot)"
    if flags == 1:
        return -0.25, "minor profile inconsistency"
    return 0.0, None


def location_fit(rec: dict) -> tuple[float, str | None]:
    """JD: Pune/Noida-preferred, Tier-1 India welcome, relocation OK, no visas."""
    p = rec.get("profile", {})
    sig = rec.get("redrob_signals", {})
    loc = (p.get("location") or "").lower()
    country = (p.get("country") or "").lower()
    relocate = bool(sig.get("willing_to_relocate"))

    if any(city in loc for city in TARGET_LOCATIONS):
        return 0.08, None
    if country == "india":
        return 0.03 if relocate else 0.0, None
    # outside India
    if relocate:
        return -0.04, "outside India but open to relocation (no visa sponsorship per JD)"
    return -0.12, "outside India, not open to relocation (JD: no visa sponsorship)"


def title_mismatch_note(rec: dict) -> str | None:
    """Soft note only (titles are unreliable). Used for reasoning transparency."""
    title = (rec.get("profile", {}).get("current_title") or "").lower()
    if any(term in title for term in NON_TECH_TITLE_TERMS):
        return f"current title '{rec['profile'].get('current_title')}' is non-technical"
    return None


# ---------------------------------------------------------------------------
# Behavioral multiplier — availability & engagement (a MODIFIER, per JD).
# ---------------------------------------------------------------------------

def behavioral_multiplier(rec: dict, now: date) -> tuple[float, str | None]:
    """Return a multiplier ~[0.5, 1.12] and an optional concern note.

    Captures: recency of activity, recruiter responsiveness, open-to-work,
    interview reliability. An inactive, unresponsive candidate is down-weighted
    because they cannot realistically be hired.
    """
    sig = rec.get("redrob_signals", {})
    m = 1.0
    concern = None

    # recency of last activity
    last = sig.get("last_active_date")
    if last:
        try:
            d = date.fromisoformat(last)
            months_idle = (now - d).days / 30.0
            if months_idle <= 1:
                m *= 1.06
            elif months_idle <= 3:
                m *= 1.0
            elif months_idle <= 6:
                m *= 0.85
            else:
                m *= 0.65
                concern = f"inactive ~{months_idle:.0f} months"
        except ValueError:
            pass

    # recruiter response rate
    rrr = sig.get("recruiter_response_rate")
    if rrr is not None:
        if rrr >= 0.5:
            m *= 1.05
        elif rrr < 0.15:
            m *= 0.8
            concern = concern or f"low recruiter response rate ({rrr:.0%})"

    # open to work
    if sig.get("open_to_work_flag") is True:
        m *= 1.05
    elif sig.get("open_to_work_flag") is False:
        m *= 0.9

    # interview reliability
    icr = sig.get("interview_completion_rate")
    if icr is not None and icr < 0.4:
        m *= 0.9

    return max(0.5, min(1.12, m)), concern


# ---------------------------------------------------------------------------
# Reasoning string — fact-grounded, varied, honest about concerns.
# Stage 4 penalizes templated / hallucinated / rank-inconsistent reasoning.
# ---------------------------------------------------------------------------

def _best_role_snippet(rec: dict) -> str:
    """Shortest informative phrase describing what they actually did."""
    hist = rec.get("career_history", [])
    cur = next((j for j in hist if j.get("is_current")), hist[0] if hist else {})
    desc = (cur.get("description") or "").strip()
    if not desc:
        return ""
    first = re.split(r"(?<=[.;])\s", desc)[0]
    return first[:160].rstrip(".") if first else ""


def build_reasoning(rec: dict, notes: list[str], tier: str) -> str:
    """Compose 1-2 sentences from REAL profile facts + collected concern notes.

    `tier` is "strong" | "moderate" | "adjacent", derived from rank position, so
    the tone always matches the rank (Stage 4 penalizes glowing reasoning on a
    low-ranked candidate and vice-versa).
    """
    p = rec.get("profile", {})
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "unknown location")
    snippet = _best_role_snippet(rec)

    lead = f"{yoe:.1f} yrs experience" if isinstance(yoe, (int, float)) else "experience"
    body = f"; recent work: {snippet}" if snippet else ""
    where = f"; based in {loc}"

    concerns = [n for n in notes if n]
    if concerns:
        # always surface real concerns, regardless of tier
        tail = ". Concerns: " + "; ".join(concerns[:2]) + "."
    elif tier == "strong":
        tail = ". Close fit to the JD's product-ML / retrieval-and-ranking profile."
    elif tier == "moderate":
        tail = ". Solid partial fit on relevant ML/engineering signal."
    else:
        tail = ". Adjacent only; included as lower-confidence filler."

    text = (lead + body + where + tail).strip()
    return re.sub(r"\s+", " ", text)[:300]


# ---------------------------------------------------------------------------
# Component weights (semantic similarity is combined in rank.py).
# Exposed here so the whole scoring policy is in one readable place.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "semantic": 1.00,   # cosine(JD, candidate evidence text) — primary driver
    "rules": 1.00,      # sum of additive rule adjustments below
}
