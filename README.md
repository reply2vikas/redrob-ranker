# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Ranks the top 100 candidates from a 100,000-profile pool for the *Senior AI
Engineer (Founding Team)* job description, best-fit first, with per-candidate
reasoning.

## Reproduce the submission

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv   # should print "Submission is valid."
```

Works on `candidates.jsonl` or `candidates.jsonl.gz`. Runs CPU-only, no network,
in well under the 5-minute / 16 GB budget (~7 s on a 2k sample; the full 100k
pool runs comfortably within the limit on a laptop CPU).

## Approach (why it ranks the way it does)

The JD is explicit that the right answer is **not** "most AI keywords." Three
ideas drive the design:

1. **Score the evidence, not the labels.** `current_title` contradicts the
   actual career-history text in ~67% of profiles, so the semantic match runs
   over each candidate's **summary + career-history descriptions**, never over
   `current_title` or the raw skills list. A "Marketing Manager" whose
   descriptions are all marketing scores low; a Tier-5 candidate who *built a
   recommendation system at a product company* scores high even without buzzwords.

2. **Hybrid retrieval** (the JD literally asks about "hybrid vs dense"):
   `0.4 × TF-IDF cosine` (lexical) `+ 0.6 × LSA cosine` (TruncatedSVD latent
   semantics). Pure scikit-learn — **no model download, no network** — so it
   reproduces in the Stage-3 sandbox with zero external dependencies.
   *(Optional upgrade: sentence-transformer embeddings via
   `precompute_embeddings.py`; see that file's header.)*

3. **Rule layer + behavioral modifier** (`scoring.py`, all explainable):
   - experience band (soft around 5–9 yrs), product-vs-services
     (entire-career consulting = JD disqualifier), IR/NLP reward vs
     CV/speech/robotics penalty, keyword-stuffing penalty (high-proficiency
     skills with no endorsements/usage), honeypot detection (internal
     impossibilities), location fit (Pune/Noida/Tier-1 India/relocation).
   - a **behavioral availability multiplier** (~0.5–1.12) from recency, recruiter
     response rate, open-to-work, interview completion — because a perfect but
     inactive/unresponsive candidate is not actually hireable.

`final = semantic + Σ(rule adjustments)`, then `× behavioral multiplier`.
Sort by score desc, tie-break candidate_id ascending, emit top 100.

## Files

| File | Purpose |
| --- | --- |
| `rank.py` | End-to-end ranker → `submission.csv` (the reproduce command). |
| `scoring.py` | All scoring judgment: cleaning, traps, behavioral, reasoning. |
| `precompute_embeddings.py` | Optional sentence-transformer embedding upgrade. |
| `validate_submission.py` | Official format validator (run before submitting). |
| `submission_metadata.yaml` | Portal metadata (fill in your real details). |
| `requirements.txt` | Pinned dependencies. |

## Honeypots

Detected via internal contradictions (e.g. "expert" in a 0-month skill, a single
role longer than the whole career). On the sample, **0 honeypots reach the top
10** — well under the >10%-in-top-100 disqualifier.

## Notes for graders

- Sentinel `-1` (`github_activity_score`, `offer_acceptance_rate`) and empty
  `skill_assessment_scores: {}` are treated as **missing**, not zero.
- No LLM calls anywhere in the ranking path; reasoning strings are composed from
  real profile fields only (no hallucinated skills).
