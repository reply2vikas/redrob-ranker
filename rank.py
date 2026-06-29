#!/usr/bin/env python3
"""
rank.py — produce the top-100 candidate ranking CSV for the Redrob JD.

Pipeline (all CPU, no network, well under the 5-min / 16 GB budget):

  1. Load candidates (.jsonl or .jsonl.gz) and the job description.
  2. Build each candidate's EVIDENCE text (career descriptions + summary).
  3. Hybrid semantic match against the JD:
        - lexical  : TF-IDF cosine        (exact term overlap)
        - semantic : TF-IDF -> TruncatedSVD (LSA) cosine  (latent meaning)
     Hybrid = 0.4*lexical + 0.6*semantic. This is the "hybrid vs dense"
     retrieval the JD explicitly cares about, and it needs NO model download,
     so it reproduces in the organizers' sandbox with zero network.
        (Optional upgrade: drop in sentence-transformer embeddings via
         precompute_embeddings.py and pass --embeddings; see README.)
  4. Add the rule layer from scoring.py (experience, product-vs-services,
     IR-vs-CV, keyword-stuffing, honeypots, location).
  5. Multiply by the behavioral availability modifier.
  6. Sort, tie-break by candidate_id ascending, write the top 100 with reasoning.

Reproduce:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

from __future__ import annotations
import argparse
import csv
import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import scoring


# --- the JD distilled to its operative signal (what it MEANS, not just says) ---
JD_QUERY = """
Senior AI Engineer for a Series A AI-native talent-intelligence product company.
Owns the intelligence layer: ranking, retrieval and matching systems that decide
what recruiters see. Needs production experience with embeddings-based retrieval
(sentence-transformers, BGE, E5, OpenAI embeddings), vector databases and hybrid
search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch),
strong Python, and rigorous evaluation of ranking systems (NDCG, MRR, MAP,
offline-to-online correlation, A/B testing). Has shipped at least one end-to-end
ranking, search or recommendation system to real users at product companies, not
pure research or pure services. Scrappy product-engineering attitude, ships fast,
writes well. Strong opinions on hybrid vs dense retrieval, when to fine-tune vs
prompt LLMs. Based in or willing to relocate to Pune or Noida.
"""


def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_now(records: list[dict]) -> date:
    """Use the latest activity date in the data as 'today' for recency math."""
    latest = date(2025, 1, 1)
    for r in records:
        d = r.get("redrob_signals", {}).get("last_active_date")
        if d:
            try:
                latest = max(latest, date.fromisoformat(d))
            except ValueError:
                pass
    return latest


def hybrid_semantic_scores(texts: list[str]) -> np.ndarray:
    """Return per-candidate hybrid similarity to the JD, scaled to [0, 1]."""
    corpus = texts + [JD_QUERY]
    tfidf = TfidfVectorizer(
        max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        stop_words="english",
    ).fit(corpus)
    X = tfidf.transform(texts)                       # candidates
    q = tfidf.transform([JD_QUERY])                  # JD

    # lexical cosine (rows already L2-normalized by TfidfVectorizer)
    lexical = (X @ q.T).toarray().ravel()

    # latent-semantic cosine via SVD on the shared space
    n_comp = min(256, X.shape[1] - 1, max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Xr = normalize(svd.fit_transform(X))
    qr = normalize(svd.transform(q))
    semantic = (Xr @ qr.T).ravel()

    def scale(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    return 0.4 * scale(lexical) + 0.6 * scale(semantic)


def score_all(records: list[dict], sem: np.ndarray, now: date):
    rows = []
    for rec, s in zip(records, sem):
        notes: list[str] = []
        adj = 0.0

        adj += scoring.experience_fit(rec)
        for fn in (scoring.product_vs_services, scoring.ir_vs_cv,
                   scoring.keyword_stuffing_penalty, scoring.honeypot_flags,
                   scoring.location_fit):
            a, note = fn(rec)
            adj += a
            if note:
                notes.append(note)

        tnote = scoring.title_mismatch_note(rec)
        if tnote:
            notes.append(tnote)

        base = scoring.WEIGHTS["semantic"] * float(s) + scoring.WEIGHTS["rules"] * adj
        mult, concern = scoring.behavioral_multiplier(rec, now)
        if concern:
            notes.append(concern)

        final = base * mult
        rows.append({
            "rec": rec,
            "candidate_id": rec["candidate_id"],
            "score": final,
            "semantic": float(s),
            "notes": notes,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    records = load_candidates(args.candidates)
    now = dataset_now(records)
    texts = [scoring.candidate_text(r) for r in records]

    sem = hybrid_semantic_scores(texts)
    rows = score_all(records, sem, now)

    # rank: score desc, tie-break candidate_id asc (matches validator rule)
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = rows[: args.top]

    # map raw -> a clean display score, non-increasing in raw, in ~[0.39, 0.99]
    raw = np.array([r["score"] for r in top], dtype=float)
    lo, hi = raw.min(), raw.max()
    disp = (raw - lo) / (hi - lo + 1e-9) * 0.6 + 0.39
    disp_rounded = [round(float(x), 4) for x in disp]
    for r, sc in zip(top, disp_rounded):
        r["disp"] = sc

    # CRITICAL: re-sort by (display score desc, candidate_id asc) so that the
    # validator's tie-break rule holds on the *emitted* (rounded) scores, then
    # enforce strict non-increasing as a final guard.
    top.sort(key=lambda r: (-r["disp"], r["candidate_id"]))

    median_sem = float(np.median(sem))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev = 1.0
        n = len(top)
        for i, r in enumerate(top, start=1):
            sc = min(r["disp"], prev)        # guarantee non-increasing
            prev = sc
            disqualified = any(
                "honeypot" in nt or "disqualifier" in nt for nt in r["notes"]
            )
            if disqualified:
                tier = "adjacent"
            elif i <= n * 0.30 and r["semantic"] >= median_sem:
                tier = "strong"
            elif i <= n * 0.70:
                tier = "moderate"
            else:
                tier = "adjacent"
            reasoning = scoring.build_reasoning(r["rec"], r["notes"], tier)
            w.writerow([r["candidate_id"], i, f"{sc:.4f}", reasoning])

    print(f"Wrote {len(top)} ranked candidates to {args.out}")


if __name__ == "__main__":
    main()
