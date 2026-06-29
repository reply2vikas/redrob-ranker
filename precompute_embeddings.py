#!/usr/bin/env python3
"""
precompute_embeddings.py — OPTIONAL quality upgrade.

The default ranker (rank.py) is fully self-contained: TF-IDF + LSA, no model
download, no network — so it reproduces cleanly in the organizers' sandbox.

If you want stronger *semantic* matching, precompute sentence-transformer
embeddings ONCE (this step may use a GPU and exceed 5 min — that's allowed; only
the ranking step is constrained), then point rank.py at the saved file.

    pip install sentence-transformers
    python precompute_embeddings.py --candidates candidates.jsonl --out cand_emb.npy

This writes:
    cand_emb.npy        float32 [N, 384] L2-normalized candidate embeddings
    cand_ids.json       the candidate_id order matching the rows
    jd_emb.npy          the JD embedding [384]

To use them, extend rank.py to load cand_emb.npy and replace the `semantic`
vector with cosine(jd_emb, cand_emb). Document this clearly so Stage 3 can
reproduce the ranking step from the saved artifact with no network.
"""
from __future__ import annotations
import argparse, gzip, json
import numpy as np

import scoring
from rank import JD_QUERY, load_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="cand_emb.npy")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    records = load_candidates(args.candidates)
    texts = [scoring.candidate_text(r) for r in records]
    ids = [r["candidate_id"] for r in records]

    emb = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")
    jd = model.encode([JD_QUERY], normalize_embeddings=True).astype("float32")[0]

    np.save(args.out, emb)
    np.save("jd_emb.npy", jd)
    json.dump(ids, open("cand_ids.json", "w"))
    print(f"Saved {emb.shape} -> {args.out}, jd_emb.npy, cand_ids.json")


if __name__ == "__main__":
    main()
