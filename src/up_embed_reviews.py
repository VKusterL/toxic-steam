#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_embed_reviews.py  -  User Predictor: per-review embedding cache (SBERT)

Produces the expensive, reusable artifact: the SBERT embedding of every EN review from the
users in the 10% subsample (hash(user_key)%mod==0). Per-user aggregation (mean -> 1 vector/user) happens LATER (up_user_vectors.py), starting from
this cache, so we avoid re-encoding when the aggregation rule changes (all reviews vs. only the
non-toxic ones, etc.).

Cache is resumable PER BUCKET of hash(review_id): emb_cache/emb_NNN.npy (float32 N x dim) +
meta_NNN.parquet (review_id, user_key, y_review, split) -- aligned row by row. An already
written bucket is skipped, so the job resumes from where it left off.

Default model: all-MiniLM-L6-v2 (384-d, English), normalize_embeddings=True.

Usage:
    python src/up_embed_reviews.py
    python src/up_embed_reviews.py --model all-MiniLM-L6-v2 --buckets 20 --batch-size 256
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, time, io
import numpy as np, pandas as pd, duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- per-review embedding cache (SBERT)")
    ap.add_argument("--reviews", default=os.path.join(ROOT, "data", "features", "reviews_resolved", "*.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "user_predictor", "emb_cache"))
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--pct", type=int, default=10, help="user subsample: hash(user_key)%%(100/pct)==0")
    ap.add_argument("--buckets", type=int, default=20, help="hash(review_id) buckets for resumable chunking")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mod = max(1, round(100 / args.pct))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out("=" * 78); out(f"USER PREDICTOR -- per-review embeddings | model={args.model} dev={dev}"); out("=" * 78)
    enc = SentenceTransformer(args.model, device=dev)
    dim = enc.get_sentence_embedding_dimension()
    out(f"dim={dim}  subsample=hash(user_key)%{mod}==0  buckets={args.buckets}  lang={args.lang}")

    con = duckdb.connect(); con.sql("PRAGMA threads=4")
    rr = f"read_parquet('{args.reviews.replace(chr(92), '/')}')"

    done_rows = 0; t0 = time.time()
    for b in range(args.buckets):
        f_npy = os.path.join(args.out, f"emb_{b:03d}.npy")
        f_meta = os.path.join(args.out, f"meta_{b:03d}.parquet")
        if os.path.exists(f_npy) and os.path.exists(f_meta):
            n = len(pd.read_parquet(f_meta, columns=["review_id"]))
            out(f"  bucket {b+1}/{args.buckets}: already exists ({n:,} reviews), skipping"); done_rows += n; continue
        df = con.sql(f"""
            SELECT review_id, user_key, y_review, split, coalesce(review_text,'') AS txt
            FROM {rr}
            WHERE review_lang='{args.lang}' AND user_key IS NOT NULL
              AND hash(user_key) % {mod} == 0 AND hash(review_id) % {args.buckets} == {b}
            ORDER BY review_id
        """).df()
        if len(df) == 0:
            out(f"  bucket {b+1}/{args.buckets}: empty");
            pd.DataFrame(columns=["review_id","user_key","y_review","split"]).to_parquet(f_meta)
            np.save(f_npy, np.zeros((0, dim), dtype=np.float32)); continue
        tb = time.time()
        emb = enc.encode(df["txt"].tolist(), batch_size=args.batch_size,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        np.save(f_npy, emb)
        df.drop(columns=["txt"]).to_parquet(f_meta, index=False)
        done_rows += len(df)
        rate = len(df) / max(1e-9, time.time() - tb)
        out(f"  bucket {b+1}/{args.buckets}: {len(df):,} reviews in {time.time()-tb:.0f}s ({rate:.0f} rev/s)  [cum {done_rows:,}]")

    out(f"\nTOTAL: {done_rows:,} reviews embedded in {(time.time()-t0)/60:.1f} min -> {args.out}")
    with open(os.path.join(os.path.dirname(args.out), "embed_reviews_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
