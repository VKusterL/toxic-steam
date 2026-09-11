#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_user_vectors.py  -  User Predictor: aggregate per-review embeddings -> 1 vector per user

Reads the review-level cache (emb_cache/emb_NNN.npy + meta_NNN.parquet from up_embed_reviews.py) and
builds each user's vector as the MEAN of the embeddings of their reviews (a dimension invariant to
the number of reviews). Produces two aggregations:

  - ALL     : mean over ALL of the user's reviews          (primary 'content' feature)
  - NONTOX  : mean over only NON-toxic reviews (y_review=0) (leakage CONTROL: removes the
              text that literally defines the label; if content-only still predicts here,
              the signal is not purely a re-derivation of the tool)

Output (results/user_predictor/):
  user_vectors_all.npy      (U x dim, float32; row i <-> user_key i)
  user_vectors_nontox.npy   (U x dim; users with no non-toxic review are set to NaN -> flagged)
  user_vectors_index.parquet(user_key, n_rev, n_nontox, row)

Usage:
    python src/up_user_vectors.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, glob, io
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- aggregate embeddings per user (mean)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "user_predictor", "emb_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "user_predictor"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metas = sorted(glob.glob(os.path.join(args.cache, "meta_*.parquet")))
    if not metas:
        out(f"ERROR: no meta_*.parquet in {args.cache} (run up_embed_reviews.py first)"); return
    out("=" * 78); out("USER PREDICTOR -- embedding aggregation -> vector per user"); out("=" * 78)
    out(f"buckets in cache: {len(metas)}")

    # 1st pass: user index + dim
    keys = []
    dim = None
    for m in metas:
        k = pd.read_parquet(m, columns=["user_key"])
        keys.append(k["user_key"].to_numpy())
        if dim is None:
            e = np.load(m.replace("meta_", "emb_").replace(".parquet", ".npy"), mmap_mode="r")
            dim = e.shape[1]
    all_keys = np.concatenate(keys)
    uniq = pd.unique(all_keys)            # order of appearance (stable)
    idx_of = {k: i for i, k in enumerate(uniq)}
    U = len(uniq)
    out(f"unique users: {U:,} | dim={dim} | reviews total: {len(all_keys):,}")

    # float32 accumulators (sums of ~hundreds of vectors in [-1,1] -> precision to spare;
    # float32 avoids OOM on mpnet-768: float64 would be ~11GB just for the two accumulators)
    sum_all = np.zeros((U, dim), dtype=np.float32); cnt_all = np.zeros(U, dtype=np.int64)
    sum_nt = np.zeros((U, dim), dtype=np.float32);  cnt_nt = np.zeros(U, dtype=np.int64)

    # 2nd pass: accumulate sums per user
    for j, m in enumerate(metas):
        meta = pd.read_parquet(m, columns=["user_key", "y_review"])
        if len(meta) == 0: continue
        emb = np.load(m.replace("meta_", "emb_").replace(".parquet", ".npy")).astype(np.float32)
        ridx = meta["user_key"].map(idx_of).to_numpy()
        yr = meta["y_review"].to_numpy()
        np.add.at(sum_all, ridx, emb)
        np.add.at(cnt_all, ridx, 1)
        nt = yr == 0
        if nt.any():
            np.add.at(sum_nt, ridx[nt], emb[nt])
            np.add.at(cnt_nt, ridx[nt], 1)
        out(f"  bucket {j+1}/{len(metas)}: +{len(meta):,} reviews", )

    emb_all = (sum_all / np.maximum(cnt_all, 1)[:, None]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        emb_nt = np.where(cnt_nt[:, None] > 0, sum_nt / np.maximum(cnt_nt, 1)[:, None], np.nan).astype(np.float32)

    np.save(os.path.join(args.out, "user_vectors_all.npy"), emb_all)
    np.save(os.path.join(args.out, "user_vectors_nontox.npy"), emb_nt)
    pd.DataFrame({"user_key": uniq, "n_rev": cnt_all, "n_nontox": cnt_nt, "row": np.arange(U)}).to_parquet(
        os.path.join(args.out, "user_vectors_index.parquet"), index=False)

    out(f"\nvector ALL    : {emb_all.shape}  -> user_vectors_all.npy")
    out(f"vector NONTOX : {emb_nt.shape}  ({int((cnt_nt==0).sum()):,} users with no non-toxic review = NaN)")
    out(f"index         : user_vectors_index.parquet")
    with open(os.path.join(args.out, "user_vectors_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
