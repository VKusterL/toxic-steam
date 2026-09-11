#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_build_dataset.py  -  User Predictor: modeling substrate (10% slice + labels + CV)

This script materializes the modeling base from data/features/user_features.parquet:

  - Deterministic 10% SLICE: hash(user_key) % 10 == 0  (~620k users; preserves the
    real prevalence because the hash is independent of the label).
  - TWO labels (alternative operationalizations, NOT a union -- see the pivot memory):
        y_count = y_user = (n_toxic >= 1)              [count, K=1]   prev ~3.97%
        y_rate  = (n_toxic / n_reviews_diag >= rho)    [proportion, rho=0.05]
    Keeps n_reviews_diag for the proportion population (n_reviews >= k_min) and n_toxic for auditing.
  - content-agnostic FEATURES = profile ONLY (no residue of any review; see the CONTENT_AGNOSTIC list).
  - CV FOLDS: StratifiedKFold(k, seed) over y_count (and also over y_rate, its own column).

Output: results/user_predictor/modeling_users_10pct.parquet  (+ report).
The embedding vector (content) is built separately by up_embed_reviews.py; here only the tabular part.

Usage:
    python src/up_build_dataset.py
    python src/up_build_dataset.py --pct 10 --kfolds 5 --rho 0.05 --k-min 5 --seed 42
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, io
import numpy as np, pandas as pd, duckdb
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k); print(*a, **k, file=_BUF)

# content-agnostic = Steam profile ONLY (nothing derived from reviews, no n_reviews).
# Excluded on purpose (review-derived): n_reviews, n_distinct_games, pct_recommended,
# hours_*, tenure_days, days_since_last_review, reviews_per_year, pct_competitive/violent/comedy,
# n_review_langs (review activity -> left out of content-agnostic by default).
CONTENT_AGNOSTIC = [
    "profile_level", "profile_level_missing",
    "library_size", "library_size_missing",
    "awards", "insignias", "screenshots", "workshop_items", "guides", "arts", "groups",
    "profile_visibility", "country", "country_is_brazil", "has_ban", "ban_recency_days",
]
DIAGNOSTIC = ["n_toxic", "n_reviews_diag", "n_reviews"]  # kept for the proportion label + auditing

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- modeling substrate (10% + labels + CV)")
    ap.add_argument("--features", default=os.path.join(ROOT, "data", "features", "user_features.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "user_predictor"))
    ap.add_argument("--pct", type=int, default=10, help="percentage of the base for CV (slice hash%%(100/pct))")
    ap.add_argument("--kfolds", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.05, help="proportion threshold for y_rate")
    ap.add_argument("--k-min", type=int, default=5, help="min n_reviews for the proportion-label population")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mod = max(1, round(100 / args.pct))

    con = duckdb.connect(); con.sql("PRAGMA threads=4")
    uf = f"read_parquet('{args.features.replace(chr(92), '/')}')"
    cols = ", ".join(CONTENT_AGNOSTIC + DIAGNOSTIC)

    out("=" * 78); out("USER PREDICTOR -- modeling substrate"); out("=" * 78)
    out(f"source: {args.features}")
    out(f"slice: hash(user_key) % {mod} == 0  (~{args.pct}% of the base)")
    df = con.sql(f"""
        SELECT user_key, y_user AS y_count, split, {cols}
        FROM {uf}
        WHERE hash(user_key) % {mod} == 0
    """).df()

    # proportion label
    rate = df["n_toxic"].astype(float) / df["n_reviews_diag"].clip(lower=1).astype(float)
    df["y_rate"] = (rate >= args.rho).astype(int)
    df["rate_pop"] = (df["n_reviews"] >= args.k_min).astype(int)  # "valid" population for the proportion

    out(f"\nusers in the slice: {len(df):,}")
    out(f"  prev y_count (>=1 toxic)       : {df['y_count'].mean()*100:.3f}%  (n_pos={int(df['y_count'].sum()):,})")
    out(f"  prev y_rate  (>={args.rho:.0%} toxic)     : {df['y_rate'].mean()*100:.3f}%  (n_pos={int(df['y_rate'].sum()):,})")
    sub = df[df["rate_pop"] == 1]
    out(f"  proportion population (n_reviews>={args.k_min}): {len(sub):,} users; prev y_rate here = {sub['y_rate'].mean()*100:.3f}%")
    out(f"  n_reviews: mediana={int(df['n_reviews'].median())}  p90={int(df['n_reviews'].quantile(0.9))}  max={int(df['n_reviews'].max())}")

    # stratified folds (one per label; reproducible)
    for tgt in ["y_count", "y_rate"]:
        skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
        fold = np.full(len(df), -1, dtype=int)
        for i, (_, te) in enumerate(skf.split(df, df[tgt].to_numpy())):
            fold[te] = i
        df[f"fold_{tgt}"] = fold
    out(f"\nStratifiedKFold k={args.kfolds} seed={args.seed} -> columns fold_y_count / fold_y_rate")
    # check: prevalence per fold (should be stable ~ global prev)
    chk = df.groupby("fold_y_count")["y_count"].mean().mul(100).round(3)
    out(f"  prev y_count per fold: {chk.to_dict()}")

    path = os.path.join(args.out, "modeling_users_10pct.parquet")
    df.to_parquet(path, index=False)
    out(f"\n-> {path}  ({len(df):,} rows, {df.shape[1]} columns)")
    out(f"   content-agnostic ({len(CONTENT_AGNOSTIC)}): {CONTENT_AGNOSTIC}")
    with open(os.path.join(args.out, "build_dataset_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
