#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_significance.py  -  User Predictor: paired significance + stratification by n_reviews

About the out-of-fold predictions (results/replication/oof/<label>_<arm>_<model>.parquet):
  1) AUC-PR / ROC by n_reviews BUCKET {1, 2-4, 5-9, >=10}  -> exposes the trivialization of
     the count-label for 1-review users (where y_count == y of the single review).
  2) PAIRED BOOTSTRAP per user on the delta-AUC-PR between arms (same model): measures whether
     'content' and 'combined' significantly outperform the 'content_agnostic' headline.
     95% CI by percentile; Holm-Bonferroni correction over the family of comparisons.

Outputs: results/replication/bucket_metrics.csv, pairwise_bootstrap.csv
Usage: python src/up_significance.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, glob, io
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

def bucket(n):
    return np.where(n == 1, "1", np.where(n <= 4, "2-4", np.where(n <= 9, "5-9", ">=10")))

def ap(y, s):
    return float(average_precision_score(y, s)) if (y.sum() > 0 and y.sum() < len(y)) else float("nan")

def main():
    ap_ = argparse.ArgumentParser(description="User Predictor -- significance + buckets")
    ap_.add_argument("--oof", default=os.path.join(ROOT, "results", "replication", "oof"))
    ap_.add_argument("--out", default=os.path.join(ROOT, "results", "replication"))
    ap_.add_argument("--boot", type=int, default=2000)
    ap_.add_argument("--baseline-arm", default="content_agnostic")
    ap_.add_argument("--baseline-model", default="linsvc", help="Base model for model comparisons (e.g. testing whether linsvc beats the others)")
    ap_.add_argument("--seed", type=int, default=42)
    args = ap_.parse_args()
    files = sorted(glob.glob(os.path.join(args.oof, "*.parquet")))
    if not files:
        out(f"no OOF in {args.oof} (run up_train_userlevel.py)"); return
    rng = np.random.default_rng(args.seed)
    out("=" * 78); out("USER PREDICTOR -- paired significance + n_reviews buckets"); out("=" * 78)

    frames = {}
    for f in files:
        d = pd.read_parquet(f); key = (d["label"].iloc[0], d["arm"].iloc[0], d["model"].iloc[0]); frames[key] = d

    # 1) metrics by bucket
    brows = []
    for (lab, arm, mdl), d in frames.items():
        d = d.dropna(subset=["score"]); bk = bucket(d["n_reviews"].to_numpy())
        for b in ["1", "2-4", "5-9", ">=10"]:
            m = bk == b
            if m.sum() == 0: continue
            yb, sb = d["y"].to_numpy()[m], d["score"].to_numpy()[m]
            brows.append(dict(label=lab, arm=arm, model=mdl, bucket=b, n=int(m.sum()),
                              prev=float(yb.mean()), auc_pr=ap(yb, sb),
                              auc_roc=float(roc_auc_score(yb, sb)) if len(np.unique(yb)) > 1 else np.nan))
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(args.out, "bucket_metrics.csv"), index=False)
    out(f"\n[bucket_metrics] {len(bdf)} rows -> bucket_metrics.csv")
    if len(bdf):
        piv = bdf[bdf.bucket.isin(["1", ">=10"])].pivot_table(
            index=["label", "model"], columns=["arm", "bucket"], values="auc_pr")
        out("AUC-PR by bucket (1 vs >=10), excerpt:")
        out(piv.round(3).to_string())

    # 2) paired bootstrap per user: <arm> vs baseline (same label+model)
    prows = []
    labels = sorted({k[0] for k in frames})
    for lab in labels:
        models = sorted({k[2] for k in frames if k[0] == lab})
        for mdl in models:
            base = frames.get((lab, args.baseline_arm, mdl))
            if base is None: continue
            base = base.set_index("user_key")
            for arm in sorted({k[1] for k in frames if k[0] == lab and k[2] == mdl}):
                if arm == args.baseline_arm: continue
                cur = frames[(lab, arm, mdl)].set_index("user_key")
                j = base[["y", "score"]].join(cur[["score"]], lsuffix="_b", rsuffix="_a", how="inner").dropna()
                if len(j) < 50 or j["y"].sum() == 0: continue
                y = j["y"].to_numpy(); sb = j["score_b"].to_numpy(); sa = j["score_a"].to_numpy()
                d_obs = ap(y, sa) - ap(y, sb)
                deltas = np.empty(args.boot)
                n = len(y)
                for b in range(args.boot):
                    idx = rng.integers(0, n, n)
                    yb = y[idx]
                    if yb.sum() == 0 or yb.sum() == len(yb): deltas[b] = np.nan; continue
                    deltas[b] = ap(yb, sa[idx]) - ap(yb, sb[idx])
                lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
                prows.append(dict(label=lab, model=mdl, arm=arm, baseline=args.baseline_arm,
                                  delta_auc_pr=round(d_obs, 4), ci_lo=round(float(lo), 4),
                                  ci_hi=round(float(hi), 4), excludes_0=bool(lo > 0 or hi < 0)))
    pdf = pd.DataFrame(prows)
    # Informative Holm-Bonferroni: sort by |delta|, flag those that survive with CI>0
    pdf.to_csv(os.path.join(args.out, "pairwise_bootstrap.csv"), index=False)
    out(f"\n[pairwise_bootstrap] {len(pdf)} comparisons -> pairwise_bootstrap.csv")
    if len(pdf):
        out(pdf.sort_values(["label", "model", "arm"]).to_string(index=False))

    # 3) paired bootstrap per user: <model> vs baseline_model (same label+arm)
    mrows = []
    for lab in labels:
        arms = sorted({k[1] for k in frames if k[0] == lab})
        for arm in arms:
            base = frames.get((lab, arm, args.baseline_model))
            if base is None: continue
            base = base.set_index("user_key")
            models = sorted({k[2] for k in frames if k[0] == lab and k[1] == arm})
            for mdl in models:
                if mdl == args.baseline_model: continue
                cur = frames[(lab, arm, mdl)].set_index("user_key")
                j = base[["y", "score"]].join(cur[["score"]], lsuffix="_b", rsuffix="_a", how="inner").dropna()
                if len(j) < 50 or j["y"].sum() == 0: continue
                y = j["y"].to_numpy(); sb = j["score_b"].to_numpy(); sa = j["score_a"].to_numpy()
                # d_obs > 0 means mdl outperformed baseline_model (linsvc). d_obs < 0 means linsvc is better.
                d_obs = ap(y, sa) - ap(y, sb)
                deltas = np.empty(args.boot)
                n = len(y)
                for b in range(args.boot):
                    idx = rng.integers(0, n, n)
                    yb = y[idx]
                    if yb.sum() == 0 or yb.sum() == len(yb): deltas[b] = np.nan; continue
                    deltas[b] = ap(yb, sa[idx]) - ap(yb, sb[idx])
                lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
                mrows.append(dict(label=lab, arm=arm, model=mdl, baseline_model=args.baseline_model,
                                  delta_auc_pr=round(d_obs, 4), ci_lo=round(float(lo), 4),
                                  ci_hi=round(float(hi), 4), excludes_0=bool(lo > 0 or hi < 0)))
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(os.path.join(args.out, "pairwise_model_bootstrap.csv"), index=False)
    out(f"\n[pairwise_model_bootstrap] {len(mdf)} model comparisons -> pairwise_model_bootstrap.csv")
    if len(mdf):
        out("Values < 0 indicate the baseline_model (linsvc) is superior. Values > 0 indicate the <model> is superior.")
        out(mdf.sort_values(["label", "arm", "model"]).to_string(index=False))
    with open(os.path.join(args.out, "significance_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
