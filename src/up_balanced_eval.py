#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_balanced_eval.py  -  User Predictor: BALANCED evaluation

EXTRA (this does not replace the real prevalence, which is the primary/honest result). Here we report the two
evaluations side by side:
  - REAL  : platform prevalence (~4% count / ~12.8% rate)  -> honest deployment number
  - BALANCED : all positives + an equal number of sampled negatives (50/50)

The balanced result is the mean +/- standard deviation over K draws of negatives (stability). The
F1-macro threshold is chosen WITHIN each evaluation. Covers the classical models (MiniLM + mpnet),
LLMs, and BERT when the fold_*_preds.csv files produced by up_bert_user.py are present.

Outputs: results/replication/balanced_eval.csv + results/figures/fig_balanced.png
Usage: python src/up_balanced_eval.py
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)

def bt(y, s, maxn=50000, seed=0):
    # subsample only for the threshold SEARCH (the threshold is robust); final metrics use the full array
    if len(y) > maxn:
        i = np.random.default_rng(seed).choice(len(y), maxn, replace=False); y, s = y[i], s[i]
    g = np.unique(np.quantile(s, np.linspace(.01, .999, 200))); b, bf = .5, -1
    for t in g:
        f = f1_score(y, (s >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, b = f, float(t)
    return b

def real_metrics(y, s):
    thr = bt(y, s); yp = (s >= thr).astype(int)
    return dict(f1m=f1_score(y, yp, average="macro"), roc=roc_auc_score(y, s),
                aucpr=average_precision_score(y, s), prev=float(y.mean()))

def balanced_metrics(y, s, K=10, seed=42):
    rng = np.random.default_rng(seed); pos = np.where(y == 1)[0]; neg_all = np.where(y == 0)[0]
    f1s, rocs, aps = [], [], []
    for k in range(K):
        neg = rng.choice(neg_all, size=len(pos), replace=False)
        idx = np.concatenate([pos, neg]); yb, sb = y[idx], s[idx]
        thr = bt(yb, sb); yp = (sb >= thr).astype(int)
        f1s.append(f1_score(yb, yp, average="macro")); rocs.append(roc_auc_score(yb, sb)); aps.append(average_precision_score(yb, sb))
    return dict(f1m=np.mean(f1s), f1m_sd=np.std(f1s, ddof=1), roc=np.mean(rocs), aucpr=np.mean(aps))

def collect():
    rows = []
    for src, oofdir in [("MiniLM", os.path.join(ROOT, "results", "replication", "oof")),
                        ("mpnet", os.path.join(ROOT, "results", "replication_mpnet", "oof"))]:
        for f in glob.glob(os.path.join(oofdir, "*.parquet")):
            d = pd.read_parquet(f).dropna(subset=["score"])
            if d.y.nunique() < 2: continue
            y, s = d.y.to_numpy(), d.score.to_numpy()
            r = real_metrics(y, s); b = balanced_metrics(y, s)
            rows.append(dict(src=src, label=d.label.iloc[0], arm=d.arm.iloc[0], model=d.model.iloc[0],
                             real_f1m=r["f1m"], real_roc=r["roc"], real_aucpr=r["aucpr"], prev=r["prev"],
                             bal_f1m=b["f1m"], bal_f1m_sd=b["f1m_sd"], bal_roc=b["roc"], bal_aucpr=b["aucpr"]))
    # LLMs as classifiers (all preds_*.csv)
    for lp in sorted(glob.glob(os.path.join(ROOT, "results", "lente4_llm", "llm_user_preds_*.csv"))):
        mod = os.path.basename(lp).replace("llm_user_preds_", "").replace(".csv", "")
        d = pd.read_csv(lp)
        if "_score" not in d.columns: continue
        y, s = d["y_count"].to_numpy(), d["_score"].to_numpy()
        r = real_metrics(y, s); b = balanced_metrics(y, s)
        rows.append(dict(src="LLM", label="count", arm=f"llm_{mod}", model=mod,
                         real_f1m=r["f1m"], real_roc=r["roc"], real_aucpr=r["aucpr"], prev=r["prev"],
                         bal_f1m=b["f1m"], bal_f1m_sd=b["f1m_sd"], bal_roc=b["roc"], bal_aucpr=b["aucpr"]))
    # BERT end-to-end: available for new/corrected runs of up_bert_user.py
    for src, bdir in [("DistilBERT", os.path.join(ROOT, "results", "replication_bert")),
                      ("DistilBERT", os.path.join(ROOT, "results", "replication_bert_undersampled"))]:
        fps = sorted(glob.glob(os.path.join(bdir, "fold_*_preds.csv")))
        if not fps: continue
        d = pd.concat([pd.read_csv(p) for p in fps], ignore_index=True).dropna(subset=["score"])
        if d.empty or d.y.nunique() < 2: continue
        y, s = d["y"].to_numpy(), d["score"].to_numpy()
        r = real_metrics(y, s); b = balanced_metrics(y, s)
        rows.append(dict(src=src, label="count", arm="bert_endtoend", model=os.path.basename(bdir),
                         real_f1m=r["f1m"], real_roc=r["roc"], real_aucpr=r["aucpr"], prev=r["prev"],
                         bal_f1m=b["f1m"], bal_f1m_sd=b["f1m_sd"], bal_roc=b["roc"], bal_aucpr=b["aucpr"]))
    return pd.DataFrame(rows)

def main():
    t = collect()
    if t.empty: print("no OOF"); return
    t = t.round(4)
    t.to_csv(os.path.join(ROOT, "results", "replication", "balanced_eval.csv"), index=False)
    # best model per (src,label,arm) for the readable table
    print("=== REAL vs BALANCED (best model per arm; F1-macro) ===")
    print(f'{"src":7s} {"label":6s} {"arm":22s} | REAL F1m/ROC/AP-prev | BALANC F1m/ROC/AP')
    best = t.sort_values("bal_f1m", ascending=False).groupby(["src", "label", "arm"]).head(1)
    for _, r in best.sort_values(["label", "arm", "src"]).iterrows():
        print(f'{r.src:7s} {r.label:6s} {r.arm:22s} | {r.real_f1m:.3f}/{r.real_roc:.3f}/{r.real_aucpr:.3f}@{r.prev*100:.0f}% '
              f'| {r.bal_f1m:.3f}±{r.bal_f1m_sd:.3f}/{r.bal_roc:.3f}/{r.bal_aucpr:.3f}  ({r.model})')
    # figure: REAL vs BALANCED (count, MiniLM, best per arm)
    s = t[(t.label == "count") & (t.src == "MiniLM")].sort_values("bal_f1m", ascending=False).groupby("arm").head(1)
    order = ["content_agnostic", "content", "combined", "content_leavetoxicout"]
    s = s.set_index("arm").reindex([a for a in order if a in s.arm.values]).dropna(how="all")
    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(s)); w = 0.38
    ax.bar(x - w/2, s.real_f1m, w, label="REAL prev. (~4%)", color="#4C72B0")
    ax.bar(x + w/2, s.bal_f1m, w, yerr=s.bal_f1m_sd, label="BALANCED 50/50 (=papers)", color="#C44E52", capsize=3)
    ax.axhline(0.85, ls="--", c="gray", label="Pinto 2024 (F1 0.85)")
    ax.axhline(0.84, ls=":", c="dimgray", label="An et al. 2021 (F1 0.84)")
    ax.set_xticks(x); ax.set_xticklabels(s.index, fontsize=8); ax.set_ylabel("F1-macro"); ax.set_ylim(0.4, 1.0)
    ax.set_title("REAL vs BALANCED evaluation (count) - matching the literature under their protocol")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_balanced.png"), dpi=150); plt.close(fig)
    print(f"\n-> results/replication/balanced_eval.csv + results/figures/fig_balanced.png")

if __name__ == "__main__":
    main()
