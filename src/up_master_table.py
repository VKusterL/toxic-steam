#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_master_table.py  -  User Predictor: consolidated MASTER TABLE (with standard deviation)

Consolidates ALL metrics for ALL models/arms/encoders from the OOF predictions (the source of
truth; resolves the metrics_agg overwrite). For each (encoder, label, arm, model) it reports,
with STANDARD DEVIATION:
  - AUC-PR, ROC, macro-F1, and the TARGET CLASS (F1+, Prec+, Rec+)
  - under TWO evaluations: REAL (platform prevalence) and BALANCED (50/50 = An/Pinto protocol)

Standard deviation:
  - REAL  -> classical models: across the 5 folds (reconstructed via a deterministic
             StratifiedKFold(5, seed=42), matched to the OOF by user_key; macro-F1
             threshold tuned per fold).
             LLMs: across the 5 blocks of 200 users from the stratified sample.
  - BALANC-> across K draws of negatives (all positives + an equal number of negatives).

LLMs: AUC-PR/ROC use _score; F1/Prec/Rec use each model's own binary decision (_pred),
as in up_llm_user.py -- no threshold is tuned on the evaluated partition.
BERT: DistilBERT = undersampled 10-fold run (threshold on an internal validation split,
replication_bert_undersampled, with the target class); mpnet-base = old 5-fold run
(threshold on the test set; use only AUC-PR/ROC for ranking).

Output: results/replication/master_table.csv  (+ readable console output)
Usage: python src/up_master_table.py
"""
from __future__ import annotations
import os, glob, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet")
LABELS = {"count": ("y_count", None), "rate": ("y_rate", "rate_pop")}

def bt(y, s, maxn=50000, seed=0):
    if len(y) > maxn: i = np.random.default_rng(seed).choice(len(y), maxn, replace=False); y, s = y[i], s[i]
    g = np.unique(np.quantile(s, np.linspace(.01, .999, 200))); b, bf = .5, -1
    for t in g:
        f = f1_score(y, (s >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, b = f, float(t)
    return b

def m_at(y, s, thr):
    yp = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average=None, labels=[0, 1], zero_division=0)
    return dict(aucpr=average_precision_score(y, s), roc=roc_auc_score(y, s),
                f1m=f1_score(y, yp, average="macro", zero_division=0), f1p=f1[1], precp=pr[1], recp=rc[1])

KEYS = ["aucpr", "roc", "f1m", "f1p", "precp", "recp"]

def agg(rows):
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if not np.isnan(r[k])]
        out[k] = np.mean(v) if v else np.nan
        out[k + "_sd"] = np.std(v, ddof=1) if len(v) > 1 else 0.0
    return out

def real_byfold(y, s, fold):
    rows = []
    for f in np.unique(fold):
        m = fold == f
        if m.sum() < 2 or len(np.unique(y[m])) < 2: continue
        thr = bt(y[m], s[m]); rows.append(m_at(y[m], s[m], thr))
    return agg(rows)

def bal_bydraw(y, s, K=10, seed=42):
    rng = np.random.default_rng(seed); pos = np.where(y == 1)[0]; negall = np.where(y == 0)[0]
    rows = []
    for k in range(K):
        neg = rng.choice(negall, len(pos), replace=False); idx = np.concatenate([pos, neg])
        thr = bt(y[idx], s[idx]); rows.append(m_at(y[idx], s[idx], thr))
    return agg(rows)

def reconstruct_folds(label, k=5, seed=42):
    df = pd.read_parquet(MOD); df["user_key"] = df["user_key"].astype(str)
    tgt, sub = LABELS[label]
    s = df if sub is None else df[df[sub] == 1]
    y = s[tgt].to_numpy().astype(int)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.full(len(s), -1)
    for i, (_, te) in enumerate(skf.split(s, y)): fold[te] = i
    return dict(zip(s["user_key"].to_numpy(), fold))

def main():
    foldmap = {lab: reconstruct_folds(lab) for lab in LABELS}
    rows = []
    # --- classical models (MiniLM + mpnet) ---
    for src, oof in [("MiniLM", "results/replication/oof"), ("mpnet", "results/replication_mpnet/oof")]:
        for f in glob.glob(os.path.join(ROOT, oof, "*.parquet")):
            d = pd.read_parquet(f).dropna(subset=["score"])
            if d.y.nunique() < 2: continue
            lab, arm, mod = d.label.iloc[0], d.arm.iloc[0], d.model.iloc[0]
            d["user_key"] = d["user_key"].astype(str)
            fold = d["user_key"].map(foldmap[lab]).fillna(-1).to_numpy()
            y, s = d.y.to_numpy(), d.score.to_numpy()
            R = real_byfold(y, s, fold); B = bal_bydraw(y, s)
            rows.append(dict(src=src, label=lab, arm=arm, model=mod, eval="REAL", prev=round(float(y.mean()), 4), **R))
            rows.append(dict(src=src, label=lab, arm=arm, model=mod, eval="BALANC", prev=0.5, **B))
    # --- LLMs as classifiers (all preds_*.csv; 5 blocks of 200) ---
    # AUC-PR/ROC use _score; F1/Prec/Rec use each model's own binary decision
    # (_pred), as in up_llm_user.py -- no threshold is tuned on the evaluated partition.
    def llm_m(y, s, yp):
        pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average=None, labels=[0, 1], zero_division=0)
        return dict(aucpr=average_precision_score(y, s), roc=roc_auc_score(y, s),
                    f1m=f1_score(y, yp, average="macro", zero_division=0), f1p=f1[1], precp=pr[1], recp=rc[1])
    for lp in sorted(glob.glob(os.path.join(ROOT, "results", "lente4_llm", "llm_user_preds_*.csv"))):
        mod = os.path.basename(lp).replace("llm_user_preds_", "").replace(".csv", "")
        d = pd.read_csv(lp)
        if "_score" not in d.columns or "block" not in d.columns or "_pred" not in d.columns: continue
        y = d["y_count"].to_numpy().astype(int); s = d["_score"].to_numpy()
        yp = d["_pred"].to_numpy().astype(int); blk = d["block"].to_numpy()
        per_blk = []
        for b in np.unique(blk):
            m = blk == b
            if len(np.unique(y[m])) < 2: continue
            per_blk.append(llm_m(y[m], s[m], yp[m]))
        R = agg(per_blk)
        rng = np.random.default_rng(42); pos = np.where(y == 1)[0]; negall = np.where(y == 0)[0]
        per_draw = []
        for _ in range(10):
            neg = rng.choice(negall, len(pos), replace=False); idx = np.concatenate([pos, neg])
            per_draw.append(llm_m(y[idx], s[idx], yp[idx]))
        B = agg(per_draw)
        rows.append(dict(src="LLM", label="count", arm=f"llm_{mod}", model=mod, eval="REAL", prev=round(float(y.mean()), 4), **R))
        rows.append(dict(src="LLM", label="count", arm=f"llm_{mod}", model=mod, eval="BALANC", prev=0.5, **B))
    # --- BERT (per-fold metrics from bert_user_metrics.csv) ---
    # DistilBERT: canonical undersampled 10-fold run (threshold on an internal validation split).
    # The old run results/replication_bert (5-fold, threshold on the test set itself) is NO
    # longer included: it was leaky and conflicted with the paper.
    for src, d2 in [("DistilBERT", "results/replication_bert_undersampled"),
                    ("mpnet-base-BERT", "results/replication_bert_mpnet")]:
        fp = os.path.join(ROOT, d2, "bert_user_metrics.csv")
        if not os.path.exists(fp): continue
        b = pd.read_csv(fp)
        def gv(col):
            if col not in b.columns or b[col].isna().all(): return (np.nan, 0.0)
            v = b[col].dropna().to_numpy()
            return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        a, asd = gv("auc_pr"); r, rsd = gv("auc_roc"); fm, fmsd = gv("f1_macro")
        f1p, f1psd = gv("f1_pos"); pp, ppsd = gv("precision_pos"); rp, rpsd = gv("recall_pos")
        rows.append(dict(src=src, label="count", arm="bert_endtoend", model=src, eval="REAL", prev=0.04,
                         aucpr=a, aucpr_sd=asd, roc=r, roc_sd=rsd, f1m=fm, f1m_sd=fmsd,
                         f1p=f1p, f1p_sd=f1psd, precp=pp, precp_sd=ppsd, recp=rp, recp_sd=rpsd))
    t = pd.DataFrame(rows)
    for c in t.columns:
        if t[c].dtype == float: t[c] = t[c].round(4)
    t.to_csv(os.path.join(ROOT, "results", "replication", "master_table.csv"), index=False)
    # console: best model per (src,label,arm,eval), focused on the target class
    print("=== MASTER TABLE (best model per arm; mean +/- std) ===")
    print("TARGET class (toxic): F1+ / Prec+ / Rec+   [and AUC-PR | macro-F1]")
    best = t.sort_values("f1m", ascending=False).groupby(["src", "label", "arm", "eval"]).head(1)
    for lab in ["count", "rate"]:
        print(f"\n--- {lab} ---")
        for _, r in best[best.label == lab].sort_values(["arm", "src", "eval"]).iterrows():
            f1p = f"{r.f1p:.3f}±{r.f1p_sd:.3f}" if not np.isnan(r.f1p) else "   N/A   "
            pp = f"{r.precp:.3f}" if not np.isnan(r.precp) else " N/A "
            rp = f"{r.recp:.3f}" if not np.isnan(r.recp) else " N/A "
            print(f"  {r.src:14s} {r.arm:20s} {r['eval']:6s} | F1+ {f1p} P {pp} R {rp} | AUC-PR {r.aucpr:.3f}±{r.aucpr_sd:.3f} F1m {r.f1m:.3f}")
    print(f"\n-> results/replication/master_table.csv  ({len(t)} rows)")

if __name__ == "__main__":
    main()
