#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_plots.py  -  User Predictor: paper figures (generated from the results)

Builds the figures from the result artifacts (OOF = source of truth for AUC-PR;
lto_control.csv; modeling table for the exposure ladder; mpnet/bert/llm if present).
Each figure is isolated in try/except so missing data does not break the rest.

Outputs: results/figures/*.png
Usage: python src/up_plots.py
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "results", "replication")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
ARMS = ["content_agnostic", "content", "content_leavetoxicout", "combined"]
ARM_LBL = {"content_agnostic": "profile (agnostic)", "content": "content (emb)",
           "content_leavetoxicout": "leave-toxic-out", "combined": "combined"}

def oof_table(oof_dir):
    rows = []
    for f in glob.glob(os.path.join(oof_dir, "*.parquet")):
        d = pd.read_parquet(f).dropna(subset=["score"])
        if d.y.nunique() < 2: continue
        rows.append(dict(label=d.label.iloc[0], arm=d.arm.iloc[0], model=d.model.iloc[0],
                         auc_pr=average_precision_score(d.y, d.score), auc_roc=roc_auc_score(d.y, d.score),
                         baseline=float(d.y.mean())))
    return pd.DataFrame(rows)

def save(fig, name):
    p = os.path.join(FIG, name); fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print("  ->", p)

def fig_ablation(t):
    for lab in ["count", "rate"]:
        s = t[t.label == lab]
        if s.empty: continue
        best = s.sort_values("auc_pr", ascending=False).groupby("arm").head(1).set_index("arm")
        arms = [a for a in ARMS if a in best.index]
        vals = [best.loc[a, "auc_pr"] for a in arms]; base = s.baseline.iloc[0]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar([ARM_LBL[a] for a in arms], vals, color=["#4C72B0", "#C44E52", "#8172B3", "#55A868"][:len(arms)])
        ax.axhline(base, ls="--", c="gray", label=f"baseline (prev={base:.3f})")
        for b, v, a in zip(bars, vals, arms): ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}\n({best.loc[a,'model']})", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("AUC-PR (best model/arm)"); ax.set_title(f"Feature ablation - label {lab}"); ax.legend()
        save(fig, f"fig_ablation_{lab}.png")

def fig_lto():
    p = os.path.join(REP, "lto_control.csv")
    if not os.path.exists(p): print("  [lto] no lto_control.csv"); return
    d = pd.read_csv(p)
    d = d[d.model == ("xgb" if "xgb" in d.model.values else d.model.iloc[0])]
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(d)); w = 0.27
    ax.bar(x - w, d.ap_agnostic, w, label="profile", color="#55A868")
    ax.bar(x, d.ap_lto, w, label="leave-toxic-out (non-toxic only)", color="#C44E52")
    ax.bar(x + w, d.ap_content, w, label="content (all)", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(d["pop"], fontsize=8); ax.set_ylabel("AUC-PR")
    ax.set_title("Diffuse trace (clean LTO): profile < LTO < content"); ax.legend(fontsize=8)
    save(fig, "fig_lto_clean.png")

def fig_exposure():
    p = os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet")
    if not os.path.exists(p): return
    d = pd.read_parquet(p, columns=["y_count", "n_reviews"])
    bk = pd.cut(d.n_reviews, [0, 1, 4, 9, 10**9], labels=["1", "2-4", "5-9", ">=10"])
    prev = d.groupby(bk)["y_count"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(prev.index.astype(str), prev.values * 100, color="#C44E52")
    for b, v in zip(bars, prev.values): ax.text(b.get_x()+b.get_width()/2, v*100+0.3, f"{v*100:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("user n_reviews"); ax.set_ylabel("% toxic users (count)")
    ax.set_title("Exposure ladder: more reviews -> more likely to be toxic")
    save(fig, "fig_exposure_ladder.png")

def fig_methods(t):
    """Headline: profile vs classical embedding vs BERT vs LLM (AUC-PR, count label)."""
    s = t[t.label == "count"]
    if s.empty: return
    pts = {}
    ag = s[s.arm == "content_agnostic"].auc_pr.max(); pts["profile\n(content-agnostic)"] = ag
    co = s[s.arm == "content"].auc_pr.max(); pts["embedding+classical\n(content)"] = co
    bj = glob.glob(os.path.join(ROOT, "results", "replication_bert", "fold_*.json"))
    if bj:
        import json; aps = [json.load(open(f))["auc_pr"] for f in bj]; pts["BERT\n(end-to-end)"] = float(np.mean(aps))
    lp = os.path.join(ROOT, "results", "lente4_llm", "llm_user_metrics.csv")
    if os.path.exists(lp):
        l = pd.read_csv(lp);
        if "auc_pr" in l.columns and len(l): pts["LLM Qwen\n(reads reviews)"] = float(l.auc_pr.iloc[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(pts), list(pts.values()), color=["#55A868", "#4C72B0", "#8172B3", "#CCB974"][:len(pts)])
    ax.axhline(s.baseline.iloc[0], ls="--", c="gray", label=f"baseline {s.baseline.iloc[0]:.3f}")
    for b, v in zip(bars, pts.values()): ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("AUC-PR (count)"); ax.set_title("Modeling paradigms (count label)"); ax.legend(fontsize=8)
    save(fig, "fig_methods.png")

def fig_dimension(t):
    mp = oof_table(os.path.join(ROOT, "results", "replication_mpnet", "oof"))
    if mp.empty: print("  [dim] no mpnet OOF"); return
    fig, ax = plt.subplots(figsize=(6, 4)); w = 0.35
    arms = ["content", "combined"]; x = np.arange(len(arms))
    mini = [t[(t.label=="count")&(t.arm==a)].auc_pr.max() for a in arms]
    mpn = [mp[(mp.label=="count")&(mp.arm==a)].auc_pr.max() for a in arms]
    ax.bar(x - w/2, mini, w, label="MiniLM-384", color="#4C72B0")
    ax.bar(x + w/2, mpn, w, label="mpnet-768", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("AUC-PR (count)")
    ax.set_title("Embedding dimension: 384 vs 768"); ax.legend()
    save(fig, "fig_dimension.png")

def main():
    print("=== generating figures -> results/figures/ ===")
    t = oof_table(os.path.join(REP, "oof"))
    if t.empty: print("  [warning] no OOF in results/replication/oof -- run up_train_userlevel first")
    for name, fn in [("ablation", lambda: fig_ablation(t)), ("lto", fig_lto), ("ladder", fig_exposure),
                     ("methods", lambda: fig_methods(t)), ("dimension", lambda: fig_dimension(t))]:
        try: fn()
        except Exception as e: print(f"  [skipped {name}] {e}")
    print("=== done ===")

if __name__ == "__main__":
    main()
