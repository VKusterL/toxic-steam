#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_lto_control.py  -  User Predictor: clean LEAVE-TOXIC-OUT control (diffuse toxic trait)

Central validity QUESTION: is the 'content' arm's signal a tautology (it re-reads the toxic
review that DEFINES the label), or is there a DIFFUSE TRAIT (toxic users write differently even
in their NON-toxic reviews)? The content_leavetoxicout arm of the full matrix has an ARTIFACT:
for users who only have toxic reviews (n_nontox=0), the vector-without-toxics becomes NaN ->
imputed, and this MISSINGNESS becomes a PERFECT proxy for the label (AUC-PR=1.000 in the
1-review bucket).

This script runs the CLEAN test: it restricts the population to users with >=1 NON-toxic review
(the LTO vector is always real, no missingness) and compares, on the SAME population and SAME
folds:
  content_agnostic  (profile-only)         -- non-circular floor
  content           (mean over ALL reviews) -- ceiling (includes the toxic review = tautology)
  content_lto       (mean over NON-toxic only) -- if > agnostic => DIFFUSE TRAIT; gap to content = tautology
Reports on two populations: (P1) n_nontox>=1, (P2) n_nontox>=1 AND n_reviews>=2 (excludes
1-review users, who are all negative). StratifiedKFold folds re-stratified WITHIN the population.

Usage: python src/up_lto_control.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, io
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)
PROFILE_CORE = ["profile_level", "profile_level_missing", "library_size", "library_size_missing",
                "awards", "insignias", "screenshots", "workshop_items", "guides", "arts", "groups",
                "profile_visibility", "country", "country_is_brazil"]

def best_thr(y, p):
    g = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 200))); bt, bf = 0.5, -1
    for t in g:
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

def prep_profile():
    num = [c for c in PROFILE_CORE if c != "country"]
    return ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                              ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15, sparse_output=False), ["country"])])
def prep_emb(dim):
    return Pipeline([("i", SimpleImputer(strategy="mean")), ("s", StandardScaler())])

def run_arm(Xdf, y, folds, prep, model_factory):
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        pipe = Pipeline([("p", prep), ("c", model_factory())]) if not isinstance(prep, Pipeline) else \
               Pipeline([("p", prep), ("c", model_factory())])
        pipe.fit(Xdf.iloc[tr], y[tr])
        s = pipe.predict_proba(Xdf.iloc[te])[:, 1] if hasattr(pipe[-1], "predict_proba") else pipe.decision_function(Xdf.iloc[te])
        oof[te] = s
    ap = average_precision_score(y, oof); roc = roc_auc_score(y, oof)
    thr = best_thr(y, oof); kap = cohen_kappa_score(y, (oof >= thr).astype(int))
    return ap, roc, kap, oof

def boot_delta(y, sa, sb, B=2000, seed=42):
    rng = np.random.default_rng(seed); n = len(y); d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); yb = y[i]
        if yb.sum() == 0 or yb.sum() == n: d[b] = np.nan; continue
        d[b] = average_precision_score(yb, sa[i]) - average_precision_score(yb, sb[i])
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))

def main():
    ap_ = argparse.ArgumentParser(description="User Predictor -- clean leave-toxic-out control")
    ap_.add_argument("--data", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap_.add_argument("--vecdir", default=os.path.join(ROOT, "results", "user_predictor"))
    ap_.add_argument("--out", default=os.path.join(ROOT, "results", "replication"))
    ap_.add_argument("--kfolds", type=int, default=10); ap_.add_argument("--seed", type=int, default=42)
    args = ap_.parse_args()
    from xgboost import XGBClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    NJ = 4  # cores per model (leaves headroom for the OS; avoids freezing the machine)
    MODELS = {"xgb": lambda: XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=NJ, random_state=42),
              "histgb": lambda: HistGradientBoostingClassifier(random_state=42),
              "logreg": lambda: LogisticRegression(max_iter=1000, n_jobs=NJ)}

    df = pd.read_parquet(args.data)
    idx = pd.read_parquet(os.path.join(args.vecdir, "user_vectors_index.parquet"))
    # mmap_mode='r': does not load the 918k x dim matrix into RAM (only the indexed rows are copied) -> safe for mpnet-768
    emb_all = np.load(os.path.join(args.vecdir, "user_vectors_all.npy"), mmap_mode="r")
    emb_nt = np.load(os.path.join(args.vecdir, "user_vectors_nontox.npy"), mmap_mode="r")
    df = df.merge(idx[["user_key", "n_nontox", "row"]], on="user_key", how="inner")
    edim = emb_all.shape[1]; ecols = [f"e{i}" for i in range(edim)]
    out("=" * 78); out("CLEAN LEAVE-TOXIC-OUT CONTROL (diffuse toxic trait vs tautology)"); out("=" * 78)

    rows = []
    for popname, mask in [("P1_nontox>=1", df["n_nontox"] >= 1),
                          ("P2_nontox>=1 & n_rev>=2", (df["n_nontox"] >= 1) & (df["n_reviews"] >= 2))]:
        sub = df[mask].reset_index(drop=True); y = sub["y_count"].to_numpy().astype(int)
        if y.sum() < 50: out(f"[{popname}] too few positives ({y.sum()}), skipping"); continue
        skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
        folds = np.full(len(sub), -1)
        for i, (_, te) in enumerate(skf.split(sub, y)): folds[te] = i
        Xprof = sub[PROFILE_CORE].reset_index(drop=True)
        Xall = pd.DataFrame(emb_all[sub["row"].to_numpy()], columns=ecols)
        Xlto = pd.DataFrame(emb_nt[sub["row"].to_numpy()], columns=ecols)  # no NaN here (n_nontox>=1)
        out(f"\n### {popname}: n={len(sub):,}  pos={int(y.sum()):,}  prev={y.mean()*100:.2f}%  baseline AP={y.mean():.4f}")
        for mdl, mk in MODELS.items():
            ap_a, roc_a, kap_a, oof_a = run_arm(Xprof, y, folds, prep_profile(), mk)
            ap_c, roc_c, kap_c, oof_c = run_arm(Xall, y, folds, prep_emb(edim), mk)
            ap_l, roc_l, kap_l, oof_l = run_arm(Xlto, y, folds, prep_emb(edim), mk)
            lo_la, hi_la = boot_delta(y, oof_l, oof_a)   # LTO vs agnostic  (>0 => diffuse trait)
            lo_cl, hi_cl = boot_delta(y, oof_c, oof_l)   # content vs LTO   (>0 => residual tautology)
            out(f"  {mdl:7s} AUC-PR: agnostic {ap_a:.3f} | LTO {ap_l:.3f} | content {ap_c:.3f}   "
                f"(ROC {roc_a:.3f}/{roc_l:.3f}/{roc_c:.3f})")
            out(f"          delta LTO-agnostic = {ap_l-ap_a:+.3f} CI[{lo_la:+.3f},{hi_la:+.3f}] "
                f"{'DIFFUSE TRAIT' if lo_la>0 else 'n.s.'}  |  content-LTO = {ap_c-ap_l:+.3f} CI[{lo_cl:+.3f},{hi_cl:+.3f}] "
                f"{'residual tautology' if lo_cl>0 else 'n.s.'}")
            rows.append(dict(pop=popname, model=mdl, n=len(sub), prev=round(float(y.mean()),4),
                             ap_agnostic=round(ap_a,4), ap_lto=round(ap_l,4), ap_content=round(ap_c,4),
                             roc_agnostic=round(roc_a,4), roc_lto=round(roc_l,4), roc_content=round(roc_c,4),
                             d_lto_agnostic=round(ap_l-ap_a,4), d_la_lo=round(lo_la,4), d_la_hi=round(hi_la,4),
                             d_content_lto=round(ap_c-ap_l,4), d_cl_lo=round(lo_cl,4), d_cl_hi=round(hi_cl,4)))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "lto_control.csv"), index=False)
    out(f"\n-> {os.path.join(args.out, 'lto_control.csv')}")
    with open(os.path.join(args.out, "lto_control_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
