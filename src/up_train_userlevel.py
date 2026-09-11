#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_train_userlevel.py  -  User Predictor: ablation matrix

Replicates the USER-level classification study with the content / content-agnostic /
combined ablation, under 5-fold CV (fixed folds shared across models -> paired tests),
with ALL preprocessing and the threshold ESTIMATED WITHIN THE FOLD (no leakage), AUC-PR
as the anchor metric and mean+/-std for every metric.

ARMS (feature arms):
  content_agnostic       PROFILE_ONLY (profile only, NO ban)          <- HEADLINE (non-circular)
  content_agnostic+ban   PROFILE_ONLY + {has_ban, ban_recency_days}   (ban is endogenous to the label)
  content                mean of MiniLM embeddings (renorm)           <- UPPER-BOUND ablation
  content_leavetoxicout  mean of ONLY the NON-toxic reviews           <- tautology CONTROL
  combined               PROFILE_ONLY + content

LABELS (SEPARATE populations, never merged):
  count  y_count (>=1 toxic)    over the full 10% base (~620k, prev ~4%)
  rate   y_rate  (>=5% toxic)   over the subpopulation n_reviews>=5 (~69k, prev ~12.8%)

Reuses the best_threshold primitive (macro-F1 on the training set). Outputs in results/replication/:
  metrics_cv.csv      (1 row per label x arm x model x fold)
  metrics_agg.csv     (mean+/-std per label x arm x model)
  oof/<label>_<arm>_<model>.parquet  (out-of-fold predictions per user; for significance + buckets)

Usage (run from the repo root):
  python src/up_train_userlevel.py
  python src/up_train_userlevel.py --label count --models logreg,linsvc,rf,histgb,xgb
  python src/up_train_userlevel.py --heavy mlp,svc --cap-heavy 30000
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, io, time, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score, accuracy_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

PROFILE_CORE = ["profile_level", "profile_level_missing", "library_size", "library_size_missing",
                "awards", "insignias", "screenshots", "workshop_items", "guides", "arts", "groups",
                "profile_visibility", "country", "country_is_brazil"]
PLATFORM_ENF = ["has_ban", "ban_recency_days"]
CAT_COLS = ["country"]

ARMS = {
    "content_agnostic":      dict(profile=PROFILE_CORE,                emb=None),
    "content_agnostic+ban":  dict(profile=PROFILE_CORE + PLATFORM_ENF, emb=None),
    "content":               dict(profile=[],                          emb="all"),
    "content_leavetoxicout": dict(profile=[],                          emb="nontox"),
    "combined":              dict(profile=PROFILE_CORE,                emb="all"),
}
LABELS = {
    "count": dict(target="y_count", fold="fold_y_count", subset=None),
    "rate":  dict(target="y_rate",  fold="fold_y_rate",  subset="rate_pop"),
}

def best_threshold(y, p, n_grid=200):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.999, n_grid)))
    bt, bf = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f1 > bf: bf, bt = f1, float(t)
    return bt

def metrics_row(y, p, thr, prevalence):
    yp = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average=None, labels=[0, 1], zero_division=0)
    return dict(
        auc_pr=float(average_precision_score(y, p)),
        auc_roc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        f1_macro=float(f1_score(y, yp, average="macro", zero_division=0)),
        precision_pos=float(pr[1]), recall_pos=float(rc[1]), f1_pos=float(f1[1]),
        kappa=float(cohen_kappa_score(y, yp)), accuracy=float(accuracy_score(y, yp)),
        baseline_ap=float(prevalence), threshold=round(float(thr), 5),
    )

def build_models(which, heavy_set, nj=4):
    # nj = cores per model (NOT -1: leaves headroom for the OS, avoids freezing the machine + reduces heat)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    cat = {
        # --- LINEAR ---
        "logreg": lambda: LogisticRegression(max_iter=1000, n_jobs=nj),
        "linsvc": lambda: LinearSVC(),                                   # Pinto core; decision_function
        # --- KERNEL ---
        "svc":    lambda: SVC(),                                         # Pinto core (RBF; heavy/capped)
        # --- BAGGING ---
        "rf":     lambda: RandomForestClassifier(n_jobs=nj, random_state=42),
        "extratrees": lambda: ExtraTreesClassifier(n_jobs=nj, random_state=42),
        # --- BOOSTING ---
        "histgb": lambda: HistGradientBoostingClassifier(random_state=42),  # scalable ~ Pinto "XGB"
        # --- DEEP (NN) ---
        "mlp":    lambda: MLPClassifier(random_state=42),                # Pinto core (heavy)
    }
    try:
        from xgboost import XGBClassifier
        cat["xgb"] = lambda: XGBClassifier(tree_method="hist", eval_metric="logloss",
                                           n_jobs=nj, random_state=42)   # real XGBoost (An et al.)
    except Exception: pass
    try:
        from lightgbm import LGBMClassifier
        cat["lgbm"] = lambda: LGBMClassifier(n_jobs=nj, random_state=42, verbose=-1)
    except Exception: pass
    try:
        from catboost import CatBoostClassifier
        cat["catboost"] = lambda: CatBoostClassifier(verbose=0, random_state=42, thread_count=nj)
    except Exception: pass
    sel = [m for m in which if m in cat]
    return {m: cat[m] for m in sel}, set(heavy_set)

def scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    s = clf.decision_function(X)                                         # LinearSVC/SVC: margin
    return s

def make_pipeline(arm_cols, model_factory):
    num = [c for c in arm_cols if c not in CAT_COLS and not c.startswith("e")]  # numeric profile
    emb = [c for c in arm_cols if c.startswith("e")]                            # embedding dims
    cat = [c for c in arm_cols if c in CAT_COLS]
    tfs = []
    if num: tfs.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if emb: tfs.append(("emb", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), emb))
    if cat: tfs.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15,
                                             sparse_output=False), cat))
    prep = ColumnTransformer(tfs, remainder="drop")
    return Pipeline([("prep", prep), ("clf", model_factory())])

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- ablation matrix (An/Pinto)")
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap.add_argument("--vecdir", default=os.path.join(ROOT, "results", "user_predictor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "replication"))
    ap.add_argument("--label", default="count,rate")
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    ap.add_argument("--models", default="logreg,linsvc,rf,histgb,xgb,mlp")
    ap.add_argument("--heavy", default="mlp,svc", help="models trained on a CAPPED training set (cost)")
    ap.add_argument("--cap-heavy", type=int, default=30000, help="max training rows for heavy models")
    ap.add_argument("--n-jobs", type=int, default=4, help="cores per model (NOT -1: leaves headroom for the OS, avoids freezing)")
    ap.add_argument("--kfolds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(os.path.join(args.out, "oof"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- tabular data + vectors (INNER JOIN on user_key: guarantees the labeled population) ---
    df = pd.read_parquet(args.data)
    idx = pd.read_parquet(os.path.join(args.vecdir, "user_vectors_index.parquet"))
    # mmap_mode='r' does NOT load the 918k x dim into RAM at once (only the indexed rows are copied);
    # emb_nt is loaded only if a leave-toxic-out arm is requested (saves ~half the RAM on mpnet-768)
    emb_all = np.load(os.path.join(args.vecdir, "user_vectors_all.npy"), mmap_mode="r")
    emb_nt = (np.load(os.path.join(args.vecdir, "user_vectors_nontox.npy"), mmap_mode="r")
              if "content_leavetoxicout" in args.arms else None)
    row_of = dict(zip(idx["user_key"].to_numpy(), idx["row"].to_numpy()))
    df["erow"] = df["user_key"].map(row_of)
    n0 = len(df); df = df[df["erow"].notna()].copy(); df["erow"] = df["erow"].astype(int)
    out("=" * 78); out("USER PREDICTOR -- An/Pinto ablation matrix"); out("=" * 78)
    out(f"users: {len(df):,} (inner-join with embeddings; -{n0-len(df):,} without a vector)")
    EMB = {"all": emb_all, "nontox": emb_nt}
    edim = emb_all.shape[1]; ecols = [f"e{i}" for i in range(edim)]

    models, heavy = build_models(args.models.split(","), args.heavy.split(","), args.n_jobs)
    arms = [a for a in args.arms.split(",") if a in ARMS]
    out(f"models: {list(models)} | heavy(capped@{args.cap_heavy}): {sorted(heavy & set(models))}")
    out(f"arms: {arms}\n")

    csv_path = os.path.join(args.out, "metrics_cv.csv")
    rows = []
    for lab in args.label.split(","):
        L = LABELS[lab]; sub = df if L["subset"] is None else df[df[L["subset"]] == 1].copy()
        y = sub[L["target"]].to_numpy().astype(int)
        # recompute folds via StratifiedKFold(kfolds) on the modeling population -> --kfolds works for any value
        # (the pre-computed column is 10-fold; without this, --kfolds 5 would test only 5 of the 10 partitions = 50% of users)
        _skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
        foldv = np.full(len(sub), -1, dtype=int)
        for _fi, (_, _te) in enumerate(_skf.split(sub, y)): foldv[_te] = _fi
        prev = y.mean()
        out(f"### LABEL={lab}  n={len(sub):,}  prev={prev*100:.3f}%  (baseline AP={prev:.4f})")
        for arm in arms:
            spec = ARMS[arm]
            # assemble X (DataFrame) with only the columns of this arm
            X = sub[spec["profile"]].copy() if spec["profile"] else pd.DataFrame(index=sub.index)
            if spec["emb"] is not None:
                E = EMB[spec["emb"]][sub["erow"].to_numpy()]
                X = pd.concat([X.reset_index(drop=True), pd.DataFrame(E, columns=ecols)], axis=1)
            else:
                X = X.reset_index(drop=True)
            arm_cols = list(X.columns)
            for mname, mk in models.items():
                tag = f"{lab}_{arm}_{mname}"
                oof_path = os.path.join(args.out, "oof", f"{tag}.parquet")
                if os.path.exists(oof_path):
                    out(f"  [skip] {tag} (oof exists)");
                    # could reload metrics if already in the csv? keep it simple: skip
                    continue
                oof_score = np.full(len(sub), np.nan); per_fold = []
                t0 = time.time()
                for f in range(args.kfolds):
                    tr = np.where(foldv != f)[0]; te = np.where(foldv == f)[0]
                    if mname in heavy and len(tr) > args.cap_heavy:           # subsample training set (cost)
                        tr = rng.choice(tr, size=args.cap_heavy, replace=False)
                    pipe = make_pipeline(arm_cols, mk)
                    pipe.fit(X.iloc[tr], y[tr])
                    s_tr = scores(pipe, X.iloc[tr]); s_te = scores(pipe, X.iloc[te])
                    thr = best_threshold(y[tr], s_tr)
                    m = metrics_row(y[te], s_te, thr, prev)
                    m.update(label=lab, arm=arm, model=mname, fold=f, n_test=len(te))
                    per_fold.append(m); oof_score[te] = s_te; rows.append(m)
                agg = {k: np.mean([r[k] for r in per_fold]) for k in ["auc_pr","auc_roc","f1_macro","kappa","recall_pos","precision_pos"]}
                sd = {k: np.std([r[k] for r in per_fold], ddof=1) for k in ["auc_pr","auc_roc","f1_macro"]}
                out(f"  {arm:22s} {mname:7s}  AUC-PR {agg['auc_pr']:.3f}±{sd['auc_pr']:.3f}  "
                    f"ROC {agg['auc_roc']:.3f}  F1m {agg['f1_macro']:.3f}±{sd['f1_macro']:.3f}  "
                    f"kappa {agg['kappa']:.3f}  ({time.time()-t0:.0f}s)")
                pd.DataFrame({"user_key": sub["user_key"].to_numpy(), "y": y,
                              "score": oof_score, "n_reviews": sub["n_reviews"].to_numpy(),
                              "label": lab, "arm": arm, "model": mname}).to_parquet(oof_path, index=False)
                pd.DataFrame(rows).to_csv(csv_path, index=False)   # incremental checkpoint
            del X
    # final aggregation
    if rows:
        d = pd.DataFrame(rows)
        agg = d.groupby(["label","arm","model"]).agg(
            auc_pr_mean=("auc_pr","mean"), auc_pr_sd=("auc_pr","std"),
            auc_roc_mean=("auc_roc","mean"), auc_roc_sd=("auc_roc","std"),
            f1_macro_mean=("f1_macro","mean"), f1_macro_sd=("f1_macro","std"),
            kappa_mean=("kappa","mean"), recall_pos_mean=("recall_pos","mean"),
            precision_pos_mean=("precision_pos","mean"), baseline_ap=("baseline_ap","first"),
        ).round(4).reset_index()
        agg.to_csv(os.path.join(args.out, "metrics_agg.csv"), index=False)
        out("\n--- AGGREGATE (mean+/-std across folds) ---")
        out(agg.to_string(index=False))
    with open(os.path.join(args.out, "train_userlevel_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
