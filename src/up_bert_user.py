#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_bert_user.py  -  User Predictor (end-to-end TRANSFORMER family): BERT fine-tuned on text

Unlike the 'content' arm (mean of SBERT embeddings + classical classifier), here a
transformer (DistilBERT by default) is fine-tuned END-TO-END, reading the concatenated TEXT
of the user's EN reviews to predict y_user. It is the genuine representative of the "deep
learning with transformers" family at the user level (it reads tokens, NOT the mean vector).

Honest caveat: BERT has a context limit (max_len tokens); for prolific users it sees only the
first reviews (truncation) -> it tends to see LESS than the mean-of-embeddings. We report this.
It is also partially tautological (it reads the toxic text that defines the label), just like
the content arm; the value lies in comparing the family's CAPACITY, not in claiming
non-circularity.

Imbalance (prev ~4%): training is balanced by OVERSAMPLING the positives or, with
--undersample, by undersampling the negatives; EVALUATION is at REAL prevalence
(stratified folds). k-fold CV -> mean+/-std. Summarizable per fold (saves json).

Usage (GPU; run from the repo root, KMP_DUPLICATE_LIB_OK=TRUE set by the script):
  python src/up_bert_user.py --model distilbert-base-uncased --cap 60000 --kfolds 5 --epochs 2
  python src/up_bert_user.py --label y_count --max-len 256 --batch 16
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, io, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

def best_thr(y, p):
    g = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 200))); bt, bf = 0.5, -1
    for t in g:
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, float(t)
    return bt

def load_user_text(features_dir, sub, max_chars):
    """Concatenate each subset user's EN reviews (inner join)."""
    import duckdb
    rr = os.path.join(features_dir, "reviews_resolved", "*.parquet").replace("\\", "/")
    con = duckdb.connect(); con.sql("PRAGMA threads=4"); con.sql("PRAGMA memory_limit='5GB'")
    con.register("sel", sub[["user_key"]])
    d = con.sql(f"""
        SELECT r.user_key, string_agg(left(r.review_text, 500), ' [SEP] '
                 ORDER BY r.review_id) AS utext
        FROM read_parquet('{rr}') r JOIN sel USING(user_key)
        WHERE r.review_lang='en' AND r.review_text IS NOT NULL
        GROUP BY r.user_key""").df()
    d["utext"] = d["utext"].str.slice(0, max_chars)
    return dict(zip(d["user_key"], d["utext"]))

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- end-to-end BERT transformer")
    ap.add_argument("--features-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "replication_bert"))
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--label", default="y_count")
    ap.add_argument("--cap", type=int, default=60000, help="stratified subsample (real prev) for GPU feasibility")
    ap.add_argument("--kfolds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Fraction of the fold's training set held out to choose the threshold without touching test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--undersample", action="store_true", help="Train on the balanced set via undersampling (discards the majority class)")
    ap.add_argument("--force", action="store_true", help="Recompute folds even when a cache exists")
    args = ap.parse_args()
    
    if args.undersample:
        args.out = args.out + "_undersampled"
        
    os.makedirs(args.out, exist_ok=True)
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_parquet(args.data, columns=["user_key", args.label, "n_reviews"])
    y_all = df[args.label].to_numpy().astype(int); prev = y_all.mean()
    # subsample stratified at real prevalence (per class, deterministic)
    rng = np.random.default_rng(args.seed)
    if len(df) > args.cap:
        npos = max(1, round(args.cap * prev)); nneg = args.cap - npos
        pos_idx = rng.choice(np.where(y_all == 1)[0], size=min(npos, int(y_all.sum())), replace=False)
        neg_idx = rng.choice(np.where(y_all == 0)[0], size=min(nneg, int((y_all == 0).sum())), replace=False)
        sub = df.iloc[np.sort(np.concatenate([pos_idx, neg_idx]))].reset_index(drop=True)
    else:
        sub = df.reset_index(drop=True)
    y = sub[args.label].to_numpy().astype(int)
    out("=" * 78); out(f"USER PREDICTOR -- end-to-end BERT | {args.model} | dev={dev}"); out("=" * 78)
    out(f"subsample: {len(sub):,} users  prev={y.mean()*100:.2f}%  (base {len(df):,}, prev {prev*100:.2f}%)")

    txt = load_user_text(args.features_dir, sub, max_chars=args.max_len * 8)
    sub["utext"] = sub["user_key"].map(lambda k: txt.get(k, ""))
    sub = sub[sub["utext"].str.len() > 0].reset_index(drop=True)
    y = sub[args.label].to_numpy().astype(int)
    texts = sub["utext"].tolist()
    out(f"with text: {len(sub):,} users | chars/user median={int(sub['utext'].str.len().median())}")

    tok = AutoTokenizer.from_pretrained(args.model)
    class DS(Dataset):
        def __init__(self, idxs): self.idxs = idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i):
            j = self.idxs[i]
            enc = tok(texts[j], truncation=True, max_length=args.max_len, padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, int(y[j])
    def collate(b):
        xs = {k: torch.stack([d[0][k] for d in b]) for k in b[0][0]}
        ys = torch.tensor([d[1] for d in b]); return xs, ys

    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
    per_fold = []
    for f, (tr, te) in enumerate(skf.split(sub, y)):
        fj = os.path.join(args.out, f"fold_{f}.json")
        pred_path = os.path.join(args.out, f"fold_{f}_preds.csv")
        if os.path.exists(fj) and os.path.exists(pred_path) and not args.force:
            cached = json.load(open(fj))
            if cached.get("threshold_source") == "validation":
                per_fold.append(cached); out(f"  fold {f}: cache"); continue
            out(f"  fold {f}: stale cache without internal validation; recomputing")
        
        tr_fit, va = train_test_split(
            tr,
            test_size=args.val_frac,
            stratify=y[tr],
            random_state=args.seed + f,
        )
        fold_rng = np.random.default_rng(args.seed * 1009 + f)
        tr_pos = tr_fit[y[tr_fit] == 1]; tr_neg = tr_fit[y[tr_fit] == 0]
        if args.undersample:
            # undersample the negatives in training to 1:1
            tr_neg_undersampled = fold_rng.choice(tr_neg, size=len(tr_pos), replace=False)
            tr_bal = np.concatenate([tr_neg_undersampled, tr_pos])
        else:
            # oversample the positives in training to an exact 1:1
            tr_pos_oversampled = fold_rng.choice(tr_pos, size=len(tr_neg), replace=True)
            tr_bal = np.concatenate([tr_neg, tr_pos_oversampled])
        
        fold_rng.shuffle(tr_bal)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        dl_gen = torch.Generator()
        dl_gen.manual_seed(args.seed + f)
        dl_tr = DataLoader(DS(tr_bal), batch_size=args.batch, shuffle=True, collate_fn=collate, generator=dl_gen)
        
        tb_writer = SummaryWriter(log_dir=os.path.join(args.out, "tensorboard", f"fold_{f}"))
        global_step = 0
        loss_log = []
        
        t0 = time.time(); model.train()
        for ep in range(args.epochs):
            for xs, ys in dl_tr:
                xs = {k: v.to(dev) for k, v in xs.items()}; ys = ys.to(dev)
                opt.zero_grad(); loss = model(**xs, labels=ys).loss; loss.backward(); opt.step()
                tb_writer.add_scalar("Loss/train", loss.item(), global_step)
                loss_log.append([time.time(), global_step, float(loss.item())])
                global_step += 1
            out(f"    fold {f} ep {ep+1}/{args.epochs} ({time.time()-t0:.0f}s)")
        
        tb_writer.close()
        json.dump(loss_log, open(os.path.join(args.out, f"fold_{f}_loss.json"), "w"))
        
        def predict(idxs):
            model.eval(); scores = np.zeros(len(idxs))
            dl = DataLoader(DS(idxs), batch_size=64, shuffle=False, collate_fn=collate)
            with torch.no_grad():
                k = 0
                for xs, _ in dl:
                    xs = {kk: v.to(dev) for kk, v in xs.items()}
                    p = torch.softmax(model(**xs).logits, dim=1)[:, 1].cpu().numpy()
                    scores[k:k+len(p)] = p; k += len(p)
            return scores

        # threshold chosen on the internal validation set; test stays at real prevalence
        va_scores = predict(va); yv = y[va]; thr = best_thr(yv, va_scores)
        scores = predict(te); yt = y[te]
        yp = (scores >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average=None, labels=[0, 1], zero_division=0)
        m = dict(fold=f, n_test=len(te), auc_pr=float(average_precision_score(yt, scores)),
                 auc_roc=float(roc_auc_score(yt, scores)),
                 f1_macro=float(f1_score(yt, yp, average="macro", zero_division=0)),
                 precision_pos=float(pr[1]), recall_pos=float(rc[1]), f1_pos=float(f1[1]),
                 kappa=float(cohen_kappa_score(yt, yp)),
                 threshold=float(thr), threshold_source="validation", n_val=len(va),
                 val_auc_pr=float(average_precision_score(yv, va_scores)),
                 val_auc_roc=float(roc_auc_score(yv, va_scores)))
        pd.DataFrame({"user_key": sub["user_key"].iloc[te].to_numpy(), "y": yt, "score": scores,
                      "fold": f, "threshold": thr}).to_csv(pred_path, index=False)
        json.dump(m, open(fj, "w")); per_fold.append(m)
        out(f"  fold {f}: AUC-PR {m['auc_pr']:.3f} ROC {m['auc_roc']:.3f} F1m {m['f1_macro']:.3f} kappa {m['kappa']:.3f}")

    agg = {k: float(np.mean([r[k] for r in per_fold])) for k in ["auc_pr","auc_roc","f1_macro","kappa"]}
    sd = {k: float(np.std([r[k] for r in per_fold], ddof=1)) for k in ["auc_pr","auc_roc","f1_macro"]}
    out(f"\n=== {args.model} ({args.label}) AGGREGATE  AUC-PR {agg['auc_pr']:.3f}±{sd['auc_pr']:.3f}  "
        f"ROC {agg['auc_roc']:.3f}±{sd['auc_roc']:.3f}  F1m {agg['f1_macro']:.3f}  kappa {agg['kappa']:.3f}")
    out(f"  baseline AP (prev) = {y.mean():.4f}  | compare against the content arm (mean-embedding)")
    pd.DataFrame(per_fold).assign(model=args.model, label=args.label, agg_auc_pr=agg["auc_pr"],
        agg_auc_pr_sd=sd["auc_pr"], agg_auc_roc=agg["auc_roc"]).to_csv(
        os.path.join(args.out, "bert_user_metrics.csv"), index=False)
    with open(os.path.join(args.out, "bert_user_report.txt"), "w", encoding="utf-8") as ff:
        ff.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
