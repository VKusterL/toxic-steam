#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_llm_user.py  -  User Predictor (LLM variant): classifies the USER from ALL of their reviews

The embedding vector is not used here (advisor's constraint): ALL of the user's EN reviews are
sent to an open-source LLM to decide whether the user is toxic. This replicates the design of
sprint4_llm_classify.py at the USER level. It evaluates on the SAME users as the 10% slice
(sample stratified at the REAL PREVALENCE ~4%), in blocks -> mean+/-std (the LLM is
deterministic/temp=0, so the variance comes from the blocks + bootstrap, not from retraining).

A mandatory INNER JOIN reviews_resolved -> user_features (via modeling_users_10pct) is done BEFORE
packing (reviews_resolved has 9.18M users + a bot with ~16M reviews; the labeled slice is
the valid population). Reviews per user: median 1, p99~15; deterministic cap (RANDOM selection,
NOT ordered by score, so the label signal does not leak) with "+K reviews not shown".

Label: count (y_count = user with >=1 toxic review). We ask for label + a 0-100 score (-> AUC).

Outputs (results/lente4_llm/):
  llm_user_metrics.csv   (model x block + aggregate mean+/-std + per n_reviews bucket)
  llm_user_preds_<modelo>.csv  (user_key, y, pred, score, worst_review, raw)

Usage (Ollama running: `& D:/Ollama/ollama.exe serve`):
  python src/up_llm_user.py --models llama3.1:8b --blocks 1 --block-size 500
  python src/up_llm_user.py --models llama3.1:8b,qwen2.5:7b --blocks 5 --block-size 2000
"""
from __future__ import annotations
import os, re, json, time, argparse, io
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             cohen_kappa_score, precision_recall_fscore_support)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

# callers/parsers (self-contained; same contract as sprint4_llm_classify.py, archived)
def call_ollama(model, system, user, host, timeout=180):
    import requests
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout); r.raise_for_status()
    return r.json()["message"]["content"]

def call_claude(client, model, system, user):
    msg = client.messages.create(model=model, max_tokens=128, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

def call_openai(client, model, system, user):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=128,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

def parse_label(text):
    if not text: return None
    m = re.search(r'"label"\s*:\s*"?(toxic|nontoxic|non-toxic)"?', text, re.I)
    if m: return 0 if m.group(1).lower().startswith("non") else 1
    low = text.strip().lower()
    if low.startswith("toxic") or '"toxic"' in low: return 1
    if "nontoxic" in low or "non-toxic" in low or low.startswith("not"): return 0
    return None

def parse_score(text, label):
    if text:
        m = re.search(r'"?user_toxicity_score"?\s*:\s*([0-9]{1,3}(?:\.[0-9]+)?)', text, re.I)
        if m:
            try: return max(0.0, min(1.0, float(m.group(1)) / 100.0))
            except Exception: pass
    return float(label) if label is not None else 0.0

SYSTEM_USER = (
"You are an expert content moderator. You are given ALL the Steam game reviews written by a "
"SINGLE user. Decide whether this USER is TOXIC, i.e. whether the user authored AT LEAST ONE "
"toxic review.\n"
"A review is TOXIC if it contains rude, disrespectful, hateful, or abusive language directed at "
"PEOPLE (other users, developers, groups): insults, slurs, identity attacks, threats, or harassment.\n"
"The user is TOXIC if ANY single review crosses the line, even if most reviews are clean.\n"
"Judge ONLY from the reviews below; you are NOT given the game's title, genre, tags, or any other context.\n")
ASK_USER = ('Respond with ONLY a JSON object: {"label": "toxic"|"nontoxic", '
            '"user_toxicity_score": <int 0-100>, "worst_review": <int index or -1>}. '
            'user_toxicity_score = how confident a human moderator would be that this user wrote >=1 abusive '
            'review toward PEOPLE (0=clearly never, 100=clearly yes). worst_review = index of the most toxic '
            'review (or -1 if none). No other text.')

def build_user_prompt(reviews, cap, rng):
    n = len(reviews)
    shown = reviews
    extra = 0
    if n > cap:
        sel = np.sort(rng.choice(n, size=cap, replace=False))   # RANDOM (not by score: anti-leakage)
        shown = [reviews[i] for i in sel]; extra = n - cap
    parts = [f"User's reviews ({n} total{', showing ' + str(cap) + ' at random' if extra else ''}):"]
    for i, t in enumerate(shown):
        parts.append(f'[{i}] """{str(t)[:1500]}"""')
    if extra: parts.append(f"... and {extra} more reviews not shown.")
    parts.append(ASK_USER)
    return "\n".join(parts)

def parse_worst(text):
    if not text: return -1
    m = re.search(r'"?worst_review"?\s*:\s*(-?\d{1,4})', text)
    return int(m.group(1)) if m else -1

def load_user_blocks(features_dir, modeling_pq, n_blocks, block_size, seed, label="y_count"):
    import duckdb
    df = pd.read_parquet(modeling_pq, columns=["user_key", label, "n_reviews"])
    prev = df[label].mean()
    rng = np.random.default_rng(seed)
    pos = df[df[label] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    neg = df[df[label] == 0].sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    npos_blk = max(1, round(block_size * prev)); nneg_blk = block_size - npos_blk
    blocks = []
    for b in range(n_blocks):
        pp = pos.iloc[b * npos_blk:(b + 1) * npos_blk]; nn = neg.iloc[b * nneg_blk:(b + 1) * nneg_blk]
        blk = pd.concat([pp, nn]).sample(frac=1, random_state=seed + b).reset_index(drop=True)
        blk["block"] = b; blocks.append(blk)
    sel = pd.concat(blocks).reset_index(drop=True)
    # fetch ALL EN reviews of the selected users (inner join)
    rr = os.path.join(features_dir, "reviews_resolved", "*.parquet").replace("\\", "/")
    con = duckdb.connect(); con.sql("PRAGMA threads=4"); con.sql("PRAGMA memory_limit='5GB'")
    con.register("sel", sel[["user_key"]])
    revs = con.sql(f"""SELECT r.user_key, r.review_text, r.y_review
                       FROM read_parquet('{rr}') r JOIN sel USING(user_key)
                       WHERE r.review_lang='en' AND r.review_text IS NOT NULL
                       ORDER BY r.user_key, r.review_id""").df()
    by_user = revs.groupby("user_key")["review_text"].apply(list).to_dict()
    sel["reviews"] = sel["user_key"].map(lambda k: by_user.get(k, []))
    sel = sel[sel["reviews"].map(len) > 0].reset_index(drop=True)
    return sel, prev

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- LLM variant (user with all their reviews)")
    ap.add_argument("--features-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--modeling", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lente4_llm"))
    ap.add_argument("--backend", choices=["ollama", "claude", "openai"], default="ollama")
    ap.add_argument("--models", default="qwen2.5:7b", help="csv of backend models (Qwen2.5-7B: best toxicity classifier in the project -- gold AUC-ROC 0.807 > Llama 0.769 -- and lighter for the GTX1070)")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--blocks", type=int, default=5, help="stratified blocks for mean+/-std (5 to match the mlp/svc 5-fold)")
    ap.add_argument("--block-size", type=int, default=200, help="200 x 5 blocks = 1000 users")
    ap.add_argument("--cap-reviews", type=int, default=300, help="max reviews shown per user (300 covers the tail; FULL list for ~everyone; Qwen's 32k context fits it)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    out("=" * 78); out("USER PREDICTOR -- LLM variant (user level)"); out("=" * 78)
    sel, prev = load_user_blocks(args.features_dir, args.modeling, args.blocks, args.block_size, args.seed)
    out(f"sample: {len(sel):,} users in {args.blocks} block(s) of ~{args.block_size}  prev={prev*100:.2f}%")
    out(f"reviews/user: median={int(sel['reviews'].map(len).median())} max={int(sel['reviews'].map(len).max())}")

    if args.backend == "ollama":
        try:
            import requests; requests.get(f"{args.ollama_host}/api/tags", timeout=5).raise_for_status()
        except Exception as e:
            out(f"[ABORT] Ollama not reachable at {args.ollama_host} ({e}). Run: & D:/Ollama/ollama.exe serve"); return
    client = None
    if args.backend == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"): out("[ABORT] ANTHROPIC_API_KEY missing."); return
        import anthropic; client = anthropic.Anthropic()
    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"): out("[ABORT] OPENAI_API_KEY missing."); return
        import openai; client = openai.OpenAI()

    summary = []
    for model in args.models.split(","):
        model = model.strip()
        out(f"\n### model={model} ({args.backend})")
        preds = np.zeros(len(sel), int); scores = np.zeros(len(sel)); worst = np.full(len(sel), -1); raws = [""] * len(sel)
        t0 = time.time()
        for i, row in sel.iterrows():
            prompt = build_user_prompt(row["reviews"], args.cap_reviews, rng)
            lab, raw = None, ""
            for att in range(3):
                try:
                    raw = (call_ollama(model, SYSTEM_USER, prompt, args.ollama_host) if args.backend == "ollama"
                           else call_openai(client, model, SYSTEM_USER, prompt) if args.backend == "openai"
                           else call_claude(client, model, SYSTEM_USER, prompt))
                    lab = parse_label(raw)
                    if lab is not None: break
                except Exception as e:
                    raw = f"ERR:{e}"; time.sleep(1.5 * (att + 1))
            preds[i] = lab if lab is not None else 0
            scores[i] = parse_score(raw, lab); worst[i] = parse_worst(raw); raws[i] = raw[:300]
            if (i + 1) % 25 == 0: out(f"    {model}: {i+1}/{len(sel)}  ({(time.time()-t0)/(i+1):.1f}s/user)")
        sel["_pred"], sel["_score"], sel["_worst"], sel["_raw"] = preds, scores, worst, raws
        y = sel["y_count"].to_numpy().astype(int)
        # per-block aggregate -> mean+/-std
        per_blk = []
        for b in range(args.blocks):
            mb = sel["block"].to_numpy() == b
            if mb.sum() == 0 or sel["y_count"][mb].nunique() < 2: continue
            per_blk.append(dict(auc_pr=average_precision_score(y[mb], scores[mb]),
                                auc_roc=roc_auc_score(y[mb], scores[mb]),
                                f1_macro=f1_score(y[mb], preds[mb], average="macro", zero_division=0),
                                kappa=cohen_kappa_score(y[mb], preds[mb])))
        agg = {k: float(np.mean([r[k] for r in per_blk])) for k in ["auc_pr","auc_roc","f1_macro","kappa"]} if per_blk else {}
        sd = {k: float(np.std([r[k] for r in per_blk], ddof=1)) if len(per_blk) > 1 else 0.0 for k in agg}
        pr, rc, _, _ = precision_recall_fscore_support(y, preds, average=None, labels=[0, 1], zero_division=0)
        out(f"  AGGREGATE  AUC-PR {agg.get('auc_pr',float('nan')):.3f}±{sd.get('auc_pr',0):.3f}  "
            f"ROC {agg.get('auc_roc',float('nan')):.3f}  F1m {agg.get('f1_macro',float('nan')):.3f}  "
            f"kappa {agg.get('kappa',float('nan')):.3f}  P={pr[1]:.2f} R={rc[1]:.2f}  ({time.time()-t0:.0f}s)")
        # per n_reviews bucket
        nb = sel["reviews"].map(len).to_numpy()
        for lo, hi, name in [(1,1,"1"),(2,4,"2-4"),(5,9,"5-9"),(10,10**9,">=10")]:
            mb = (nb >= lo) & (nb <= hi)
            if mb.sum() >= 20 and sel["y_count"][mb].nunique() == 2:
                out(f"    bucket n_reviews {name:5s}: n={int(mb.sum())} AUC-PR={average_precision_score(y[mb],scores[mb]):.3f} "
                    f"ROC={roc_auc_score(y[mb],scores[mb]):.3f}")
        summary.append(dict(model=model, backend=args.backend, n=len(sel), prev=round(float(prev),4),
                            **{k: round(agg.get(k,float('nan')),4) for k in ["auc_pr","auc_roc","f1_macro","kappa"]},
                            **{f"{k}_sd": round(sd.get(k,0),4) for k in ["auc_pr","auc_roc","f1_macro"]},
                            precision_tox=round(float(pr[1]),4), recall_tox=round(float(rc[1]),4)))
        sel[["user_key","block","y_count","n_reviews","_pred","_score","_worst","_raw"]].to_csv(
            os.path.join(args.out, f"llm_user_preds_{model.replace(':','_').replace('/','_')}.csv"), index=False)
        pd.DataFrame(summary).to_csv(os.path.join(args.out, "llm_user_metrics.csv"), index=False)
    out("\n-> results/lente4_llm/llm_user_metrics.csv")
    with open(os.path.join(args.out, "llm_user_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
