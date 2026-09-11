#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_llm_narrative.py  -  User Predictor: zero-shot LLM narratives (explains the WHY)

USER-level EXPLAINABILITY layer (consistent with the pivot; this is NOT the review-level
explainability that went out of scope). The LLM receives the COMPLETE LIST of the user's reviews
(text; NEVER the embedding -- the embedding is not interpretable by the LLM) and, zero-shot, produces:
  - label (toxic/non-toxic) + score 0-100
  - a NARRATIVE explaining WHY it classified this way, citing specific reviews
  - the most relevant evidence review (quote) + index of the worst review

CURATED sample for the qualitative section: if classifier LLM predictions are available
(results/lente4_llm/llm_user_preds_*.csv), select cases by type of correct/incorrect outcome
(TP/FP/FN/TN); otherwise, stratify by y_count. Prefers users with SEVERAL reviews (richer
narrative). Low cost (a few dozen users).

Outputs (results/lente4_llm/narratives/): narratives.jsonl + narratives.csv + narratives.md
Usage (Ollama up): python src/up_llm_narrative.py --backend ollama --model qwen2.5:7b --per-group 8
"""
from __future__ import annotations
import os, re, json, time, argparse, io
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

def call_ollama(model, system, user, host, timeout=240):
    import requests
    payload = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout); r.raise_for_status()
    return r.json()["message"]["content"]

def call_claude(client, model, system, user):
    msg = client.messages.create(model=model, max_tokens=512, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

def parse_json(text):
    if not text: return {}
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return {}

SYSTEM_NARRATIVE = (
"You are an expert content moderator. You are given ALL the Steam game reviews written by a SINGLE "
"user. Decide whether the USER is TOXIC (authored AT LEAST ONE toxic review) and EXPLAIN your reasoning.\n"
"A review is TOXIC if it contains rude, hateful, or abusive language directed at PEOPLE (other users, "
"developers, groups): insults, slurs, identity attacks, threats, harassment.\n"
"Be specific: cite the actual review text that drives your decision. Judge ONLY from the reviews shown; "
"you are NOT given the game's title, genre, tags, or any other context.")
ASK_NARRATIVE = ('Respond ONLY a JSON object: {"label":"toxic"|"nontoxic", "user_toxicity_score":<0-100>, '
                 '"reasoning":"<2-4 sentences explaining WHY, citing specific reviews>", '
                 '"evidence_quote":"<the single most decisive review snippet, verbatim>", '
                 '"worst_review":<int index or -1>}. No text outside the JSON.')

def build_prompt(reviews, cap, rng):
    n = len(reviews); shown = reviews; extra = 0
    if n > cap:
        sel = np.sort(rng.choice(n, size=cap, replace=False)); shown = [reviews[i] for i in sel]; extra = n - cap
    parts = [f"User's reviews ({n} total{', showing ' + str(cap) + ' at random' if extra else ''}):"]
    for i, t in enumerate(shown): parts.append(f'[{i}] """{str(t)[:1500]}"""')
    if extra: parts.append(f"... and {extra} more not shown.")
    parts.append(ASK_NARRATIVE)
    return "\n".join(parts)

def curate(modeling_pq, preds_csv, per_group, seed):
    df = pd.read_parquet(modeling_pq, columns=["user_key", "y_count", "n_reviews"])
    df["user_key"] = df["user_key"].astype(str)            # steam_id is a string in the corpus; CSV reads it as int64
    rng = np.random.default_rng(seed)
    groups = {}
    if preds_csv and os.path.exists(preds_csv):
        praw = pd.read_csv(preds_csv)
        pr = praw[["user_key", "_pred"]].copy() if "_pred" in praw.columns else None
        if pr is not None:
            pr["user_key"] = pr["user_key"].astype(str)    # match the dtype for the merge
            m = df.merge(pr, on="user_key", how="inner")
            m["cell"] = np.where((m.y_count == 1) & (m._pred == 1), "TP (toxic, correct)",
                        np.where((m.y_count == 0) & (m._pred == 1), "FP (false positive)",
                        np.where((m.y_count == 1) & (m._pred == 0), "FN (toxic, missed)", "TN (non-toxic, correct)")))
            for cell, g in m.groupby("cell"):
                g = g.sort_values("n_reviews", ascending=False)  # prefer multi-review users
                groups[cell] = g.head(per_group)
            return pd.concat(groups.values()).reset_index(drop=True)
    # fallback: stratify by y_count, preferring multi-review users
    for yv, name in [(1, "toxic"), (0, "non-toxic")]:
        g = df[df.y_count == yv].sort_values("n_reviews", ascending=False).head(per_group * 2)
        groups[name] = g.sample(min(per_group, len(g)), random_state=seed)
        groups[name]["cell"] = name
    return pd.concat(groups.values()).reset_index(drop=True)

def fetch_reviews(features_dir, sub):
    import duckdb
    rr = os.path.join(features_dir, "reviews_resolved", "*.parquet").replace("\\", "/")
    con = duckdb.connect(); con.sql("PRAGMA threads=4"); con.sql("PRAGMA memory_limit='4GB'")
    con.register("sel", sub[["user_key"]])
    d = con.sql(f"""SELECT r.user_key, r.review_text FROM read_parquet('{rr}') r JOIN sel USING(user_key)
                    WHERE r.review_lang='en' AND r.review_text IS NOT NULL ORDER BY r.user_key, r.review_id""").df()
    return d.groupby("user_key")["review_text"].apply(list).to_dict()

def main():
    ap = argparse.ArgumentParser(description="User Predictor -- zero-shot narratives (explains the why)")
    ap.add_argument("--features-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--modeling", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap.add_argument("--preds", default=os.path.join(ROOT, "results", "lente4_llm", "llm_user_preds_qwen2.5_7b.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lente4_llm", "narratives"))
    ap.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--per-group", type=int, default=8, help="users per cell (TP/FP/FN/TN or toxic/non-toxic)")
    ap.add_argument("--max-reviews", type=int, default=300, help="reviews shown per user (COMPLETE list for the LLM; 300 covers the tail)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(args.seed)

    if args.backend == "ollama":
        try:
            import requests; requests.get(f"{args.ollama_host}/api/tags", timeout=5).raise_for_status()
        except Exception as e:
            out(f"[ABORT] Ollama unreachable ({e}). Run: & D:/Ollama/ollama.exe serve"); return
    client = None
    if args.backend == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"): out("[ABORT] ANTHROPIC_API_KEY missing."); return
        import anthropic; client = anthropic.Anthropic()

    out("=" * 78); out(f"USER PREDICTOR -- zero-shot narratives | {args.model} ({args.backend})"); out("=" * 78)
    sub = curate(args.modeling, args.preds, args.per_group, args.seed)
    revs = fetch_reviews(args.features_dir, sub)
    sub["reviews"] = sub["user_key"].map(lambda k: revs.get(k, []))
    sub = sub[sub["reviews"].map(len) > 0].reset_index(drop=True)
    out(f"curated sample: {len(sub)} users | cells: {sub['cell'].value_counts().to_dict()}")

    rows = []
    for i, r in sub.iterrows():
        prompt = build_prompt(r["reviews"], args.max_reviews, rng); raw = ""
        for att in range(3):
            try:
                raw = (call_ollama(args.model, SYSTEM_NARRATIVE, prompt, args.ollama_host)
                       if args.backend == "ollama" else call_claude(client, args.model, SYSTEM_NARRATIVE, prompt))
                if parse_json(raw): break
            except Exception as e:
                raw = f"ERR:{e}"; time.sleep(1.5 * (att + 1))
        j = parse_json(raw)
        rows.append(dict(user_key=r["user_key"], cell=r["cell"], y_count=int(r["y_count"]), n_reviews=int(r["n_reviews"]),
                         pred_label=j.get("label", ""), score=j.get("user_toxicity_score", ""),
                         reasoning=j.get("reasoning", ""), evidence=j.get("evidence_quote", ""),
                         worst_review=j.get("worst_review", -1)))
        if (i + 1) % 5 == 0: out(f"  {i+1}/{len(sub)}")
    df = pd.DataFrame(rows)
    df.to_json(os.path.join(args.out, "narratives.jsonl"), orient="records", lines=True, force_ascii=False)
    df.to_csv(os.path.join(args.out, "narratives.csv"), index=False)
    # human-readable markdown for the paper
    with open(os.path.join(args.out, "narratives.md"), "w", encoding="utf-8") as f:
        f.write("# Zero-shot narratives (explanation of the user classification)\n\n")
        for _, r in df.iterrows():
            f.write(f"## {r['cell']} — y_count={r['y_count']} | n_reviews={r['n_reviews']} | LLM={r['pred_label']} ({r['score']})\n")
            f.write(f"**Why:** {r['reasoning']}\n\n> {r['evidence']}\n\n---\n")
    out(f"\n-> {args.out}/narratives.{{jsonl,csv,md}}  ({len(df)} narratives)")
    with open(os.path.join(args.out, "narrative_report.txt"), "w", encoding="utf-8") as f: f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
