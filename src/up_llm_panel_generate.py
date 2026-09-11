#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_llm_panel_generate.py  -  Narrative panel: GENERATION with 4 models

Each of the 4 models (Opus, GPT-4o, Qwen, Llama) generates, zero-shot, a narrative explaining
WHY the user was classified, reading the FULL LIST of the user's reviews (never the
embedding). Same curated sample (TP/FP/FN/TN) and same prompt for all -> comparable.

Output: results/lente4_llm/panel/narratives.jsonl  (1 line per (user, generator))
Usage: python src/up_llm_panel_generate.py --generators opus,gpt4o,qwen,llama --per-group 6
       (only runs the models whose backend is available: Ollama up / keys set)
"""
from __future__ import annotations
import os, json, time, argparse, importlib.util
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fn, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
nb = _load("up_llm_narrative.py", "up_llm_narrative")   # curate, fetch_reviews, build_prompt, SYSTEM_NARRATIVE
be = _load("up_llm_backends.py", "up_llm_backends")     # MODELS, available, call_model, parse_json

def main():
    ap = argparse.ArgumentParser(description="Narrative panel -- generation with 4 models")
    ap.add_argument("--features-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--modeling", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap.add_argument("--preds", default=os.path.join(ROOT, "results", "lente4_llm", "llm_user_preds_qwen2.5_7b.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lente4_llm", "panel"))
    ap.add_argument("--generators", default="opus,gpt4o,qwen,llama")
    ap.add_argument("--per-group", type=int, default=6, help="users per cell (TP/FP/FN/TN)")
    ap.add_argument("--max-reviews", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(args.seed)

    gens = [g for g in args.generators.split(",") if g in be.MODELS]
    avail = {g: be.available(g) for g in gens}
    print("generators:", {g: ("OK" if avail[g] else "UNAVAILABLE (no backend)") for g in gens})
    gens = [g for g in gens if avail[g]]
    if not gens: print("[ABORT] no generator available."); return

    sub = nb.curate(args.modeling, args.preds, args.per_group, args.seed)
    revs = nb.fetch_reviews(args.features_dir, sub)
    sub["reviews"] = sub["user_key"].map(lambda k: revs.get(k, []))
    sub = sub[sub["reviews"].map(len) > 0].reset_index(drop=True)
    print(f"curated sample: {len(sub)} users | cells: {sub['cell'].value_counts().to_dict()}")

    outpath = os.path.join(args.out, "narratives.jsonl")
    done = set()
    if os.path.exists(outpath):
        for line in open(outpath, encoding="utf-8"):
            try: r = json.loads(line); done.add((r["user_key"], r["generator"]))
            except Exception: pass
        print(f"resuming: {len(done)} (user,generator) already generated")

    with open(outpath, "a", encoding="utf-8") as f:
        for g in gens:
            t0 = time.time(); n = 0
            for _, row in sub.iterrows():
                if (row["user_key"], g) in done: continue
                prompt = nb.build_prompt(row["reviews"], args.max_reviews, rng)
                raw = be.call_model(g, nb.SYSTEM_NARRATIVE, prompt, max_tokens=400)
                j = be.parse_json(raw)
                f.write(json.dumps(dict(user_key=row["user_key"], cell=row["cell"], y_count=int(row["y_count"]),
                    n_reviews=int(row["n_reviews"]), generator=g, label=j.get("label", ""),
                    score=j.get("user_toxicity_score", ""), reasoning=j.get("reasoning", ""),
                    evidence=j.get("evidence_quote", ""), worst_review=j.get("worst_review", -1)),
                    ensure_ascii=False) + "\n"); f.flush(); n += 1
            print(f"  [{g}] {n} narratives in {time.time()-t0:.0f}s")
    print(f"-> {outpath}")

if __name__ == "__main__":
    main()
