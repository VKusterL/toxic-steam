#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_llm_batch.py  -  USER classification via the BATCH API (-50% cost) for proprietary LLMs

Same task as up_llm_user (1000 users stratified on the true prevalence, the COMPLETE LIST of reviews,
a NEUTRAL prompt with no game context -- reuses SYSTEM_USER/build_user_prompt/parse_* from up_llm_user),
submitted as an asynchronous BATCH (-50%):
  - OpenAI Batch API: CHUNKED by token budget (the org caps queued tokens at ~90k ->
    sequential sub-batches, each below the ceiling; resumes via _partial_<model>.jsonl).
  - Anthropic Message Batches: a single batch (without the cap above); resumes via batch_state_<model>.json.

Writes llm_user_preds_<model>.csv in the SAME format as up_llm_user => master_table/balanced_eval pick it up.

Usage:
  export ANTHROPIC_API_KEY=...
  python src/up_llm_batch.py --jobs openai:gpt-4o,claude:claude-opus-4-8
"""
from __future__ import annotations
import os, json, time, argparse, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             cohen_kappa_score, precision_recall_fscore_support)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fn, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
U = _load("up_llm_user.py", "up_llm_user")   # SYSTEM_USER, build_user_prompt, load_user_blocks, parse_*

def safe(model): return model.replace(":", "_").replace("/", "_")
def est_tokens(p): return len(p) // 4 + 150   # conservative estimate (input + reserved output)

# ---------------- OpenAI: CHUNKED by token budget ----------------
def _oai_line(model, i, prompt):
    return json.dumps({"custom_id": f"r{i}", "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": model, "temperature": 0, "max_tokens": 128,
                 "response_format": {"type": "json_object"},
                 "messages": [{"role": "system", "content": U.SYSTEM_USER}, {"role": "user", "content": prompt}]}})

def run_openai(client, model, prompts, outdir, budget, poll):
    # partition indices into chunks of <= budget queued tokens
    chunks, cur, tok = [], [], 0
    for i, p in enumerate(prompts):
        t = est_tokens(p)
        if cur and tok + t > budget: chunks.append(cur); cur, tok = [], 0
        cur.append(i); tok += t
    if cur: chunks.append(cur)
    print(f"[{model}] {len(prompts)} reqs -> {len(chunks)} sub-batches (<= {budget} tok/chunk)", flush=True)
    res = {}
    partial = os.path.join(outdir, f"_partial_{safe(model)}.jsonl")
    if os.path.exists(partial):
        for line in open(partial, encoding="utf-8"):
            o = json.loads(line); res[o["i"]] = o["t"]
        print(f"[{model}] resuming: {len(res)} already collected", flush=True)
    pf = open(partial, "a", encoding="utf-8")
    for ci, idxs in enumerate(chunks):
        if all(i in res for i in idxs): continue                 # chunk already collected
        path = os.path.join(outdir, f"_batchin_{safe(model)}_{ci}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in idxs: f.write(_oai_line(model, i, prompts[i]) + "\n")
        with open(path, "rb") as fh:                              # close the handle -> os.remove works (Windows)
            up = client.files.create(file=fh, purpose="batch")
        try: os.remove(path)
        except OSError: pass
        b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
        print(f"[{model}] sub-batch {ci+1}/{len(chunks)} ({len(idxs)} reqs) {b.id} ...", flush=True)
        while True:
            bb = client.batches.retrieve(b.id)
            if bb.status == "completed": break
            if bb.status in ("failed", "expired", "cancelled"):
                print(f"[{model}] sub-batch {ci+1} {bb.status}: {bb.errors}", flush=True); break
            time.sleep(poll)
        if bb.output_file_id:
            for line in client.files.content(bb.output_file_id).text.splitlines():
                if not line.strip(): continue
                o = json.loads(line); i = int(o["custom_id"][1:])
                body = (o.get("response") or {}).get("body") or {}
                ch = (body.get("choices") or [{}])[0]
                txt = (ch.get("message") or {}).get("content") or ""
                res[i] = txt; pf.write(json.dumps({"i": i, "t": txt}) + "\n")
            pf.flush()
        print(f"[{model}]   collected so far: {len(res)}/{len(prompts)}", flush=True)
    pf.close()
    try: os.remove(partial)
    except OSError: pass
    return res

# ---------------- Anthropic: a single batch ----------------
def run_anthropic(client, model, prompts, outdir, poll):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    sp = os.path.join(outdir, f"batch_state_{safe(model)}.json")
    if os.path.exists(sp):
        bid = json.load(open(sp))["batch_id"]; print(f"[{model}] resuming batch {bid}", flush=True)
    else:
        reqs = [Request(custom_id=f"r{i}", params=MessageCreateParamsNonStreaming(
                    model=model, max_tokens=128, system=U.SYSTEM_USER,
                    messages=[{"role": "user", "content": p}])) for i, p in enumerate(prompts)]
        bid = client.messages.batches.create(requests=reqs).id
        json.dump({"backend": "claude", "batch_id": bid}, open(sp, "w"))
        print(f"[{model}] batch {bid} submitted ({len(prompts)} reqs)", flush=True)
    while client.messages.batches.retrieve(bid).processing_status != "ended":
        time.sleep(poll); print(f"  [{model}] in_progress", flush=True)
    res = {}
    for r in client.messages.batches.results(bid):
        i = int(r.custom_id[1:])
        res[i] = ("".join(x.text for x in r.result.message.content if getattr(x, "type", "") == "text")
                  if r.result.type == "succeeded" else f"ERR:{r.result.type}")
    try: os.remove(sp)
    except OSError: pass
    return res

def write_preds(sel, res, model, outdir):
    n = len(sel); preds = np.zeros(n, int); scores = np.zeros(n); worst = np.full(n, -1); raws = [""] * n
    for i in range(n):
        raw = res.get(i, ""); lab = U.parse_label(raw)
        preds[i] = lab if lab is not None else 0
        scores[i] = U.parse_score(raw, lab); worst[i] = U.parse_worst(raw); raws[i] = str(raw)[:300]
    d = sel.copy(); d["_pred"], d["_score"], d["_worst"], d["_raw"] = preds, scores, worst, raws
    d[["user_key", "block", "y_count", "n_reviews", "_pred", "_score", "_worst", "_raw"]].to_csv(
        os.path.join(outdir, f"llm_user_preds_{safe(model)}.csv"), index=False)
    y = d["y_count"].to_numpy().astype(int)
    miss = sum(1 for i in range(n) if not res.get(i) or str(res.get(i)).startswith("ERR"))
    pr, rc, _, _ = precision_recall_fscore_support(y, preds, average=None, labels=[0, 1], zero_division=0)
    print(f"[{model}] AUC-PR {average_precision_score(y, scores):.3f} ROC {roc_auc_score(y, scores):.3f} "
          f"F1m {f1_score(y, preds, average='macro', zero_division=0):.3f} kappa {cohen_kappa_score(y, preds):.3f} "
          f"P={pr[1]:.2f} R={rc[1]:.2f} | faltas/erros={miss} -> llm_user_preds_{safe(model)}.csv", flush=True)

def main():
    ap = argparse.ArgumentParser(description="LLM classification via the Batch API (-50%)")
    ap.add_argument("--features-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--modeling", default=os.path.join(ROOT, "results", "user_predictor", "modeling_users_10pct.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lente4_llm"))
    ap.add_argument("--jobs", default="openai:gpt-4o,claude:claude-opus-4-8", help="csv of backend:model")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=200)
    ap.add_argument("--cap-reviews", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll", type=int, default=45, help="polling interval (s)")
    ap.add_argument("--openai-budget", type=int, default=60000, help="tokens/chunk for OpenAI (< 90k org ceiling)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(args.seed)

    jobs = [(j.split(":", 1)[0].strip(), j.split(":", 1)[1].strip()) for j in args.jobs.split(",")]
    clients = {}
    if any(be == "openai" for be, _ in jobs):
        if not os.environ.get("OPENAI_API_KEY"): print("[ABORT] OPENAI_API_KEY missing."); return
        import openai; clients["openai"] = openai.OpenAI()
    if any(be == "claude" for be, _ in jobs):
        if not os.environ.get("ANTHROPIC_API_KEY"): print("[ABORT] ANTHROPIC_API_KEY missing."); return
        import anthropic; clients["claude"] = anthropic.Anthropic()

    sel, prev = U.load_user_blocks(args.features_dir, args.modeling, args.blocks, args.block_size, args.seed)
    prompts = [U.build_user_prompt(r, args.cap_reviews, rng) for r in sel["reviews"]]
    print(f"sample: {len(sel)} users, prev {prev*100:.2f}% | jobs: {jobs}", flush=True)

    for be, mo in jobs:
        res = (run_openai(clients["openai"], mo, prompts, args.out, args.openai_budget, args.poll)
               if be == "openai" else run_anthropic(clients["claude"], mo, prompts, args.out, args.poll))
        write_preds(sel, res, mo, args.out)

if __name__ == "__main__":
    main()
