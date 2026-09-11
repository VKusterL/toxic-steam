#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
up_llm_panel_judge.py  -  Narrative panel: CROSS-JUDGING (anti-bias) + analysis

Each narrative is judged by the models that did NOT write it (no self-judging -> eliminates
self-preference). The judge reads the FULL LIST of the user's reviews + the narrative and scores:
faithfulness, coverage, clarity, calibration, overall (1-5). It reports:
  - RANKING of the generators by mean score (only from judges that are not the generator)
  - INTER-JUDGE AGREEMENT (Spearman) -> detects residual family bias
  - (optional --include-self) self-preference diagnostic: own score vs third-party scores

Input: results/lente4_llm/panel/narratives.jsonl (from up_llm_panel_generate.py)
Outputs: panel/judgments.csv + panel/panel_report.txt
Usage: python src/up_llm_panel_judge.py --judges opus,gpt4o,qwen,llama
"""
from __future__ import annotations
import os, json, time, argparse, importlib.util, io
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(fn, mod):
    spec = importlib.util.spec_from_file_location(mod, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
nb = _load("up_llm_narrative.py", "up_llm_narrative")
be = _load("up_llm_backends.py", "up_llm_backends")
_BUF = io.StringIO()
def out(*a, **k): print(*a, **k, flush=True); print(*a, **k, file=_BUF)

SYSTEM_JUDGE = (
"You are an impartial evaluator of EXPLANATIONS for a user-level toxicity classification. You are "
"given a Steam user's reviews and an explanation (written by another model) of whether the user is "
"toxic and why. Judge ONLY the explanation's quality against the reviews — not whether you agree "
"with the toxic/nontoxic label itself. Score 1-5 on:\n"
"  faithfulness: do the cited reviews/claims actually appear and support the explanation? (5=fully)\n"
"  coverage: does it address the relevant evidence (the toxic review if any)?\n"
"  clarity: is the reasoning clear and specific (not vague)?\n"
"  calibration: does the confidence/score match the strength of evidence?\n")
ASK_JUDGE = ('Respond ONLY JSON: {"faithfulness":<1-5>,"coverage":<1-5>,"clarity":<1-5>,'
             '"calibration":<1-5>,"overall":<1-5>}. No text outside JSON.')

def judge_prompt(reviews, narr, cap, rng):
    n = len(reviews); shown = reviews; extra = 0
    if n > cap:
        sel = np.sort(rng.choice(n, cap, replace=False)); shown = [reviews[i] for i in sel]; extra = n - cap
    parts = [f"User's reviews ({n} total{', ' + str(cap) + ' shown' if extra else ''}):"]
    for i, t in enumerate(shown): parts.append(f'[{i}] """{str(t)[:1200]}"""')
    if extra: parts.append(f"... and {extra} more.")
    parts.append(f'\nEXPLANATION to judge (label={narr.get("label","")}, score={narr.get("score","")}):')
    parts.append(f'"{narr.get("reasoning","")}" Evidence cited: "{narr.get("evidence","")}"')
    parts.append(ASK_JUDGE)
    return "\n".join(parts)

DIMS = ["faithfulness", "coverage", "clarity", "calibration", "overall"]

def main():
    ap = argparse.ArgumentParser(description="Panel -- cross-judging no-self + analysis")
    ap.add_argument("--features-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--panel", default=os.path.join(ROOT, "results", "lente4_llm", "panel"))
    ap.add_argument("--judges", default="opus,gpt4o,qwen,llama")
    ap.add_argument("--include-self", action="store_true", help="also judge a model's own narratives (bias diagnostic)")
    ap.add_argument("--max-reviews", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    narrs = [json.loads(l) for l in open(os.path.join(args.panel, "narratives.jsonl"), encoding="utf-8")]
    nd = pd.DataFrame(narrs)
    out(f"narratives: {len(nd)} | generators: {sorted(nd.generator.unique())}")
    judges = [j for j in args.judges.split(",") if j in be.MODELS and be.available(j)]
    out(f"available judges: {judges}")
    if not judges: out("[ABORT] no judge available."); return

    revs = nb.fetch_reviews(args.features_dir, nd.drop_duplicates("user_key"))
    jpath = os.path.join(args.panel, "judgments.csv")
    rows = []; done = set()
    if os.path.exists(jpath):
        old = pd.read_csv(jpath); rows = old.to_dict("records")
        done = {(r["user_key"], r["generator"], r["judge"]) for r in rows}
        out(f"resuming: {len(done)} judgments already done")

    for _, nr in nd.iterrows():
        for jg in judges:
            if jg == nr["generator"] and not args.include_self: continue   # NO-SELF (anti-bias)
            key = (nr["user_key"], nr["generator"], jg)
            if key in done: continue
            rv = revs.get(nr["user_key"], [])
            raw = be.call_model(jg, SYSTEM_JUDGE, judge_prompt(rv, nr, args.max_reviews, rng), max_tokens=120)
            s = be.parse_json(raw)
            rows.append(dict(user_key=nr["user_key"], generator=nr["generator"], judge=jg, self=(jg == nr["generator"]),
                             **{d: s.get(d, np.nan) for d in DIMS}))
            pd.DataFrame(rows).to_csv(jpath, index=False)
    J = pd.DataFrame(rows)
    for d in DIMS: J[d] = pd.to_numeric(J[d], errors="coerce")

    # RANKING of the generators (only judges != generator)
    cross = J[J["self"] == False]
    out("\n=== RANKING of the generators (mean 'overall' score, no-self cross judges) ===")
    rk = cross.groupby("generator")["overall"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    out(rk.round(3).to_string())
    out("\n  per dimension (cross mean):")
    out(cross.groupby("generator")[DIMS].mean().round(3).to_string())

    # INTER-JUDGE AGREEMENT (Spearman on 'overall', paired by narrative)
    out("\n=== Inter-judge agreement (Spearman on 'overall') ===")
    piv = cross.pivot_table(index=["user_key", "generator"], columns="judge", values="overall")
    js = list(piv.columns)
    for i in range(len(js)):
        for k in range(i + 1, len(js)):
            a, b = piv[js[i]], piv[js[k]]; m = a.notna() & b.notna()
            if m.sum() >= 5:
                rho, _ = spearmanr(a[m], b[m]); out(f"  {js[i]} x {js[k]}: rho={rho:.3f} (n={int(m.sum())})")

    # self-preference diagnostic (if any self judgments exist)
    if args.include_self and (J["self"] == True).any():
        out("\n=== Self-preference (OWN score vs THIRD-PARTY score for the SAME narratives) ===")
        for g in nd.generator.unique():
            selfsc = J[(J.generator == g) & (J.self == True)]["overall"].mean()
            othersc = cross[cross.generator == g]["overall"].mean()
            out(f"  {g}: self={selfsc:.3f} vs others={othersc:.3f}  (bias={selfsc-othersc:+.3f})")

    with open(os.path.join(args.panel, "panel_report.txt"), "w", encoding="utf-8") as f: f.write(_BUF.getvalue())
    out(f"\n-> {jpath} + panel_report.txt")

if __name__ == "__main__":
    main()
