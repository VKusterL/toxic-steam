# Artifact evaluation guide

This artifact accompanies the accepted AIIDE 2026 paper *Toxic Steam: A
Large-Scale Characterization and Prediction of Toxic Users in a Gaming
Platform*. It provides the implementation, documented experimental setup,
and selected aggregate results for inspection. It does **not** provide the
Steam corpus, user-level records, trained models, or a complete independent
reproduction of every paper result. The artifact evaluation invitation is
optional and separate from the accepted paper's layout correction.

## Reviewer entry point

1. Read [the project overview](../README.md) and the claim-to-file map below.
2. Verify the delivered ZIP with Python 3.10 or newer; this step uses only
   the Python standard library, requires no GPU, account, or API key, and
   does not run the research pipeline:

   ```bash
   python tools/package_artifact.py --verify /path/to/toxic-steam-artifact.zip
   ```

   Expected output begins with `Verified` and reports the included/excluded
   file counts and ZIP SHA-256. Verification reads the archive without
   extracting it. The manifest records a SHA-256 for each included file,
   its original working-tree SHA-256, and any export-only transformation.
   These hashes check integrity against the supplied manifest; they are
   not an independent signature of the research results.
3. Inspect the CSV and JSON files directly. [Results](results.md) explains
   the metric definitions, populations, and known evidence gaps.
4. To run analysis on a separately held corpus, use [data](data.md),
   [pipeline](pipeline.md), [reproducibility](reproducibility.md), and the
   [characterization guide](../characterization/README.md). Research code
   needs the documented dependencies; the standard-library-only statement
   applies to artifact packaging and verification.

## Paper-to-artifact map

Paper labels are given alongside numbers so the mapping remains useful
when floats move during typesetting.

| Paper item | Implementation / supplied result | What can be checked without the corpus |
|---|---|---|
| Corpus size, prevalence, correlations, engagement medians, and tag-rate claims | `characterization/verify_numbers.py`; `results/characterization/corpus_verification.json` | Recorded paper/recomputed pairs and comparison precision for 26 claims; recomputation itself requires the original corpus. |
| Table 1: annotation agreement (`tab:kappa_intervalos`) | `characterization/annotation_agreement/run_agreement_table.py` | Method and implementation; the human annotation input and a separate aggregate table are not shipped. |
| Table 2: regular/toxic TF-IDF trends | `characterization/step03_tfidf_analysis/run_tfidf.py`, `run_top_terms.py` | Method and implementation; the global lexical table is not separately supplied. The per-tag heatmap matrix is a different aggregate. |
| Figure 1: top game tags | `characterization/figures/run_tag_toxicity.py`; `results/characterization/tag_toxicity_top.csv`, `tag_toxicity_all.csv`, `tag_toxicity_report.json` | Plotted counts/rates, the volume threshold, and lower-rate tags quoted in the text. |
| Figure 2: TF-IDF heatmap | `characterization/figures/run_tag_tfidf_heatmap.py`; `results/characterization/tag_tfidf_means.csv`, `tag_tfidf_report.json` | Every plotted cell, vectorizer summary, and recorded term-order checks. |
| Figure 3: user profile CDFs | `characterization/figures/run_user_profile.py`; `results/characterization/user_profile_report.json` | Medians, populations, ban rates, and recommendation share. The full empirical CDF coordinates are not distributed; redrawing those curves needs the corpus. |
| Table 3: confirmatory statistical tests | `src/up_significance.py`; see [results caveats](results.md) | Bootstrap implementation and reported conclusions. `results/replication/pairwise_bootstrap.csv` is an ablation result with a different schema, not a complete backing file for the paper's four confirmatory tests. |
| Table 4: user-level prediction (`tab:user_prediction_results`) | `results/replication/master_table.csv`, `metrics_agg.csv`; `results/replication_bert_undersampled/bert_user_metrics.csv`; `results/lente4_llm/llm_user_metrics.csv` | Aggregate model/fold metrics. Row-level predictions are excluded, so independent metric recomputation is unavailable in the public export. |
| Table 5: leave-toxic-out control (`tab:lto`) | `src/up_lto_control.py`; `results/replication/lto_control.csv` | Aggregate scores, paired differences, confidence intervals, and population sizes. |
| Table 6: prevalence comparison (`tab:balanced`) | `src/up_balanced_eval.py`; `results/replication/balanced_eval.csv`; see [results caveats](results.md) | A separate exploratory pooled export is supplied; it does not reproduce every fold-level value in the paper. The original balanced DistilBERT aggregate export remains missing. |
| LLM narrative panel | `src/up_llm_panel_generate.py`, `src/up_llm_panel_judge.py`; [results discussion](results.md) | Prompts, procedure, and documented scores. User-level narratives and judgments are excluded, so the public export does not independently reproduce panel scores or agreement. |

The files under `results/replication_bert/` and
`results/replication_bert_mpnet/` record additional transformer runs;
`results/replication_mpnet/` records the classical encoder robustness check.
Keep these runs distinct from the paper's ten-fold DistilBERT result in
`replication_bert_undersampled/`. Read [results](results.md) before equating
metrics across populations or evaluation protocols.

## Preparing the public ZIP

From the working repository, run:

```bash
python tools/package_artifact.py
python tools/package_artifact.py --verify dist/toxic-steam-artifact.zip
```

Building requires Git and Python 3.10 or newer. It reads current contents
of allowlisted tracked files, plus the explicitly listed publication files
in `EXTRA_FILES`. It does not run experiments, access model APIs, or read
the raw corpus. Rebuild after final documentation or code edits so the
archive reflects the delivered revision. Other untracked files are not
included automatically. An alternative `--output` must be a ZIP directly
under this repository's `dist/` directory.

The ZIP includes code, dependency lists, documentation, the license, and
allowlisted aggregate CSV/JSON outputs. `ARTIFACT_MANIFEST.json` lists every
included file and every tracked file omitted by the export policy. It also
records export-only normalization of machine-specific figure locations in
three characterization JSON reports. Original result files and numerical
values remain unchanged.

The export excludes:

- All raw/intermediate data, caches, model weights, binaries, manuscript
  delivery files, and `.git` history.
- `results/lente4_llm/llm_user_preds_*.csv` and
  `results/replication_bert_undersampled/fold_*_preds.csv`, which contain
  user-level account keys.
- `results/lente4_llm/panel/judgments.csv` and `narratives.jsonl`, which
  contain account-linked judgments or free text.
- Supplementary notebooks, whose embedded outputs are outside this
  reviewed export.

Validation rejects unsafe archive paths, duplicate members, symlinks,
files outside the allowlist, binary contents, oversized entries, selected
identifier/credential patterns, and prohibited row-level fields in result
tables. This is a bounded release check, not a general anonymization tool
or proof that arbitrary new files are safe; new result schemas require
review before expanding the allowlist.

The working repository retains its original experiments, including
identifiable row-level results. This utility preserves those originals and
does not rewrite existing Git history. Use the verified ZIP as the public
artifact; a public Git release should be created from its reviewed contents
without copying the working repository's history. Provide the corrected
paper PDF and its LaTeX sources through the separate paper submission.

No artifact has been submitted to the conference by this utility. It
creates a local package for review and delivery.
