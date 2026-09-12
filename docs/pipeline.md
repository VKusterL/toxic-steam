# Pipeline

The user-level modeling pipeline is a sequence of standalone scripts. Each one reads the artifacts produced by earlier stages and writes its own outputs under `results/`. All scripts are meant to be run from the repository root and default to reading the collected corpus under `data/` and writing under `results/`.

The raw corpus is not distributed (see [data.md](data.md)), so a full rerun requires the parquet corpus under `data/`. The stored `results/` include aggregate summaries and some identifiable per-user outputs; use the publication export described in [artifact.md](artifact.md) for distribution. Embeddings, the modeling table, and classical out-of-fold (OOF) predictions are not included. Consequently, inspecting the supplied summaries is possible without the corpus, but rebuilding every summary is not.

## Stage 1: Feature construction

`build_features.py` reads the raw parquet corpus and materializes three modeling artifacts, all restricted to English reviews.

- `data/features/reviews_resolved/`, the review-level table, deduplicated by review URL and joined to user profiles through the Steam identifier. The review label is `y_review = (toxicity >= 0.9) OR (perspective_score >= 0.7)`. The train/test split is defined per user so that no user appears on both sides.
- `data/features/review_train_sample.parquet`, a training-split sample containing every positive and up to four negatives per positive by default (`--neg-ratio 4`). This auxiliary review-level sample is not the user-level sample used by the later transformer and LLM scripts. The test partition keeps the real prevalence.
- `data/features/user_features.parquet`, the user-level table with one row per matchable user. The user label is `y_user = MAX(y_review)`, meaning the user authored at least one toxic review. This table holds only behavioral and profile features, never review text or toxicity scores.

It also writes `feature_dictionary.csv` and `feature_cols.py`, which encode the anti-leakage feature contract used by the modeling scripts.

```bash
python src/build_features.py --root . --out data/features --langs en
```

## Stage 2: Modeling substrate

`up_build_dataset.py` turns the user-level table into the modeling slice.

- A deterministic 10 percent slice, selected by `hash(user_key) % 10 == 0`. Because the hash is independent of the label, the slice preserves the real prevalence and is exactly reproducible without a stored seed.
- Two label definitions, kept as separate populations rather than unioned: `y_count` (at least one toxic review, prevalence about 4 percent) and `y_rate` (at least 5 percent of a user's reviews toxic, among users with at least 5 reviews, prevalence about 12.8 percent).
- Stratified cross-validation fold assignments.

Output: `results/user_predictor/modeling_users_10pct.parquet`.

```bash
python src/up_build_dataset.py --kfolds 5
```

## Stage 3: Text representation

`up_embed_reviews.py` computes the expensive, reusable artifact: a Sentence-BERT embedding for every English review of the sliced users. The default encoder is `all-MiniLM-L6-v2` (384 dimensions). The cache is resumable per hash bucket, so an interrupted run continues where it stopped.

`up_user_vectors.py` aggregates that per-review cache into one vector per user, by averaging. It produces two aggregations: `ALL` (mean over every review, the primary content representation) and `NONTOX` (mean over the user's non-toxic reviews only, the leave-toxic-out representation). Outputs are `user_vectors_all.npy`, `user_vectors_nontox.npy`, and `user_vectors_index.parquet` under `results/user_predictor/`.

The MPNet robustness check (`all-mpnet-base-v2`, 768 dimensions) uses a separate cache and output directory under `results/user_predictor_mpnet/`. Never reuse a MiniLM cache for MPNet: cache reuse is based on filenames, not a model-configuration check.

```bash
python src/up_embed_reviews.py
python src/up_user_vectors.py
```

For the alternative encoder:

```bash
python src/up_embed_reviews.py --model all-mpnet-base-v2 --out results/user_predictor_mpnet/emb_cache
python src/up_user_vectors.py --cache results/user_predictor_mpnet/emb_cache --out results/user_predictor_mpnet
```

## Stage 4: Classical benchmark

`up_train_userlevel.py` runs the feature-family ablation with five-fold cross-validation, folds shared across models so that comparisons are paired. All preprocessing and threshold selection happen inside each fold to avoid leakage, and AUC-PR is the anchor metric. The feature arms are:

- `content_agnostic`: profile metadata only (the non-circular headline).
- `content_agnostic+ban`: metadata plus public ban indicators.
- `content`: mean review embedding (the content upper bound).
- `content_leavetoxicout`: mean of the non-toxic reviews only.
- `combined`: profile metadata plus content.

It writes `metrics_cv.csv`, `metrics_agg.csv`, and per-arm out-of-fold predictions under `results/replication/oof/`.

`up_lto_control.py` runs the clean leave-toxic-out control. It restricts the population to users with at least one non-toxic review, so that the leave-toxic-out vector is always real (never imputed), which removes the missingness artifact that would otherwise turn label into a perfect proxy. It reports on two populations and writes `results/replication/lto_control.csv`.

```bash
python src/up_train_userlevel.py --kfolds 5 --models logreg,linsvc,histgb,xgb,lgbm,mlp,svc
python src/up_lto_control.py
```

The explicit five-fold setting and model list match `metrics_cv.csv`; the trainer's defaults instead select ten folds and a different model set. For the MPNet benchmark, use the same options with `--vecdir results/user_predictor_mpnet --out results/replication_mpnet`. The clean LTO script has its own `--kfolds` option, defaulting to ten; its aggregate CSV does not record the fold count. The full-matrix `content_leavetoxicout` arm imputes missing vectors and is a diagnostic of that artifact; use `lto_control.csv` for the restricted-population result quoted in the paper.

## Stage 5: Transformer

`up_bert_user.py` fine-tunes DistilBERT end to end on the concatenated review text of each user, under a stratified user cap for GPU feasibility. Class balancing by undersampling is applied only to the fit portion of each training fold, while validation and test folds keep the real prevalence. Per-fold metrics and predictions land under `results/replication_bert_undersampled/`. `up_bert_diagnostics.py` produces lightweight training diagnostics from the fold logs.

```bash
python src/up_bert_user.py --model distilbert-base-uncased --cap 60000 --kfolds 10 --undersample
python src/up_bert_diagnostics.py --out data/diagnostics/bert_undersampled
```

`--undersample` appends `_undersampled` to `--out`, including when an explicit output path is supplied. The tokenizer keeps at most 256 tokens by default; input assembly also caps each review at 500 characters. The supplied ten-fold undersampled run is the canonical DistilBERT result. The older five-fold directories are legacy experiments, and their threshold-dependent metrics should not be substituted for the canonical run.

## Stage 6: LLMs as classifiers

`up_llm_user.py` evaluates open-weight LLMs (served locally through Ollama) as direct user-level classifiers on a stratified 1,000-user sample split into five prevalence-preserving blocks. Each model receives up to 300 reviews per user, with each review capped at 1,500 characters, and returns a label plus a 0 to 100 score. For longer histories, reviews are sampled independently of toxicity scores. `up_llm_batch.py` runs the same task for the proprietary models through the Anthropic and OpenAI batch APIs. Both write `llm_user_preds_<model>.csv` in a shared format, plus `llm_user_metrics.csv`, under `results/lente4_llm/`. `up_llm_backends.py` supplies the multi-backend caller for the narrative panel.

```bash
python src/up_llm_user.py  --models llama3.1:8b,qwen2.5:7b --blocks 5 --block-size 200
python src/up_llm_batch.py --jobs openai:gpt-4o,claude:claude-opus-4-8
```

## Stage 7: LLMs as explanation generators

`up_llm_panel_generate.py` has four models (Opus, GPT-4o, Qwen, Llama) each generate a zero-shot narrative that explains why a sampled user was associated with toxic behavior, reading only the review texts. `up_llm_panel_judge.py` scores those narratives with a cross-model judge panel, where a model never judges its own narrative, along faithfulness, coverage, clarity, calibration, and overall quality. Outputs are `results/lente4_llm/panel/narratives.jsonl` and `results/lente4_llm/panel/judgments.csv`. `up_llm_narrative.py` produces the single-model curated narrative set used for the qualitative section.

```bash
python src/up_llm_panel_generate.py --generators opus,gpt4o,qwen,llama
python src/up_llm_panel_judge.py    --judges     opus,gpt4o,qwen,llama
```

## Stage 8: Consolidation

- `up_balanced_eval.py` produces an exploratory real-prevalence versus balanced (50/50) comparison from OOF predictions, writing `results/replication/balanced_eval.csv` and a plot. It retunes thresholds on the evaluated labels, so its F1 values are not the original held-out-threshold benchmark.
- `up_significance.py` computes AUC-PR by `n_reviews` bucket and paired bootstrap deltas between classical feature arms, writing `bucket_metrics.csv` and `pairwise_bootstrap.csv`. It also supports classical model comparisons in `pairwise_model_bootstrap.csv`. It does not implement the paper's four cross-family confirmatory tests or their Holm-adjusted p-values.
- `up_master_table.py` consolidates models, arms, and encoders into `results/replication/master_table.csv`. Classical threshold-dependent metrics are retuned per evaluated OOF fold. LLM metrics use the model's binary decision, and canonical DistilBERT metrics come from its validation-threshold fold files. See [results.md](results.md) for this distinction.
- `up_plots.py` produces exploratory prediction plots under `results/figures/`, using classical OOF files and the modeling table where required. With aggregate summaries alone, only the clean LTO plot is available. The characterization figure PDFs used in the manuscript come from `characterization/figures/`, not this script.

```bash
python src/up_balanced_eval.py
python src/up_significance.py
python src/up_master_table.py
python src/up_plots.py
```

Run consolidation only after generating its private intermediate inputs, in a working copy of the repository. These commands overwrite result CSVs; running with missing OOF files can replace a complete table with a partial one. To inspect published results without changing them, follow [artifact.md](artifact.md).
