# Toxic Steam: A Large-Scale Characterization and Prediction of Toxic Users in a Gaming Platform

This repository holds the code and quantitative results behind the study *"Toxic Steam: A Large-Scale Characterization and Prediction of Toxic Users in a Gaming Platform"*, accepted at AIIDE 2026. It is organized so that a reader can inspect the supplied results or run the modeling pipeline with a separately held corpus.

For the optional artifact evaluation, start with [docs/artifact.md](docs/artifact.md). Generate the public package with `python tools/package_artifact.py`; it excludes identifiable user-level outputs and Git history while preserving the research originals. See [docs/results.md](docs/results.md) for the exact coverage and remaining provenance gaps. The corrected paper has a separate Overleaf delivery described in [docs/paper_delivery.md](docs/paper_delivery.md).

Steam is one of the largest digital distribution platforms for PC games, and its user reviews form a public arena where toxic discourse frequently slips past moderation. Most automatic tooling scores each review in isolation and says little about the people who repeatedly produce toxic content, even though those users are the more practical target for platform-level moderation. This work looks at toxicity from both angles. It first characterizes how toxic language manifests across 36.8 million English-language reviews written by 14.1 million users, and then asks whether toxic users can be predicted from account-level data, and whether general-purpose Large Language Models (LLMs) can support that decision as classifiers and as explanation generators.

> **Content warning.** This project and the data it describes contain examples of offensive and hateful language. Illustrative quotes appear in the paper and in some generated artifacts.

## Research questions

The study is organized around three research questions.

- **RQ1.** How does toxicity manifest in English-language Steam reviews, lexically, across game contexts, and in user behavior?
- **RQ2.** Can toxic users be predicted from user-level data, and does the resulting signal reflect a genuine behavioral pattern rather than circularity with the label definition?
- **RQ3.** Can general-purpose LLMs support user-level moderation, either as direct classifiers or as generators of evidence-grounded explanations?

A design choice runs through the whole prediction study: unlike most prior toxicity work, which reports numbers on artificially balanced or case-control samples, every user-level model here is evaluated under the real platform prevalence of roughly 4 percent toxic users. Balanced numbers are reported only as a secondary reference for comparison with earlier literature.

## What the study finds

- Toxicity is rare but structurally patterned. It accounts for 1.86 percent of reviews and 3.95 percent of users, concentrates in competitive and team-based game contexts (Team-Based 2.93 percent, Competitive 2.90 percent, PvP 2.57 percent), and frequently coexists with positive evaluations: 57.5 percent of toxic reviews still recommend the game.
- Public profile metadata is a weak predictor of toxic users (AUC-PR 0.070), while text-derived representations are far stronger: AUC-PR 0.476 with a classical model on content embeddings, and 0.643 with a fine-tuned DistilBERT.
- A leave-toxic-out control, which removes the exact reviews that define each user's label before building their representation, still beats the metadata baseline (AUC-PR 0.166 versus 0.063 for XGBoost). Most of the predictive power is circular with respect to the label, but a weaker and statistically reliable behavioral trace survives outside the flagged text.
- As direct classifiers, general-purpose LLMs trail the supervised models, but they add clear value as an explanation layer, producing auditable, evidence-grounded narratives for human review. A cross-model judge panel rates Claude Opus as the strongest generator, under weak and uneven inter-judge agreement that keeps this evaluation diagnostic rather than definitive.

The take-away is that user-level toxicity prediction is best positioned as a triage instrument that prioritizes human attention at platform scale, not as a verdict on individual character.

## Repository structure

```
.
├── README.md                 This document
├── LICENSE                   MIT license
├── requirements.txt          Python dependencies (pinned)
├── characterization/         Characterization pipeline (RQ1) - see its own README
│   ├── step01_cleaning_and_language_detection/  Corpus cleaning, dedup, language ID
│   ├── step02_run_detoxify/                     Detoxify scoring, model agreement, labeling
│   ├── step03_tfidf_analysis/                   TF-IDF toxic vs. non-toxic lexicon
│   ├── annotation_agreement/                    Inter-annotator agreement, threshold calibration
│   ├── review_examples/                         Sampling the reviews quoted in the paper
│   └── figures/                                 The characterization figures and their numbers
├── src/                      User Toxicity Predictor pipeline (one script per stage)
│   ├── build_features.py         Materialize review-level and user-level feature tables
│   ├── up_build_dataset.py       10 percent modeling slice, count/rate labels, CV folds
│   ├── up_embed_reviews.py       Sentence-BERT embedding cache, one vector per review
│   ├── up_user_vectors.py        Aggregate review embeddings into one vector per user
│   ├── up_train_userlevel.py     Classical ablation matrix (profile / content / LTO / combined)
│   ├── up_lto_control.py         Clean leave-toxic-out control on a well-defined population
│   ├── up_bert_user.py           End-to-end DistilBERT fine-tuning on user review text
│   ├── up_bert_diagnostics.py    Lightweight diagnostics for the transformer runs
│   ├── up_balanced_eval.py       Real-prevalence versus balanced (50/50) evaluation
│   ├── up_significance.py        Paired bootstrap tests and n_reviews stratification
│   ├── up_llm_backends.py        Unified multi-LLM caller (Ollama, Anthropic, OpenAI)
│   ├── up_llm_user.py            LLMs as direct user-level classifiers (local backends)
│   ├── up_llm_batch.py           Same task via the Anthropic and OpenAI batch APIs
│   ├── up_llm_narrative.py       Zero-shot user-level explanation narratives
│   ├── up_llm_panel_generate.py  Four-model narrative generation panel
│   ├── up_llm_panel_judge.py     Cross-model LLM-as-a-judge scoring of narratives
│   ├── up_master_table.py        Consolidated results table with standard deviations
│   └── up_plots.py               Regenerate figures locally from the result artifacts
├── notebooks/                Supplementary language-distribution analysis
│   ├── language_distribution_full.ipynb
│   └── language_distribution_6months.ipynb
├── docs/                     Extended documentation
│   ├── pipeline.md               Stage-by-stage inputs and outputs
│   ├── data.md                   Data collection, labeling, feature groups, availability
│   ├── results.md                Full result tables and statistical tests
│   ├── reproducibility.md        Environment, seeds, hardware, determinism
│   ├── artifact.md               Public export and reviewer guide
│   ├── paper_delivery.md         Corrected manuscript and Overleaf instructions
│   └── ethics.md                 Responsible use and limitations
├── tools/                    Publication packaging and PDF layout checks
└── results/                  Research results; export filters identifiable records
    ├── characterization/         RQ1 counts, rates, TF-IDF matrix, verification reports
    ├── replication/              Classical models: metrics, out-of-fold summaries, LTO, balanced
    ├── replication_bert*/        DistilBERT fold metrics (MiniLM, MPNet, undersampled)
    ├── replication_mpnet/        MPNet robustness check
    └── lente4_llm/               LLM classifier metrics and the narrative judge panel
```

The repository is split along the paper's two halves. `characterization/`
holds everything behind the *Toxicity Characterization* section (RQ1): the
corpus construction, the toxicity labeling and its calibration, the lexical
and contextual analysis, and the figures. Everything else - `src/`,
`docs/`, `results/`, `notebooks/` - is the user-level prediction benchmark
(RQ2 and RQ3). The two were run separately and expect their data in
different historical layouts. The commands below use `data/` for both;
some characterization step READMEs retain the earlier `steam-data/`
examples, whose paths are configurable through command-line arguments.

## Data availability and ethics

The raw review, user, and game data are **not distributed** with this repository. They were collected from the public Steam Web API between September 2024 and April 2025 and are subject to Steam's Terms of Service. The users who wrote these reviews did not consent to toxicity profiling, so identifiable raw data is not redistributed. Any re-collection should pseudonymize Steam identifiers and profile URLs.

The user-level predictor is a research and measurement instrument, not a moderation oracle. Its labels are tool-derived and partially circular, its estimates carry real uncertainty, and it should only ever inform aggregate analysis or human-review triage, never an automatic verdict on an individual. See [docs/ethics.md](docs/ethics.md) for the full discussion.

Running `src/` requires the collected parquet corpus under `data/` and the intermediate outputs described in [docs/pipeline.md](docs/pipeline.md). Supplied summaries support many paper results, but the four confirmatory tests and some balanced-evaluation results lack their original aggregate exports. The research checkout also contains identifiable per-user predictions and narrative records, excluded from the public artifact ZIP. Figures are generated locally: `characterization/figures/` produces the manuscript figures; `src/up_plots.py` produces exploratory prediction plots and requires excluded intermediates for most of them.

## Setup

The pipeline was developed with Python 3.13 on Windows 11 (CPU plus an NVIDIA GTX 1070, 8 GB). A GPU is only needed for the transformer and embedding stages.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# Install torch first from the CUDA index if you want GPU support:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The LLM stages need extra services: the Anthropic and OpenAI APIs for the proprietary models, and a local [Ollama](https://ollama.com) server for the open-weight models.

```bash
export ANTHROPIC_API_KEY=...   # Claude Opus (classifier and generator)
export OPENAI_API_KEY=...      # GPT-4o (classifier and LLM-as-a-judge)
ollama serve                   # serves llama3.1:8b and qwen2.5:7b on port 11434
```

On Windows with Anaconda, set `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch to avoid an OpenMP conflict. The pipeline scripts set this automatically.

## Reproducing the characterization (RQ1)

The characterization half lives in [characterization/](characterization/) and runs on its own, before and independently of the prediction pipeline. Three numbered steps plus three cross-cutting tools; each step reads the previous one's output. Full detail, including the cluster tuning that actually worked, is in [characterization/README.md](characterization/README.md).

Steps 1 and 2 rebuild the corpus from the raw scrape. If you already hold the scored corpus (`data/corpus/reviews_w_detoxify/`), start at step 3 - the figures and the verification read it directly.

```bash
# 1) Corpus: clean games/users/reviews, deduplicate, assign language
cd characterization/step01_cleaning_and_language_detection
python run_clean_games.py \
  --input ../../data/raw/games/todos_jogos.json \
  --output ../../data/corpus/games/games.parquet
python run_clean_users.py \
  --input ../../data/raw/users \
  --output ../../data/corpus/users/all_users.parquet
python run_clean_reviews_dedup_noshuffle.py \
  --input ../../data/raw/reviews \
  --output ../../data/corpus/reviews_deduped.parquet
python run_detect_language.py \
  --input ../../data/corpus/reviews_deduped.parquet \
  --output-dir ../../data/corpus/reviews_by_lang

# 2) Toxicity: Detoxify scoring, model agreement, the calibrated union label
cd ../step02_run_detoxify
python run_detoxify.py \
  --input ../../data/corpus/reviews_by_lang \
  --output-dir ../../data/corpus/reviews_w_detoxify
python run_score_correlation.py \
  --input ../../data/corpus/reviews_w_detoxify \
  --output ../../data/corpus/score_correlation_report.json
python run_toxicity_mask.py \
  --input ../../data/corpus/reviews_w_detoxify \
  --output ../../data/corpus/toxicity_report.json

# 3) Lexicon: TF-IDF toxic vs. non-toxic, then the paper's top-terms table
cd ../step03_tfidf_analysis
python run_tfidf.py \
  --input ../../data/corpus/reviews_w_detoxify \
  --output-dir ../../data/step03-output --lang en
python run_top_terms.py \
  --input ../../data/step03-output/tfidf_lexicon_en.csv \
  --output ../../data/step03-output/top_terms_en.csv

# Threshold calibration (reads the annotation spreadsheets only, no pipeline output)
cd ../annotation_agreement
python run_agreement_table.py \
  --input ../../data/raw/annotations \
  --output ../../data/annotation-output/agreement_table.json

# Figures: run_tag_toxicity.py first, the heatmap reuses its tag list.
# The heatmap streams the corpus rather than loading it - see its README.
cd ../figures
python run_tag_toxicity.py \
  --games ../../data/corpus/games/games.parquet \
  --step02-dir ../../data/corpus/reviews_w_detoxify \
  --output-dir ../../data/figures-output
python run_tag_tfidf_heatmap.py \
  --step02-dir ../../data/corpus/reviews_w_detoxify \
  --fit-cache ../../data/figures-cache/corpus_fit_en.npz \
  --games ../../data/corpus/games/games.parquet \
  --output-dir ../../data/figures-output \
  --tag "Team-Based" --tag "Competitive" --tag "PvP" --tag "FPS" --tag "Shooter" \
  --tag "Military" --tag "Free to Play" --tag "Gore" --tag "Violent" --tag "Comedy" \
  --term sucks --term ass --term trash --term kill --term suck --term stupid \
  --term garbage --term balls --term like --term good --term people --term fun
python run_user_profile.py \
  --users ../../data/corpus/users/all_users.parquet \
  --step02-dir ../../data/corpus/reviews_w_detoxify \
  --output-dir ../../data/figures-output \
  --user-counts ../../data/figures-cache/user_counts_en.parquet
```

Two checks close the loop, and both exit non-zero on failure:

```bash
# Recompute all 26 numbers the characterization section states
python characterization/verify_numbers.py --data data \
    --output results/characterization/corpus_verification.json

# Confirm every figure is authored at the width its \includegraphics uses,
# and that no label lands under AAAI's 7pt floor
python characterization/figures/check_figures.py --dir data/figures-output
```

The figure PDFs land under the filenames `main.tex` expects, so they copy into `main/images/` with no renaming - each at the width its include gives it, so LaTeX scales them by 1.0 and the type sizes survive. `characterization/figures/README.md` records which include width each file is built for.

## Reproducing the prediction benchmark (RQ2, RQ3)

Every stage is a standalone script that reads the artifacts produced by earlier stages and writes into `results/`. Run them from the repository root, in this order. A full stage-by-stage description of inputs and outputs is in [docs/pipeline.md](docs/pipeline.md).

```bash
# 1) Features: review-level and user-level tables from the raw corpus
python src/build_features.py --root . --out data/features --langs en

# 2) Modeling substrate: 10 percent deterministic slice, count/rate labels, CV folds
python src/up_build_dataset.py --kfolds 5

# 3) Text representation: per-review Sentence-BERT cache, then one vector per user
python src/up_embed_reviews.py
python src/up_user_vectors.py

# 4) Classical benchmark: feature-family ablation and the clean leave-toxic-out control
python src/up_train_userlevel.py --kfolds 5 --models logreg,linsvc,histgb,xgb,lgbm,mlp,svc
python src/up_lto_control.py

# 5) Transformer: end-to-end DistilBERT on concatenated user review text
python src/up_bert_user.py --model distilbert-base-uncased --cap 60000 --kfolds 10 --undersample
python src/up_bert_diagnostics.py

# 6) LLMs as classifiers (open-weight locally, proprietary via batch APIs)
python src/up_llm_user.py  --models llama3.1:8b,qwen2.5:7b --blocks 5 --block-size 200
python src/up_llm_batch.py --jobs openai:gpt-4o,claude:claude-opus-4-8

# 7) LLMs as explanation generators, judged by a cross-model panel
python src/up_llm_panel_generate.py --generators opus,gpt4o,qwen,llama
python src/up_llm_panel_judge.py    --judges     opus,gpt4o,qwen,llama

# 8) Consolidation: evaluation views, significance, master table (figures render locally)
python src/up_balanced_eval.py
python src/up_significance.py
python src/up_master_table.py
python src/up_plots.py
```

## Main results

All numbers below preserve the real platform prevalence unless stated otherwise. The full tables live in `results/` and are reproduced with commentary in [docs/results.md](docs/results.md).

### User-level prediction (count label, real prevalence)

| Model family | AUC-PR | ROC-AUC | F1-macro | F1 (toxic) |
| :--- | :---: | :---: | :---: | :---: |
| Profile metadata (LightGBM) | 0.070 | 0.637 | 0.533 | 0.101 |
| Content embeddings (XGBoost) | 0.476 | 0.932 | 0.721 | 0.463 |
| Profile + content (XGBoost) | 0.474 | 0.932 | 0.719 | 0.460 |
| DistilBERT, end to end | 0.643 | 0.968 | 0.794 | 0.605 |
| Claude Opus, classifier | 0.408 | 0.921 | 0.684 | 0.389 |
| GPT-4o, classifier | 0.313 | 0.688 | 0.692 | 0.405 |
| Llama-3.1-8B, classifier | 0.308 | 0.818 | 0.643 | 0.344 |
| Qwen-2.5-7B, classifier | 0.268 | 0.888 | 0.614 | 0.305 |
| Rate label (MPNet + linear SVM) | 0.486 | 0.841 | 0.700 | 0.483 |

Sources: [classical fold metrics](results/replication/metrics_cv.csv), [DistilBERT fold metrics](results/replication_bert_undersampled/bert_user_metrics.csv), and [LLM block metrics](results/lente4_llm/llm_user_metrics.csv). See [results provenance](docs/results.md) before using retrospective threshold-dependent values from `master_table.csv`.

### Leave-toxic-out control

| Model | Metadata | Leave-toxic-out | Content | LTO minus metadata (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 0.063 | 0.166 | 0.425 | +0.102 [+0.098, +0.108] |
| HistGB | 0.067 | 0.171 | 0.397 | +0.104 [+0.099, +0.109] |
| Logistic regression | 0.057 | 0.053 | 0.137 | -0.004 [-0.006, -0.003] |

The boosted-tree arms stay clearly above the metadata baseline even after every toxic review is removed, and the trace is nonlinear (logistic regression gains nothing). Source: [results/replication/lto_control.csv](results/replication/lto_control.csv).

### Statistical tests

Paired tests over the exact users shared by the compared models, with Holm family-wise correction. All four confirmatory comparisons reach the attainable bootstrap floor of p = 0.004.

| Comparison | Metric | Delta (95% CI) |
| :--- | :--- | :---: |
| Content embeddings vs. metadata | AUC-PR | +0.406 [+0.388, +0.425] |
| DistilBERT vs. content XGBoost | AUC-PR | +0.108 [+0.083, +0.134] |
| DistilBERT balanced vs. real prevalence | F1-macro | +0.118 [+0.115, +0.122] |
| Claude Opus vs. GPT-4o narratives | judge overall | +0.306 [+0.125, +0.500] |

These four confirmatory comparisons are transcribed from the accepted paper. The supplied `pairwise_bootstrap.csv` contains different, within-model feature comparisons; the original confirmatory analysis and its aggregate export remain to be recovered. See [the evidence gap](docs/results.md#statistical-tests).

## Limitations

The measurements rest on tool-derived labels. Toxicity is operationalized through two automatic detectors, Perspective API and Detoxify, whose thresholds were calibrated on 400 reviews labeled by three annotators, so the labels inherit both tools' error profiles. The primary count label is sensitive to activity volume, since more active users have more chances to produce at least one toxic review; the complementary rate label mitigates but does not remove this. Content-based models are partially circular because they read the reviews that define the label, which is exactly why the leave-toxic-out control matters. The corpus is English-only, from a single collection window, and covers the 43.7 percent of users whose histories could be matched to a public profile. The LLM classifier comparison rests on 1,000 users with only 40 positives, so its estimates are wide, and inter-judge agreement in the narrative panel is weak. See [docs/ethics.md](docs/ethics.md) for the full treatment.

## Citation

The paper has been accepted at AIIDE 2026. Page numbers and DOI will be added when the proceedings are published.

```bibtex
@inproceedings{toxicsteam2026,
  title     = {Toxic Steam: A Large-Scale Characterization and Prediction of Toxic Users in a Gaming Platform},
  author    = {Gibrim, Paula T. M. and Lodi, Vinicius K. and Andrade, Gabriel S. and Ribeiro, Marcus V. G. and Gruppi, Maur{\'i}cio and Barbosa, Daniel M. and Melo, Philipe F. and Reis, Julio C. S.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Preserve its copyright and permission notice when redistributing it. Please cite the paper when using this work in research. The software license does not grant rights to the excluded Steam corpus or third-party review text.
