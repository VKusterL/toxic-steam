# Toxic Steam — Characterization Pipeline

This folder holds the code behind the **characterization** half of *"Toxic
Steam: A Large-Scale Characterization and Prediction of Toxic Users in a
Gaming Platform"* — everything under the paper's *Toxicity
Characterization* section, up to and including the behavioral profile of
Steam users. The user-level **prediction** benchmark (classical models,
DistilBERT, the LLM classifier and narrative panel) lives in the main
repository, not here.

> **Content warning.** This pipeline processes and surfaces examples of
> offensive and hateful language. The example-sampling tool in
> `review_examples/` exists specifically to print them.

## What each folder reproduces

| Folder | What it produces in the paper |
|---|---|
| `step01_cleaning_and_language_detection/` | The working corpus: cleaned games/users/reviews, deduplicated, language assigned by `langdetect`. Yields 36,823,127 English reviews from 14,183,630 users across 67,477 titles. |
| `step02_run_detoxify/` | Detoxify scores alongside the Perspective API scores; the Pearson/Spearman agreement between the two models; the calibrated union rule (`detoxify >= 0.9` **OR** `perspective >= 0.7`) that yields 685,536 toxic reviews, about 1.86% of the corpus. |
| `step03_tfidf_analysis/` | Table 2 — the TF-IDF term weights for the regular and toxic trends, and the discriminative ratios quoted in the text. |
| `annotation_agreement/` | Table 1 — inter-annotator agreement per toxicity bin, per model. Three human annotators, 200 reviews per model. This is what calibrated the thresholds step02 applies. |
| `review_examples/` | The reviews quoted verbatim in the text, with their game names. |
| `figures/` | The three figures of the characterization section - top-10 tags by toxicity rate, the per-tag TF-IDF heatmap, and the toxic-vs-non-toxic engagement CDFs - plus the behavioral statistics quoted with them (group medians, ban rates, and the 57.5% of toxic reviews that still recommend the game). |

The three numbered steps run in order, and each reads the previous one's
output. They were renumbered for this release: the TF-IDF stage was step05
in the original pipeline, which also carried a topic-modeling and a
sentiment-analysis stage between it and Detoxify. Neither feeds this paper,
so both were removed and the numbering closed up. Output directories follow
the code (`step03-output`, not `step05-output`), so a lab artifact produced
before this release will carry the old name.

## Pipeline order

Each stage reads the previous stage's output. Every command is written
relative to its own step folder as `../../steam-data/...`, which resolves
to a `steam-data/` sitting next to this folder:

```
<repo>/
  characterization/        this folder
    step01_.../            commands run from here, so ../.. is <repo>/
    step02_.../
    ...
  steam-data/              not distributed - see Data availability below
    raw/
    step01-output/
    step02-output/
    ...
```

Nothing needs editing per machine as long as that layout holds.

```
raw scrape
    |
    v
step01  clean games / clean users / dedup reviews / detect language
    |        -> step01-output/games/games.parquet
    |        -> step01-output/users/all_users.parquet
    |        -> step01-output/reviews_by_lang/reviews_cleaned.parquet/
    v
step02  Detoxify scoring -> score correlation -> toxicity labeling
    |        -> step02-output/*.parquet  (flat; review_lang is a column)
    |        -> score_correlation_report.json, toxicity_report.json
    v
step03  TF-IDF toxic vs. non-toxic -> top terms per class
    |        -> tfidf_lexicon_<lang>.csv, top_terms_<lang>.csv
    v
figures tag toxicity -> per-tag TF-IDF heatmap; user CDFs
             -> top-10-tags-tox.pdf, heatmap_tfidf_tags.pdf,
                cdf-*.pdf, and the CSV/JSON behind each one

annotation_agreement   reads the raw annotation spreadsheets only
review_examples        reads step01's games + step02's scores
```

Within `figures/`, `run_tag_toxicity.py` runs before
`run_tag_tfidf_heatmap.py` (the heatmap reuses its tag list so both figures
describe the same ten tags); `run_user_profile.py` is independent of both.

`step01` and `step02` are heavy. Review dedup and language detection want a
many-core machine; Detoxify wants a GPU. `step03`, `figures`,
`annotation_agreement` and `review_examples` run anywhere (CPU-only). Each folder's
own README carries the exact commands, the tuning that actually worked, and
why.

## Language

The paper reports **English only**. The scripts stay
language-parameterized and still accept `pt`, but `--lang en` is what
reproduces the published numbers. `annotation_agreement/` processes every
language subfolder it finds; the table printed in the paper is the English
one.

## Setup

Python 3.13. Dependencies are per-step rather than global, because the
heavy steps are meant to be copied to different machines:

```bash
pip install -r requirements.txt                    # step01 (dask, langdetect)
pip install -r step02_run_detoxify/requirements.txt # torch, detoxify
pip install -r step03_tfidf_analysis/requirements.txt
pip install -r annotation_agreement/requirements.txt
pip install -r review_examples/requirements.txt
pip install -r figures/requirements.txt   # matplotlib
```

## Numbers quoted in these READMEs

The step-level READMEs record measurements from the runs that produced
them - correlation coefficients, per-language row counts, agreement
percentages, timings. **The paper is authoritative for every published
result.** The corpus behind this folder went through later modifications,
so a figure noted here can differ slightly from the corresponding value in
the paper; where the two disagree, the paper's is the reported one.

These notes are kept rather than trimmed because they document *why* a
given decision was made - which is what a reviewer re-running the pipeline
needs - not because they restate the paper's tables.

## Data availability

The raw review, user, and game data are **not distributed**. They were
collected from the public Steam Web API between September 2024 and April
2025, are subject to Steam's Terms of Service, and describe users who did
not consent to toxicity profiling. Running this pipeline end to end
requires re-collecting that corpus into `../steam-data/raw/`
(`games/`, `users/`, `reviews/`, and `annotations/<lang>/*.xlsx` for the
agreement table). Any re-collection should pseudonymize Steam identifiers
and profile URLs.

## Figures

`figures/` writes the PDFs under the exact filenames `main.tex` expects
(`top-10-tags-tox.pdf`, `heatmap_tfidf_tags.pdf`, `cdf-reviews-per-user.pdf`,
`cdf-library-size.pdf`, `cdf-steam-level.pdf`, `cdf-legend.pdf`), so
regenerating them is a copy into `main/images/` with no renaming. Each
script also writes the numbers behind its figure as CSV and JSON, so any
value quoted in the text can be checked without reading it off a plot.
