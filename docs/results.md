# Results

All results preserve the real platform prevalence unless a table says otherwise. Cross-validation is five-fold for the classical models and ten-fold for DistilBERT, with folds shared across models so that comparisons are paired. Thresholds are tuned inside the training portion of each fold and applied unchanged to the evaluated partition. AUC-PR is the anchor metric because the task is rare-positive ranking under about 4 percent prevalence.

The machine-readable versions of every table below are in `results/`.

## User-level prediction, count label

Reported as mean plus or minus standard deviation across folds. `F1+`, `Prec.+`, and `Rec.+` refer to the toxic-user class.

| Model family | Label | AUC-PR | ROC-AUC | F1-macro | F1+ | Prec.+ | Rec.+ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Profile metadata (LightGBM) | count | 0.070 | 0.637 | 0.533 | 0.101 | 0.107 | 0.096 |
| Content embeddings (XGBoost) | count | 0.476 | 0.932 | 0.721 | 0.463 | 0.483 | 0.444 |
| Profile + content (XGBoost) | count | 0.474 | 0.932 | 0.719 | 0.460 | 0.480 | 0.442 |
| DistilBERT, end to end | count | 0.643 | 0.968 | 0.794 | 0.605 | 0.588 | 0.635 |
| Claude Opus, classifier | count | 0.408 | 0.921 | 0.684 | 0.389 | 0.591 | 0.300 |
| GPT-4o, classifier | count | 0.313 | 0.688 | 0.692 | 0.405 | 0.547 | 0.325 |
| Llama-3.1-8B, classifier | count | 0.308 | 0.818 | 0.643 | 0.344 | 0.232 | 0.675 |
| Qwen-2.5-7B, classifier | count | 0.268 | 0.888 | 0.614 | 0.305 | 0.192 | 0.750 |
| Rate label (MPNet + linear SVM) | rate | 0.486 | 0.841 | 0.700 | 0.483 | 0.452 | 0.519 |

Files: `results/replication/master_table.csv`, `results/replication/metrics_agg.csv`, `results/lente4_llm/llm_user_metrics.csv`, `results/replication_bert_undersampled/bert_user_metrics.csv`.

Profile metadata alone is a weak signal (AUC-PR 0.070), and adding public ban indicators does not change it, which suggests visible enforcement is not tightly coupled with review-level toxicity. Text-derived representations improve prediction sharply. Replacing MiniLM with MPNet leaves the content result essentially unchanged (AUC-PR 0.479), so the finding is not specific to one encoder. DistilBERT gives the strongest ranking among supervised runs; a parallel MPNet-base fine-tuning reaches a comparable AUC-PR of 0.639, indicating the gain comes from reading review text end to end rather than from a specific backbone.

## Leave-toxic-out control

Paired percentile-bootstrap 95 percent confidence intervals (B = 2000), on the population of users with at least one non-toxic review.

| Model | Metadata | Leave-toxic-out | Content | LTO minus metadata | Content minus LTO |
| :--- | :---: | :---: | :---: | :---: | :---: |
| XGBoost | 0.063 | 0.166 | 0.425 | +0.102 [+0.098, +0.108] | +0.259 [+0.252, +0.266] |
| HistGB | 0.067 | 0.171 | 0.397 | +0.104 [+0.099, +0.109] | +0.226 [+0.220, +0.232] |
| Logistic regression | 0.057 | 0.053 | 0.137 | -0.004 [-0.006, -0.003] | +0.085 [+0.082, +0.087] |

File: `results/replication/lto_control.csv`.

This is the central diagnostic. Even after every toxic review is removed before building the user embedding, boosted-tree models stay clearly above the metadata baseline, and the trace is nonlinear (logistic regression gains nothing). At the same time, full-content embeddings keep a large advantage over leave-toxic-out embeddings, so most of the predictive power is circular with respect to the label, and the non-circular residual, while statistically reliable, is substantially weaker.

## Real prevalence versus balanced evaluation

| Model | Real F1-macro | Balanced F1-macro | Real AUC-PR | Balanced AUC-PR |
| :--- | :---: | :---: | :---: | :---: |
| DistilBERT | 0.794 | 0.912 | 0.643 | 0.964 |
| XGBoost content (MiniLM) | 0.721 | 0.856 | 0.476 | 0.926 |
| XGBoost combined (MiniLM) | 0.719 | 0.856 | 0.474 | 0.926 |
| XGBoost content (MPNet) | 0.722 | 0.857 | 0.479 | 0.927 |
| Claude Opus | 0.684 | 0.596 | 0.408 | 0.896 |
| GPT-4o | 0.692 | 0.614 | 0.313 | 0.690 |
| Llama-3.1-8B | 0.643 | 0.782 | 0.308 | 0.794 |
| Qwen-2.5-7B | 0.614 | 0.807 | 0.268 | 0.852 |

File: `results/replication/balanced_eval.csv`.

Balancing the evaluation set inflates apparent performance considerably. It is useful for comparison with case-control and balanced protocols common in prior work, but it should not be read as deployment performance on the real Steam population, for which the real-prevalence setting stays primary.

## Statistical tests

Non-parametric paired tests over the exact users shared by the compared models, with Holm family-wise correction across the four confirmatory tests. Bootstrap p-values are floored at 1/B, and because every test reaches its floor, the Holm-adjusted values coincide at p = 0.004.

| Comparison | Metric | N | Delta (95% CI) | Holm p |
| :--- | :--- | :---: | :---: | :---: |
| Content embeddings vs. metadata | AUC-PR | 620,891 | +0.406 [+0.388, +0.425] | 0.004 |
| DistilBERT vs. content XGBoost | AUC-PR | 60,000 | +0.108 [+0.083, +0.134] | 0.004 |
| DistilBERT balanced vs. real prevalence | F1-macro | 10 | +0.118 [+0.115, +0.122] | 0.004 |
| Claude Opus vs. GPT-4o narratives | judge overall | 24 | +0.306 [+0.125, +0.500] | 0.004 |

File: `results/replication/pairwise_bootstrap.csv`.

The paired deltas cannot be reproduced by subtracting rows of the main table. The DistilBERT comparison uses the 60,000 users shared with the fine-tuning run and pooled out-of-fold scores, whose pooled AUC-PR of 0.583 is lower than the per-fold mean of 0.643 because each fold trains a separate network on a different scale, so the reported delta is conservative.

## LLM classifiers and narrative panel

As direct classifiers, Claude Opus leads the LLMs under real prevalence (AUC-PR 0.408), followed by GPT-4o, Llama, and Qwen, but with only eight positive users per evaluation block the ordering among the weaker three is not statistically stable, and these results are read separately from the classical benchmark.

The narrative panel is where LLMs add the most value. Across 96 narratives, mean overall judge scores from non-self judgments are Claude Opus 3.718, GPT-4o 3.417, Qwen 3.099, and Llama 3.056. Opus also leads on faithfulness and clarity. Inter-judge agreement is weak and uneven, with pairwise Spearman correlations on the overall score ranging from 0.49 down to -0.27, so LLM-based evaluation is treated as diagnostic rather than as ground truth.

Files: `results/lente4_llm/panel/judgments.csv`, `results/lente4_llm/panel/narratives.jsonl`.
