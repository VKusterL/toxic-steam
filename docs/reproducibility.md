# Reproducibility

## Determinism

- The base seed is 42 for dataset sampling, cross-validation, and most model fitting and bootstrap operations. Some procedures derive fold/block seeds from that base, and threshold-search subsampling uses a separate seed. These settings improve repeatability but do not guarantee bitwise-identical results across hardware and package versions.
- The 10 percent modeling slice is selected by `hash(user_key) % 10 == 0` using DuckDB's built-in string hash. Reuse the pinned DuckDB version, original key representation, and corpus to reproduce membership; changing identifiers before selection changes the slice. The observed prevalence is reported rather than enforced by this hash sample.
- Cross-validation folds are stratified and shared within a model comparison. Use the explicit fold settings in [pipeline.md](pipeline.md): current script defaults do not all match the stored experiments.
- In the classical training script, preprocessing and threshold selection use the training fold. The canonical DistilBERT run selects its threshold on an internal validation split. The later `up_master_table.py` and `up_balanced_eval.py` scripts also include retrospective threshold optimization on evaluated labels; their threshold-dependent metrics must be distinguished from those original held-out evaluations.
- LLM calls use temperature 0 where implemented and supported. Stored metrics quantify variation across evaluation blocks and resamples, not variation across independent generations. Proprietary model updates and backend nondeterminism can still change a rerun; no repeat-generation stability estimate is supplied.

## Environment

The study was run with Python 3.13 (Anaconda) on Windows 11. Recorded package versions are pinned in `requirements.txt`. TensorBoard is additionally declared because the transformer script imports `SummaryWriter`; its original run version was not recorded, so its declared range is not an exact environment lock.

Reference hardware: an Intel CPU with an NVIDIA GeForce GTX 1070 (8 GB, Pascal, compute capability 6.1) and 17 GB of RAM. Embedding and transformer scripts use CUDA when available and otherwise fall back to CPU; GPU acceleration is recommended for their full-corpus workloads. Classical models and consolidation run on CPU. The characterization pipeline has separate dependencies and resource requirements documented in [its README](../characterization/README.md).

The torch wheel is the CUDA 12.4 build (`torch==2.6.0+cu124`). To reproduce the GPU environment, install torch from the CUDA index before installing the rest of the requirements. On Windows with Anaconda, set `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch to avoid an OpenMP library conflict; the pipeline scripts set this automatically.

## External services

- Ollama serves the open-weight LLMs (`llama3.1:8b` and `qwen2.5:7b`) locally on port 11434.
- The Anthropic API serves Claude Opus, through the `ANTHROPIC_API_KEY` environment variable.
- The OpenAI API serves GPT-4o for the classifier and judge roles, through the `OPENAI_API_KEY` environment variable.

Re-executing the API stages requires service access and sends review text to the selected provider. The scripts record model names but do not pin all backend revisions. Inspecting the supplied aggregate results requires no API credentials or model downloads.

## Scope of the supplied artifact

The publication export supports source inspection and checking stored aggregate results. It does not include the raw corpus, review text, model checkpoints, per-review embeddings, modeling table, or classical OOF predictions. Full end-to-end execution therefore requires the private corpus and intermediate outputs described in [data.md](data.md) and [pipeline.md](pipeline.md). The lightweight validation path and the distinction between aggregate verification and rerunning the study are documented in [artifact.md](artifact.md).

## Notes on cross-setting comparison

The classical models use five-fold cross-validation over the full slice, while DistilBERT uses ten-fold over a capped 60,000-user subsample, following the data regime of each. Because the two settings differ in fold size and composition, their per-model fold dispersions are not directly comparable, and no claim is made that one configuration would raise or lower a given model's score. For this reason, model comparisons are not read off the reported means but are assessed through the paired tests computed over the exact set of users shared across the compared models.
