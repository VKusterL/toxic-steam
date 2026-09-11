# Reproducibility

## Determinism

- A single global seed of 42 is used across data splitting, cross-validation, model fitting, and bootstrap resampling.
- The 10 percent modeling slice is selected by `hash(user_key) % 10 == 0` using DuckDB's built-in 64-bit string hash. Because the hash is independent of user behavior, the slice is a uniform sample that preserves the real prevalence, and it is exactly reproducible without storing a seed or an identifier list. The same predicate selects the same users at every stage.
- Cross-validation folds are stratified and shared across models, so model comparisons are paired over identical users.
- Preprocessing and decision-threshold selection happen strictly inside each training fold, and the chosen threshold is applied unchanged to the evaluated partition, so no information from the evaluated data leaks into the decision rule.
- LLM calls use temperature 0 where the backend allows it, so the variance reported for LLMs comes from the evaluation blocks and bootstrap resampling rather than from re-sampling the model.

## Environment

The study was run with Python 3.13 (Anaconda) on Windows 11. The pinned package versions are in `requirements.txt` and reflect the versions actually used, verified through `importlib.metadata`.

Reference hardware: an Intel CPU with an NVIDIA GeForce GTX 1070 (8 GB, Pascal, compute capability 6.1) and 17 GB of RAM. A GPU is only required for the embedding and transformer stages; the classical models and the consolidation stages run on CPU.

The torch wheel is the CUDA 12.4 build (`torch==2.6.0+cu124`). To reproduce the GPU environment, install torch from the CUDA index before installing the rest of the requirements. On Windows with Anaconda, set `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch to avoid an OpenMP library conflict; the pipeline scripts set this automatically.

## External services

- Ollama serves the open-weight LLMs (`llama3.1:8b` and `qwen2.5:7b`) locally on port 11434.
- The Anthropic API serves Claude Opus, through the `ANTHROPIC_API_KEY` environment variable.
- The OpenAI API serves GPT-4o for the classifier and judge roles, through the `OPENAI_API_KEY` environment variable.

## Notes on cross-setting comparison

The classical models use five-fold cross-validation over the full slice, while DistilBERT uses ten-fold over a capped 60,000-user subsample, following the data regime of each. Because the two settings differ in fold size and composition, their per-model fold dispersions are not directly comparable, and no claim is made that one configuration would raise or lower a given model's score. For this reason, model comparisons are not read off the reported means but are assessed through the paired tests computed over the exact set of users shared across the compared models.
