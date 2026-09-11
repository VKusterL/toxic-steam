# Review Examples

Generates example-message files for manual inspection - sampled reviews
matching a language + toxicity + (optionally) a text term / game tag, with
the perspective/detoxify scores and the game name joined in.

This is how the reviews quoted verbatim in the paper's characterization
section were pulled: the low-complexity hostility examples, the
contextual-toxicity cases built around `people`, the performative uses of
`ass` and `kill`, and the per-tag illustrations for `Gore`, `Violent`,
`Comedy` and the competitive cluster.

Not a pipeline stage like `step01`-`step03` - a cross-cutting tool that
reads *outputs* from two of them (games from step01, scores from step02),
so it doesn't fit under any single step's number. Self-contained, doesn't
depend on any step folder being present - only their output files.

## What it pulls from where

| column | source |
|---|---|
| `game_id`, `review_url`, `review_text`, `review_lang`, `perspective_score`, `detoxify_score`, `user_url`, `review_date`, `is_recommended`, `hours_played`, `detection_confidence` | step02's output |
| `game_name` | step01's `games.parquet` (joined by `game_id`) |
| `review_text_clean` | this tool - the original text with Steam's early-access/refund boilerplate stripped, i.e. what the models actually scored. `review_text` itself is never altered. |

Toxicity labeling uses the same union rule and thresholds as everywhere
else in this project (`perspective_score >= 0.7` OR `detoxify_score >=
0.9`), with rows carrying an invalid/sentinel score excluded before
filtering (not labeled non-toxic).

## Setup

```bash
pip install -r requirements.txt
```

Just `pandas` and `pyarrow` - no models, no GPU, runs anywhere.

## Running it

```bash
python run_show_examples.py \
  --lang en --toxic --n 50 \
  --games ../../steam-data/step01-output/games/games.parquet \
  --step02-dir ../../steam-data/step02-output \
  --output ../../steam-data/examples/toxic_en.csv
```

With the optional term and tag filters - this is the shape of call that
produced the paper's per-tag examples:

```bash
python run_show_examples.py \
  --lang en --toxic --n 50 \
  --games ../../steam-data/step01-output/games/games.parquet \
  --step02-dir ../../steam-data/step02-output \
  --contains "kill" --game-tag "Gore" --seed 42 \
  --output ../../steam-data/examples/toxic_en_kill_gore.csv
```

- `--lang` - language code. The paper reports `en` throughout.
- `--toxic` / `--non-toxic` - required, mutually exclusive.
- `--n` - how many examples to sample (returns fewer if not enough match).
- `--games` / `--step02-dir` - optional; both default to this project's
  standard layout, derived from `--lang` (see `_default_paths`).
- `--contains` - optional substring the review text must contain
  (case-insensitive, plain substring match, not regex).
- `--game-tag` - optional; game must have this tag in its `popular_tags`
  (case-insensitive).
- `--seed` - random seed. **Pass it for anything meant to be reproducible** -
  without it the sample differs on every run.
- `--light-mode` - filters each file as it is read instead of loading the
  whole language into memory first. Same output, much lower peak RAM.

## Using it from code directly

```python
from show_review_examples import get_review_examples

examples = get_review_examples(
    lang="en", toxic=True, n=50,
    games_path="../../steam-data/step01-output/games/games.parquet",
    step02_dir="../../steam-data/step02-output",
    contains="kill", game_tag="Gore", seed=42,
)
```

Returns a `pandas.DataFrame`, same columns as the CSV output.
