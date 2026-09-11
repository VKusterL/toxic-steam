# Figures

Regenerates the three figures of the paper's characterization section and
the descriptive statistics quoted alongside them.

Not a pipeline stage like `step01`-`step03` - a cross-cutting tool that
reads *outputs* from several of them, so it doesn't fit under any single
step's number. Everything here is CPU-only and runs in minutes once the
pipeline outputs exist.

Output filenames match what `main.tex` expects, so the PDFs drop straight
into `main/images/` with no renaming.

**The design is not a choice made here.** Colors, font, font sizes, line
widths, dash patterns and canvas sizes were read back out of the published
PDFs' content streams (they were produced by matplotlib 3.10.7, which
records all of it) and are reproduced exactly, so a regenerated figure is
indistinguishable from the one already in the paper. The values and why
they are pinned are in `plotting.py`'s module docstring. Do not adjust
them: a figure that differs from its neighbours in the same paper reads as
an error to a reviewer even when its numbers are right.

## What each script produces

| Script | Paper artifact | Reads |
|---|---|---|
| `run_tag_toxicity.py` | `top-10-tags-tox.pdf` - top-10 game tags by toxicity rate | step01 games + step02 scores |
| `run_tag_tfidf_heatmap.py` | `heatmap_tfidf_tags.pdf` - mean TF-IDF salience across tag contexts | step01 games + step03 outputs |
| `run_user_profile.py` | `cdf-reviews-per-user.pdf`, `cdf-library-size.pdf`, `cdf-steam-level.pdf`, `cdf-legend.pdf`, plus the medians, ban rates and the share of toxic reviews that still recommend the game | step01 users + step02 scores |

Every script also writes the underlying numbers as CSV/JSON next to the
figure, so a value quoted in the text can be checked without reading it off
a plot.

## Order

`run_tag_toxicity.py` must run before `run_tag_tfidf_heatmap.py`: the
heatmap reads `tag_toxicity_top.csv` so that both figures describe the same
ten tags. `run_user_profile.py` is independent of both.

## Setup

```bash
pip install -r requirements.txt
```

`pandas`, `pyarrow`, `numpy`, `matplotlib`, and - for the heatmap only -
`scikit-learn` and `nltk`, which come in through step03's `tfidf_analysis`.

## 1. Tag toxicity rates

```bash
python run_tag_toxicity.py \
  --games ../../steam-data/step01-output/games/games.parquet \
  --step02-dir ../../steam-data/step02-output \
  --output-dir ../../steam-data/figures-output
```

- `--volume-quantile` (default `0.90`) - the review-volume percentile a tag
  must reach to be eligible. **This floor is not cosmetic.** A toxicity rate
  is a ratio, so a tag carried by three games with eleven reviews between
  them reaches 30% on three toxic reviews; ranking by rate without a floor
  returns those, not the competitive cluster the paper reports.
- `--top-n` (default `10`), `--lang` (default `en`).

Each tag is one horizontal bar split into its non-toxic (green) and toxic
(red) review counts, with the toxicity rate overlaid as a black line on a
second x-axis at the top. The two quantities differ by orders of magnitude
- millions of reviews against single-digit percentages - so a shared axis
would flatten the rate into the baseline. The bar carries the volume that
makes a rate trustworthy; the line carries the rate. The bottom axis is in
millions with a decimal comma, matching the published figure.

Writes `top-10-tags-tox.pdf`, `tag_toxicity_top.csv` (the plotted tags,
also the heatmap's input), `tag_toxicity_all.csv` (**every** tag - this is
where the low end quoted in the text, Exploration/Puzzle/Cute, is readable,
since the figure only shows the top) and `tag_toxicity_report.json`.

Counting is per `(game, tag)` pair: a game tagged both `PvP` and `Free to
Play` contributes its full review count to both. Tags are not mutually
exclusive on Steam, and the paper's reading of the Free to Play tag - that
its toxicity is partly inherited from co-occurring competitive tags -
depends on that overlap being preserved rather than divided up. It also
means the tag table's columns do **not** sum to the corpus total;
`tag_coverage` in the report gives the honest denominator, measured over
distinct games.

## 2. Per-tag TF-IDF heatmap

```bash
python run_tag_tfidf_heatmap.py \
  --labeled ../../steam-data/step03-output/reviews_cleaned_labeled_en.parquet \
  --lexicon ../../steam-data/step03-output/tfidf_lexicon_en.csv \
  --games ../../steam-data/step01-output/games/games.parquet \
  --top-tags ../../steam-data/figures-output/tag_toxicity_top.csv \
  --output-dir ../../steam-data/figures-output
```

Rows are the terms the lexicon ranks as most disproportionately toxic
(`--n-terms`, default 10), so a reader meets the same terms here as in the
paper's TF-IDF table; `--term` (repeatable) overrides that with an explicit
set. Columns are the tags from `--top-tags`, or `--tag` (repeatable).

**This is the one script in this folder that reaches into another step's
code.** `--step03-code` (default `../step03_tfidf_analysis`) is added to
`sys.path` so `tfidf_analysis` can be imported. A TF-IDF weight only means
something relative to the vocabulary and document frequencies it was fitted
against, so for a cell here to be comparable to the published table, both
must come from a vectorizer fitted the same way, on the same corpus, with
the same stopword list, `min_df`, `max_df` and `max_features`. Copying
those settings into this folder would make them two constants that have to
be kept equal by hand; importing makes them one.

The vectorizer is therefore re-fitted on the full language corpus, toxic
and non-toxic alike, exactly as `run_tfidf.py` does, and only then applied
per tag. Fitting per tag instead would give every tag its own document
frequencies and make the columns incomparable - which is the comparison the
figure exists to make. Re-fitting is the slow part of this script.

The color scale is shared across the whole grid for the same reason: a
per-column normalization would make every tag's most salient term equally
dark and erase the contrast the figure reports. Because it is shared, the
light end is hard to read off color alone, so every cell is annotated with
its own value - which is also how the published figure does it. Rows run
weakest-at-top to strongest-at-bottom.

Behind the grid, `CLUSTER_BANDS` shades runs of adjacent columns by tag
family - the competitive cluster, the violence-tagged pair, and Comedy - at
the published alpha of 0.15. That is faint enough to tint only the white
gaps between cells and never a cell itself, so it groups the columns
without touching a single value. Membership is editorial, not computed: it
is the grouping the paper argues for, so it is written down in the module
rather than inferred. A tag absent from the map gets no band, and a run
that draws none says so in the log.

## 3. User behavioral profile

```bash
python run_user_profile.py \
  --users ../../steam-data/step01-output/users/all_users.parquet \
  --step02-dir ../../steam-data/step02-output \
  --output-dir ../../steam-data/figures-output \
  --user-counts ../../steam-data/cache/user_counts_en.parquet
```

This is the heaviest script here - it reduces every English review to one
row per user before anything is plotted. `--user-counts` caches that
aggregate and is read back on later runs, so tuning a figure does not mean
re-reading 36.8M reviews.

A user is toxic when they authored at least one review meeting the
calibrated criterion, the same count rule used everywhere else in this
project. That rule is sensitive to activity volume by construction, which
is exactly why the reviews-per-user panel is reported alongside the other
two rather than quietly omitted: it is the panel that shows the confound.

### Two populations, deliberately

Reviews per user is defined for **every** user in the corpus. Library size
and profile level exist only for users whose review URL carries a SteamID64
*and* whose profile was collected - about 43.7% of them, which is why the
paper's matched population is 6.2M rather than 14.1M. Vanity URLs
(`/id/<name>`) cannot be joined to a profile without a resolution step this
project never ran.

Mixing the two would silently change the denominator between panels of the
same figure, so they are computed separately and both population sizes are
written to the report. Ban rates are restricted to matched profiles for the
same reason: an unmatched user is not an unbanned user, and counting them
as one would push both rates toward zero by roughly the same unmatched
share, making the comparison look tighter than it is.

### Log axes and zeros

The panels use log x-axes, since all three metrics are heavy-tailed, and
match the published axis labels verbatim. A log axis cannot render 0 (an
empty library, account level 0), so those points are absent from the drawn
curve, exactly as in the published panels.

Medians are computed over the raw values *before* plotting, so they are not
affected by that, and each run reports how many points each group lost
under `<group>_zeros_not_drawn` - a number worth glancing at, because it is
invisible in the figure itself.
