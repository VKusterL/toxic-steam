# Figures

Regenerates the three figures of the paper's characterization section and
the descriptive statistics quoted alongside them.

Not a pipeline stage like `step01`-`step03` - a cross-cutting tool that
reads *outputs* from several of them, so it doesn't fit under any single
step's number. Everything here is CPU-only and runs in minutes once the
pipeline outputs exist.

Output filenames match what `main.tex` expects, so the PDFs drop straight
into `main/images/` with no renaming.

**Colors and structure are not a choice made here.** They were read back
out of the published PDFs' content streams (matplotlib records all of it)
and are reproduced exactly: `#2ca02c` for non-toxic, `#d62728` for toxic,
Times New Roman, the stacked-bar-plus-rate-line construction, the annotated
heatmap with its tag-family bands. A figure that differs from its
neighbours in the same paper reads as an error even when its numbers are
right.

## Every figure is authored at the size it will be printed

This is the rule the first submission broke, and the reason AAAI's editors
rejected Figures 1 and 2. Those figures were drawn on large canvases and
then pulled into a 3.3125in column by LaTeX; the scale factor rode straight
through to the type:

| figure | canvas | included at | scale | label size on the page |
|---|---|---|---|---|
| Figure 1 | 15.90in | `\columnwidth` (3.3125in) | 0.21 | **4.17pt** |
| Figure 2 | 6.64in | `\columnwidth` (3.3125in) | 0.50 | **4.99pt** |
| Figure 3 | 7.45in | 0.30`\linewidth` (2.1in) | 0.28 | 7.04pt |

AAAI asks for 10pt and forbids anything under 7pt, so the first two failed
and the third scraped by - which is exactly the pair the editors named.

Every canvas is now declared in the units LaTeX will use, so the scale is
1.0 and a 9pt label is 9pt on the page. `plotting.py` holds the geometry
(`\textwidth` 7.0in, `\columnwidth` 3.3125in, both from `aaai24.sty`),
`save_figure` refuses to write a figure whose canvas does not match the
width its include expects, and `assert_compliant` fails the run if any
declared size falls under 7pt.

**These are the include widths each file is built for.** Changing one
without changing the other silently reintroduces the original bug:

| file | canvas | must be included at |
|---|---|---|
| `top-10-tags-tox.pdf` | 3.3125 x 2.25in | `width=\columnwidth` in a `figure` |
| `heatmap_tfidf_tags.pdf` | 7.00 x 2.45in | `width=\textwidth` in a `figure*` |
| `cdf-*.pdf` (3 panels) | 2.10 x 1.50in | `width=\linewidth` in a 0.30`\linewidth` subfigure of a `figure*` |
| `cdf-legend.pdf` | 3.15 x 0.24in | `width=0.45\linewidth` |

Two traps worth knowing, both handled in `plotting.py`:

- **`bbox_inches='tight'` is not used.** It trims the canvas to its
  content, so the saved PDF comes out narrower than the figsize that was
  requested, and LaTeX scales it back up to the column - undoing the whole
  guarantee. Constrained layout fits the content to the canvas instead.
- **Mathtext superscripts are 0.7 of the base size, and the floor applies
  to them too.** A log axis labelled `10^4` at a 9pt base puts the exponent
  at 6.3pt, under the floor, in a figure whose every other label passes.
  The CDF panels therefore run at 10pt, which lands the exponent at exactly
  7.0pt.

## Checking the figures before submitting

`assert_compliant` guards the sizes the code declares, but the rule the
editors enforce is a property of the printed page. `check_figures.py` reads
it back out of the PDFs:

```bash
python characterization/figures/check_figures.py --dir data/figures-output
```

For each figure it prints the canvas width, the width its include gives it,
the resulting scale, and what the smallest text in the file will measure on
the page - then exits non-zero if any scale is not 1.0 or any label lands
under 7pt. It catches the one thing the authoring-time guard cannot: sizes
nobody declared, which is how a 9pt CDF panel ends up printing 6.3pt
superscripts.

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
`scikit-learn` and `nltk` (which come in through step03's `tfidf_analysis`)
plus `duckdb`, which reduces the streamed term counts out of core.

## 1. Tag toxicity rates

```bash
python run_tag_toxicity.py \
  --games ../../data/corpus/games/games.parquet \
  --step02-dir ../../data/corpus/reviews_w_detoxify \
  --output-dir ../../data/figures-output
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
millions. It carried a decimal comma in the originally published figure,
inherited from the authoring locale while the rate axis above it used a
point; the camera-ready uses a point on both, so one figure no longer
carries two decimal separators.

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
  --labeled ../../data/step03-output/reviews_cleaned_labeled_en.parquet \
  --lexicon ../../data/step03-output/tfidf_lexicon_en.csv \
  --games ../../data/corpus/games/games.parquet \
  --top-tags ../../data/figures-output/tag_toxicity_top.csv \
  --output-dir ../../data/figures-output
```

**On the full corpus, use `--step02-dir` rather than `--labeled`.** The
`--labeled` path reads step03's intermediate, which means step03 has to
have loaded the whole language into memory first - impossible here, since
the English corpus is ~12.8GB of text. `--step02-dir` streams instead:

```bash
python run_tag_tfidf_heatmap.py \
  --step02-dir ../../data/corpus/reviews_w_detoxify \
  --fit-cache ../../data/figures-cache/corpus_fit_en.npz \
  --games ../../data/corpus/games/games.parquet \
  --output-dir ../../data/figures-output \
  --tag "Team-Based" --tag "Competitive" --tag "PvP" --tag "FPS" --tag "Shooter" \
  --tag "Military" --tag "Free to Play" --tag "Gore" --tag "Violent" --tag "Comedy" \
  --term sucks --term ass --term trash --term kill --term suck --term stupid \
  --term garbage --term balls --term like --term good --term people --term fun
```

That pass takes about 50 minutes and is cached by `--fit-cache`, so a later
run that only changes the figure reuses it. `corpus_vectorizer.py` explains
how it stays inside a few GB and why its output is interchangeable with a
full in-memory fit.

The tags are given explicitly, in the published order, rather than read
from `--top-tags`: the order groups the columns into the tag families the
figure shades (competitive, violent, Comedy), which the toxicity-rate
ordering would interleave. The terms are the published twelve.

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

### The claims the text makes about this figure are checked, not assumed

A TF-IDF cell is not a number the paper quotes - what the text asserts is an
*ordering*: which terms dominate the competitive tags, and that `kill`
overtakes `sucks` under `Gore`, `Violent` and `Comedy`. `PAPER_CLAIMS` in
`tag_tfidf_heatmap.py` states those three sentences, and every run checks
them against the matrix it just computed, printing `holds`, `CONTRADICTED`
or `skipped` and recording the outcome in `tag_tfidf_report.json`.

`skipped` rather than `holds` is the deliberate choice for a run whose tags
or terms do not include the ones a sentence names: the script is
parameterizable, and a run over a different tag set says nothing about a
sentence describing the published one.

This matters more here than for the other two figures. Their numbers are
counts and rates that `verify_numbers.py` compares directly against the
paper. This figure's numbers depend on a vectorizer fitted over the whole
corpus, so the honest thing to verify is the claim rather than the cell -
and a regeneration that would falsify a sentence now says so instead of
letting the sentence go stale.

## 3. User behavioral profile

```bash
python run_user_profile.py \
  --users ../../data/corpus/users/all_users.parquet \
  --step02-dir ../../data/corpus/reviews_w_detoxify \
  --output-dir ../../data/figures-output \
  --user-counts ../../data/figures-cache/user_counts_en.parquet
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

These populations are computed separately and both sizes are written to
the report. The paper's ban percentages (1.4% versus 1.2%) use all users
in each toxicity group as the denominator, counting only observed bans.
The report also supplies matched-profile percentages (3.12% versus 2.69%).
An unmatched profile has unknown ban status; distinguish these two
denominators when interpreting the reported rates.

### Log axes and zeros

The panels use log x-axes, since all three metrics are heavy-tailed, and
match the published axis labels verbatim. A log axis cannot render 0 (an
empty library, account level 0), so those points are absent from the drawn
curve, exactly as in the published panels.

Medians are computed over the raw values *before* plotting, so they are not
affected by that, and each run reports how many points each group lost
under `<group>_zeros_not_drawn` - a number worth glancing at, because it is
invisible in the figure itself.
