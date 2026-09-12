# Characterization results

The numbers behind the paper's characterization section (RQ1), small enough
to commit. Figure PDFs are delivered with the manuscript rather than
committed here. Counts and the complete per-tag heatmap matrix are supplied
as CSV/JSON; the CDF report contains medians and population summaries, not
every point of the empirical curves. Redrawing the CDFs requires the corpus.

| File | What it holds |
|---|---|
| `corpus_verification.json` | Every number the characterization section states, recomputed from the corpus and compared against the published value. Produced by `characterization/verify_numbers.py`. |
| `tag_toxicity_all.csv` | Toxicity rate for all 447 game tags. The figure shows ten; the text also quotes the low end (Exploration 1.22%, Puzzle 1.26%, Cute 1.53%), which is readable here. |
| `tag_toxicity_top.csv` | The ten tags plotted in Figure 1. |
| `tag_toxicity_report.json` | Row accounting for Figure 1: rows read and kept, the p90 volume threshold, how many tags qualified, and the coverage of the tag join. |
| `tag_tfidf_means.csv` | The Figure 2 matrix: mean TF-IDF per term per tag over each tag's toxic reviews. |
| `tag_tfidf_report.json` | The vectorizer behind Figure 2 - corpus size, vocabulary size, and the IDF of each plotted term - plus the toxic-review count per tag. |
| `user_profile_report.json` | Figure 3's medians and populations, both ban rates, and the share of toxic reviews that still recommend the game. |

## The verification

`corpus_verification.json` is the artifact worth reading first. It carries
26 claims, each with the published value, the recomputed value, and the
precision the comparison was made at. All 26 reproduce on the corpus behind
the paper, including both correlation coefficients to six decimals
(Pearson 0.770234, Spearman 0.606591).

To regenerate it:

```bash
python characterization/verify_numbers.py --data data \
    --output results/characterization/corpus_verification.json
```

## Two definitions that change the number

Both are places where a reasonable-looking alternative gives a different
answer, so both are stated explicitly rather than left to the reader:

- **Matched users** (6,204,110; 43.7%) are those whose review URL carries a
  SteamID64 that is present in the collected profile table. Deciding it
  from whether the joined profile fields came back non-null instead counts
  private profiles as unmatched and gives 5,067,115.
- **Ban rates** (1.4% toxic vs. 1.2% non-toxic) divide by every user in the
  group; a profile that was never collected cannot show a ban and counts as
  unbanned. Over matched profiles only, the same counts read 3.12% vs.
  2.69%. `user_profile_report.json` carries both, as `*_ban_pct` and
  `*_ban_pct_matched`.

Neither choice changes the finding it supports: toxic users are banned
slightly more often than non-toxic ones, and nowhere near in proportion to
how much more they review.
