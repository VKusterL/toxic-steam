# Data, labels, and features

## Availability

The raw review, user, and game corpus is not part of the publication artifact. It was collected from the public Steam Web API between September 2024 and April 2025 and describes users who did not consent to toxicity profiling. The public export includes aggregate numeric summaries, including the characterization tables under `results/characterization/`.

The research checkout also contains tracked per-user prediction files with Steam identifiers, and narrative-panel records with review quotations. These are identifiable research outputs, not anonymous aggregates. They are excluded by the publication export; a direct archive or publication of the existing Git history does not provide that exclusion. See [artifact.md](artifact.md) for the release boundary.

To re-run the pipeline from scratch, the collected corpus must be placed under `data/` in the parquet layout expected by `build_features.py`. Any re-collection should pseudonymize Steam identifiers and profile URLs.

The prediction entry point discovers these inputs relative to the repository root:

```text
data/corpus/
├── reviews_w_detoxify/       Scored review parquet files, with English paths marked en or lang=en
├── users/all_users.parquet   Public profile table
└── games/games.parquet       Game metadata and tags
```

Use `python src/build_features.py --root . --out data/features --langs en` from the repository root. The review files must already include both `toxicity` and `perspective_score`: the feature builder does not call either scoring service. The characterization scripts document the earlier cleaning and Detoxify stages. There is no bundled collector that can recreate the historical Steam snapshot from scratch, and the Perspective scores must be supplied in the input corpus.

## Collection and scope

Collection proceeded through the Steam API in two passes. First, game titles and their metadata (names and community tags) were gathered. From those titles, reviews were collected as text, timestamps, and public user identifiers, which then yielded a set of unique users with public profile details such as library size, platform level, and self-reported location.

After keeping only English reviews and dropping users and games with no associated reviews, the working corpus is 36,823,127 reviews from 14,183,630 unique users across 67,477 titles that have at least one English review.

## Toxicity labeling

Two automatic detectors score each review on a continuous scale in the range 0 to 1: the Perspective API and Detoxify. They agree moderately (Pearson r = 0.77, Spearman rho = 0.61), which suggests they capture complementary facets of toxicity, so their union is used to maximize coverage.

Thresholds were calibrated with a human annotation study. Reviews were sampled across score bins of width 0.1 for each model, 200 per model, and three annotators labeled each review as toxic or non-toxic without seeing model scores or game identifiers. Based on the resulting agreement, a review is labeled toxic if it scores at least 0.9 on Detoxify or at least 0.7 on the Perspective API. This yields 685,536 toxic reviews (about 1.86 percent of the corpus) from 560,079 unique users.

## User-level modeling dataset

For user-level prediction, the dataset is restricted to users with English reviews whose histories match a collected public profile. Of the full user set, 6,204,110 (43.7 percent) matched, with a toxic-user prevalence of 3.97 percent. Modeling then runs on a deterministic 10 percent slice, selected by a hash of the public user identifier, which preserves prevalence and is exactly reproducible. The slice holds 620,891 users with 24,836 positives, a prevalence of 4.00 percent.

### Label definitions

Two complementary labels are used, kept as separate populations:

- **Count label** (`y_count`): a user is positive if they authored at least one review classified as toxic. This records whether a user has ever met the toxicity criterion, but it is sensitive to how many reviews a user wrote.
- **Rate label** (`y_rate`): among users with at least five English reviews, a user is positive if at least 5 percent of their reviews are toxic. This demands that toxicity be a repeated or proportionally salient part of the reviewing behavior. Prevalence on this subpopulation is about 12.8 percent.

## Feature groups

Five user-level feature families are evaluated.

- **Profile metadata**: Steam level, library size, awards, badges, groups, country, profile visibility, missingness flags, and related public profile fields.
- **Profile + ban**: metadata plus public ban indicators and ban recency.
- **Content embeddings**: user vectors obtained by averaging Sentence-BERT embeddings over the user's English reviews. `all-MiniLM-L6-v2` is the primary encoder; `all-mpnet-base-v2` is a robustness check.
- **Leave-toxic-out embeddings**: the same averaging, but excluding reviews labeled toxic, defined only for users with at least one non-toxic review.
- **Combined**: profile metadata concatenated with content embeddings.

The content and combined arms are intentionally strong but partially circular, because they read the text that defines the label. The leave-toxic-out arm is the critical control: if it performs above the metadata baseline, toxic users have a broader textual or behavioral signature beyond the exact review that triggered the label. For the users whose every review is toxic, the leave-toxic-out vector is undefined, and imputing it would turn its absence into a near-perfect proxy of the label, so that control is trained and evaluated only on users with at least one non-toxic review.
