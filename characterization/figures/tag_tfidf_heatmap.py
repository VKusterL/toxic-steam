"""Mean TF-IDF salience of selected terms across game tag contexts.

Produces the paper's `heatmap_tfidf_tags.pdf`: for each of the ten tags in
the toxicity figure, the mean TF-IDF weight of a shared set of
high-salience terms, computed over that tag's *toxic* reviews only. This
is what shows that toxicity is not lexically uniform - `sucks`/`trash`
dominate the competitive tags while `kill` overtakes them under `Gore`,
`Violent` and `Comedy`.

WHY THIS IMPORTS STEP05 RATHER THAN RE-IMPLEMENTING IT. A TF-IDF weight is
meaningful only relative to the vocabulary and document frequencies it was
fitted against. For a cell here to be comparable to the paper's TF-IDF
table, both must come from a vectorizer fitted the same way, on the same
corpus, with the same stopword list, min_df, max_df and max_features.
Copying those settings into this folder would make them two constants that
have to be kept equal by hand; importing `tfidf_analysis` makes them one.
This is the only place in this folder that reaches into another step's
code, and `--step03-code` makes the path explicit.

The vectorizer is therefore re-fitted on the full language corpus (toxic
and non-toxic alike), exactly as `run_tfidf.py` does, and only then
applied per tag. Fitting per tag instead would give every tag its own
document frequencies and make the rows incomparable across columns, which
is precisely the comparison the figure exists to make.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import corpus_io as cio
import plotting
from pipeline_utils import info

LANG_NLTK_NAME = {"pt": "portuguese", "en": "english"}
DEFAULT_N_TERMS = 10
CHUNK_SIZE = 500_000

# Faint background bands grouping the columns into the three tag families
# the paper's reading of this figure rests on: the competitive cluster, the
# violence-tagged pair, and Comedy. Recovered from the published PDF, which
# draws them at alpha 0.15 - low enough that they only tint the white gaps
# between cells, and never the cells themselves.
#
# Membership is editorial, not computed: it is the grouping the paper
# argues for, so it is written down here rather than inferred. Tags absent
# from this map simply get no band.
CLUSTER_BANDS = {
    "#d6eaf8": ["Team-Based", "Competitive", "PvP", "FPS", "Shooter", "Military", "Free to Play"],
    "#fadbd8": ["Gore", "Violent"],
    "#d5f5e3": ["Comedy"],
}
CLUSTER_ALPHA = 0.15


def import_step03(step03_code: Path):
    """Imports step03's tfidf_analysis from its own folder - see this
    module's docstring for why the settings are shared rather than copied."""
    step03_code = Path(step03_code).resolve()
    if not (step03_code / "tfidf_analysis.py").exists():
        raise SystemExit(
            f"tfidf_analysis.py not found in {step03_code}. "
            "Pass --step03-code pointing at step03_tfidf_analysis/."
        )
    sys.path.insert(0, str(step03_code))
    import tfidf_analysis  # noqa: E402
    return tfidf_analysis


def load_labeled_reviews(labeled_path: Path) -> pd.DataFrame:
    """step03's slim intermediate: review_text_clean, game_id, is_toxic.
    Reusing it means the text here was cleaned by exactly the code that
    produced the published lexicon, and saves re-cleaning ~27M reviews."""
    df = pd.read_parquet(labeled_path, columns=["game_id", "review_text_clean", "is_toxic"])
    info(f"Loaded {len(df):,} labeled review(s) from {labeled_path}")
    return df


def select_terms(lexicon_path: Path, n_terms: int, explicit: list = None) -> list:
    """The shared term set. Defaults to the terms the lexicon ranks as most
    disproportionately toxic (`diferenca`), which is the same ordering the
    paper's TF-IDF table is built from, so the heatmap's rows are the terms
    a reader has already met in that table."""
    if explicit:
        info(f"Using {len(explicit)} explicitly requested term(s)")
        return list(explicit)

    lexicon = pd.read_csv(lexicon_path)
    terms = lexicon.sort_values("diferenca", ascending=False)["termo"].head(n_terms).tolist()
    info(f"Selected top {len(terms)} term(s) by diferenca from {lexicon_path}: {terms}")
    return terms


def tag_term_means(
    reviews: pd.DataFrame,
    games: pd.DataFrame,
    tags: list,
    terms: list,
    vectorizer,
) -> tuple:
    """Mean TF-IDF per (term, tag) over each tag's toxic reviews.

    Transforms in chunks and accumulates column sums rather than building
    one sparse matrix for the whole corpus - the same bounded-memory shape
    as step03's compute_group_means, for the same reason.
    """
    vocabulary = vectorizer.vocabulary_
    missing = [t for t in terms if t not in vocabulary]
    if missing:
        info(f"WARNING: {len(missing)} term(s) absent from the fitted vocabulary, dropped: {missing}")
    kept_terms = [t for t in terms if t in vocabulary]
    if not kept_terms:
        raise SystemExit("None of the requested terms are in the fitted vocabulary.")
    columns = [vocabulary[t] for t in kept_terms]

    pairs = cio.explode_game_tags(games)
    toxic = reviews[reviews["is_toxic"].fillna(False).astype(bool)]
    info(f"{len(toxic):,} toxic review(s) available for the per-tag means")

    means = pd.DataFrame(index=kept_terms, columns=tags, dtype="float64")
    counts = {}

    for tag in tags:
        tag_games = set(pairs.loc[pairs["tag"] == tag, "game_id"])
        subset = toxic[toxic["game_id"].isin(tag_games)]
        n = len(subset)
        counts[tag] = n
        if n == 0:
            info(f"[{tag}] no toxic reviews - column left empty")
            means[tag] = np.nan
            continue

        accumulated = np.zeros(len(columns), dtype="float64")
        for start in range(0, n, CHUNK_SIZE):
            chunk = subset.iloc[start:start + CHUNK_SIZE]["review_text_clean"].fillna("").astype(str)
            X = vectorizer.transform(chunk)
            accumulated += np.asarray(X[:, columns].sum(axis=0)).ravel()
        means[tag] = accumulated / n
        info(f"[{tag}] {n:,} toxic review(s) averaged")

    return means, counts


def _draw_cluster_bands(ax, columns: list) -> int:
    """Shades each run of adjacent columns belonging to the same tag family.
    Returns how many bands were drawn, so a run over a tag set the map does
    not cover reports 0 instead of silently looking different."""
    colour_of = {tag: colour for colour, tags in CLUSTER_BANDS.items() for tag in tags}
    drawn = 0
    start = 0
    while start < len(columns):
        colour = colour_of.get(columns[start])
        end = start
        while end + 1 < len(columns) and colour_of.get(columns[end + 1]) == colour:
            end += 1
        if colour is not None:
            ax.axvspan(start, end + 1, color=colour, alpha=CLUSTER_ALPHA, zorder=0, linewidth=0)
            drawn += 1
        start = end + 1
    return drawn


def plot_heatmap(means: pd.DataFrame, output_path: Path) -> Path:
    """Terms as rows, tags as columns, one shared color scale, every cell
    carrying its own value.

    The scale is shared on purpose: a per-column normalization would make
    every tag's most salient term equally red and erase the very contrast
    the figure reports, which is that `kill` under Gore reaches a weight
    `sucks` never reaches under Puzzle. Because the scale is shared, the
    light end of the grid is hard to read off the color alone - which is
    why every cell is annotated rather than left to the colorbar.
    """
    plotting.apply_style(plotting.HEATMAP_FONTSIZE)
    import matplotlib.pyplot as plt

    data = means.to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=plotting.HEATMAP_FIGSIZE)

    n_bands = _draw_cluster_bands(ax, list(means.columns))
    if n_bands == 0:
        info("No tag-family bands drawn - none of these tags are in CLUSTER_BANDS")

    image = ax.pcolormesh(data, cmap="Reds", edgecolors="white", linewidth=0.5)

    ax.set_xticks([i + 0.5 for i in range(len(means.columns))])
    ax.set_xticklabels(means.columns, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks([i + 0.5 for i in range(len(means.index))])
    ax.set_yticklabels(means.index)
    ax.set_ylabel("Term", fontsize=plotting.HEATMAP_LABEL_FONTSIZE)
    # No invert_yaxis: pcolormesh puts row 0 at the bottom, and `means` is
    # ordered strongest-first, so the most salient term lands on the bottom
    # row - the published orientation, weakest at the top.
    ax.grid(visible=False)
    ax.tick_params(length=0)

    # White on the dark cells, black on the light ones. The threshold is a
    # fraction of the scale's own maximum, so it follows the data instead of
    # assuming a fixed range.
    finite = data[~pd.isna(data)]
    switch = finite.max() * 0.55 if len(finite) else 0
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if pd.isna(value):
                continue
            ax.text(
                col + 0.5, row + 0.5, f"{value:.3f}",
                ha="center", va="center",
                color="white" if value > switch else "black",
                fontsize=plotting.HEATMAP_FONTSIZE,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean TF-IDF Score", fontsize=plotting.HEATMAP_CBAR_FONTSIZE)
    colorbar.ax.tick_params(labelsize=plotting.HEATMAP_CBAR_FONTSIZE)

    saved = plotting.save_figure(fig, output_path)
    info(f"Wrote figure: {saved}")
    return saved


def export_means(means: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    means.to_csv(output_path, index_label="termo")
    info(f"Wrote table: {output_path}")
    return output_path
