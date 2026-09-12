"""Per-tag toxicity rates: how much of a game tag's review volume is toxic.

Produces the paper's `top-10-tags-tox.pdf` - the ten tags with the highest
toxicity rate among those above the 90th percentile of total review volume,
each drawn as a stacked bar of its non-toxic and toxic review counts with
the rate overlaid as a line - plus the full per-tag table the surrounding
text quotes from, including the low end (Exploration, Puzzle, Cute) that
the figure itself does not show.

THE VOLUME THRESHOLD IS NOT COSMETIC. A toxicity rate is a ratio, and a
tag carried by three games with eleven reviews between them can reach 30%
on three toxic reviews. Ranking tags by rate without a volume floor
returns those, not the competitive cluster the paper reports. The 90th
percentile is computed over the tags' own total review volume, so the
floor adapts to the corpus rather than being a hand-picked constant.

Counting is per (game, tag) pair: a game tagged both `PvP` and `Free to
Play` contributes its full review count to both tags. Tags are not
mutually exclusive on Steam, and the paper's reading of the Free to Play
tag - that its toxicity is partly inherited from co-occurring competitive
tags - depends on that overlap being preserved rather than divided up.
"""
from pathlib import Path

import pandas as pd

import corpus_io as cio
import plotting
from pipeline_utils import info

VOLUME_QUANTILE = 0.90
TOP_N = 10


def per_game_counts(step02_dir: Path, lang: str) -> tuple:
    """Total and toxic review counts per game, accumulated one step02 file
    at a time. Returns (counts, stats) where `counts` is indexed by game_id
    and `stats` carries the row accounting for the run summary."""
    totals = None
    n_read = n_kept = n_invalid = 0

    for df, read, invalid in cio.iter_scored_reviews(step02_dir, lang, ["game_id"]):
        n_read += read
        n_invalid += invalid
        n_kept += len(df)
        if df.empty:
            continue
        df = df.assign(game_id=cio.normalize_game_id(df["game_id"]))
        part = df.groupby("game_id", dropna=True)["is_toxic"].agg(["size", "sum"])
        part.columns = ["n_reviews", "n_toxic"]
        totals = part if totals is None else totals.add(part, fill_value=0)

    if totals is None:
        raise SystemExit(f"No reviews found for language '{lang}' under {step02_dir}")

    totals = totals.astype("int64").reset_index()
    info(
        f"[{lang}] {n_read:,} row(s) read, {n_kept:,} kept "
        f"({n_invalid:,} dropped for an invalid score), {len(totals):,} game(s) with reviews"
    )
    stats = {
        "rows_read": n_read,
        "rows_kept": n_kept,
        "rows_dropped_invalid": n_invalid,
        "games_with_reviews": len(totals),
        "reviews_toxic": int(totals["n_toxic"].sum()),
    }
    return totals, stats


def tag_table(game_counts: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Sums each tag's review counts over the games carrying it and adds
    the toxicity rate. Games with reviews but absent from the games table
    (or carrying no tags) contribute to no tag - see tag_coverage for how
    much of the corpus that leaves out."""
    merged = pairs.merge(game_counts, on="game_id", how="inner")

    table = merged.groupby("tag", as_index=False)[["n_reviews", "n_toxic"]].sum()
    table["toxicity_rate"] = table["n_toxic"] / table["n_reviews"]
    table["toxicity_pct"] = 100 * table["toxicity_rate"]
    return table.sort_values("toxicity_rate", ascending=False).reset_index(drop=True)


def tag_coverage(game_counts: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """How much of the corpus the tag table actually speaks for.

    Summing `n_toxic` down the tag table does NOT give this back: a game
    carrying ten tags is counted ten times there, deliberately. Coverage
    has to be measured over distinct games instead, which is the number
    worth reporting - it says how much of the corpus is invisible to this
    figure because its game carries no tags (or is missing from the games
    table entirely)."""
    tagged = set(pairs["game_id"])
    covered = game_counts[game_counts["game_id"].isin(tagged)]
    return {
        "games_covered": len(covered),
        "games_uncovered": len(game_counts) - len(covered),
        "reviews_covered": int(covered["n_reviews"].sum()),
        "reviews_toxic_covered": int(covered["n_toxic"].sum()),
    }


def apply_volume_threshold(table: pd.DataFrame, quantile: float = VOLUME_QUANTILE) -> tuple:
    """Keeps tags at or above the `quantile`-th percentile of total review
    volume. Returns (filtered_table, threshold)."""
    threshold = table["n_reviews"].quantile(quantile)
    kept = table[table["n_reviews"] >= threshold].reset_index(drop=True)
    info(
        f"Volume threshold (p{quantile:.0%}): {threshold:,.0f} reviews - "
        f"{len(kept)} of {len(table)} tag(s) qualify"
    )
    return kept, float(threshold)


def _millions(value, _pos) -> str:
    """0.0 / 1.0 / ... - one decimal, and a decimal point.

    The originally published figure carried a decimal comma here, inherited
    from the authoring locale, while the toxicity axis above it and every
    number in the running text use a point. Two separators on one figure is
    a typo the camera-ready should not keep, so the axis now matches the
    rest of the paper.
    """
    return f"{value:.1f}"


def plot_top_tags(top: pd.DataFrame, output_path: Path) -> Path:
    """The published design: one horizontal bar per tag, split into its
    non-toxic and toxic review counts, with the toxicity rate overlaid as a
    black line on a second x-axis at the top.

    The two quantities live on different scales - millions of reviews
    against single-digit percentages - so a shared axis would flatten the
    rate into the baseline. The bar carries the volume that makes a rate
    trustworthy, and the line carries the rate itself.
    """
    plotting.apply_style(plotting.BAR_FONTSIZE)
    plotting.assert_compliant([plotting.BAR_FONTSIZE], "top-10-tags-tox")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Highest rate at the top: barh draws the first row at the bottom.
    ordered = top.sort_values("toxicity_rate", ascending=True).reset_index(drop=True)
    positions = list(range(len(ordered)))

    non_toxic_m = (ordered["n_reviews"] - ordered["n_toxic"]) / 1e6
    toxic_m = ordered["n_toxic"] / 1e6

    fig, ax = plt.subplots(figsize=plotting.BAR_FIGSIZE, layout="constrained")
    ax.barh(positions, non_toxic_m, color=plotting.NONTOXIC_COLOR, label="Non-toxic", zorder=2)
    # The hatch is what keeps the toxic slice visible in a grayscale print,
    # where its red and the non-toxic green differ by only 8% of luminance -
    # see plotting.py. In color it reads as the same bar it always was.
    ax.barh(
        positions, toxic_m, left=non_toxic_m, color=plotting.TOXIC_COLOR,
        label="Toxic", zorder=2, hatch=plotting.TOXIC_HATCH,
        edgecolor="white", linewidth=0.4,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["tag"])
    ax.set_ylim(-0.7, len(ordered) - 0.5)  # room for the legend under the bars
    ax.set_xlabel("Number of reviews (Millions)")
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax.grid(axis="x", linestyle="--", linewidth=plotting.GRID_LINEWIDTH, zorder=0)
    ax.set_axisbelow(True)

    rate_ax = ax.twiny()
    rate_ax.plot(
        ordered["toxicity_pct"], positions,
        color=plotting.RATE_LINE_COLOR, marker="o", markersize=2.6,
        linewidth=plotting.RATE_LINEWIDTH, label="% Toxicity", zorder=3,
    )
    rate_ax.set_xlabel("Toxic reviews (%)")
    rate_ax.set_ylim(ax.get_ylim())
    rate_ax.grid(visible=False)

    # One legend for both axes - the bars live on `ax`, the line on `rate_ax`.
    handles = ax.get_legend_handles_labels()[0] + rate_ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + rate_ax.get_legend_handles_labels()[1]
    ax.legend(
        handles, labels, loc="lower right", fontsize=plotting.BAR_FONTSIZE,
        handlelength=1.4, borderpad=0.35, labelspacing=0.3, borderaxespad=0.3,
    )

    saved = plotting.save_figure(fig, output_path, expected_width=plotting.BAR_FIGSIZE[0])
    info(f"Wrote figure: {saved}")
    return saved


def export_table(table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    info(f"Wrote table: {output_path}")
    return output_path
