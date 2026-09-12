"""Behavioral profile of Steam users: toxic versus non-toxic.

Produces the paper's three CDF panels (`cdf-reviews-per-user.pdf`,
`cdf-library-size.pdf`, `cdf-steam-level.pdf`) with their shared legend
(`cdf-legend.pdf`), plus the descriptive statistics the surrounding text
quotes: the group medians, the ban rates, and the share of toxic reviews
that still recommend the reviewed game.

A user is toxic when they authored at least one review meeting the
calibrated criterion - the same count rule used everywhere else in this
project. That rule is sensitive to activity volume by construction (a user
with more reviews has more chances to cross it), which is exactly why the
reviews-per-user panel is reported alongside the other two rather than
quietly omitted: it is the panel that shows the confound.

TWO POPULATIONS, NOT ONE. Reviews per user is defined for every user in
the corpus. Library size and profile level exist only for users whose
review URL carries a SteamID64 *and* whose profile was collected - about
43.7% of them. Mixing the two would silently change the denominator
between panels of the same figure, so they are computed separately and
both population sizes are written to the report.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import corpus_io as cio
import plotting
from pipeline_utils import info

# Reduce the per-file groupby results back down every N files, so the list
# of partial frames never grows to the size of the corpus.
COMPACT_EVERY = 20

# An empirical CDF over millions of points cannot be distinguished from one
# drawn at this resolution, and the PDF stays small enough to embed.
PLOT_POINTS = 2000

# The x axes are logarithmic, matching the published panels. A log axis
# cannot render 0 (an empty library, account level 0), so those points are
# absent from the drawn curve. Medians are computed over the raw values
# before plotting, so they are unaffected; plot_panel reports how many
# points each group lost, under `<group>_zeros_not_drawn`.

# Axis labels are the published ones, verbatim.
PANELS = {
    "reviews_per_user": {
        "column": "n_reviews",
        "xlabel": "Number of reviews per user",
        "filename": "cdf-reviews-per-user.pdf",
        "population": "all",
    },
    "library_size": {
        "column": "library_size",
        "xlabel": "Games in library",
        "filename": "cdf-library-size.pdf",
        "population": "matched",
    },
    "steam_level": {
        "column": "profile_level",
        "xlabel": "Steam account level",
        "filename": "cdf-steam-level.pdf",
        "population": "matched",
    },
}


def per_user_counts(step02_dir: Path, lang: str) -> tuple:
    """Review and toxic-review counts per user, plus the review-level
    recommendation split, accumulated one step02 file at a time."""
    partials = []
    totals = None
    n_read = n_kept = n_invalid = 0
    toxic_recommended = toxic_total = 0

    def compact(frames, accumulated):
        merged = pd.concat(frames).groupby(level=0).sum()
        return merged if accumulated is None else accumulated.add(merged, fill_value=0)

    for df, read, invalid in cio.iter_scored_reviews(
        step02_dir, lang, ["user_url", "is_recommended"]
    ):
        n_read += read
        n_invalid += invalid
        n_kept += len(df)
        if df.empty:
            continue

        toxic_rows = df[df["is_toxic"]]
        toxic_total += len(toxic_rows)
        toxic_recommended += int(toxic_rows["is_recommended"].fillna(False).astype(bool).sum())

        part = df.groupby("user_url", dropna=True)["is_toxic"].agg(["size", "sum"])
        part.columns = ["n_reviews", "n_toxic"]
        partials.append(part)

        if len(partials) >= COMPACT_EVERY:
            totals = compact(partials, totals)
            partials = []

    if partials:
        totals = compact(partials, totals)
    if totals is None:
        raise SystemExit(f"No reviews found for language '{lang}' under {step02_dir}")

    users = totals.astype("int64").reset_index()
    users["is_toxic_user"] = users["n_toxic"] >= 1

    info(
        f"[{lang}] {n_read:,} row(s) read, {n_kept:,} kept "
        f"({n_invalid:,} dropped for an invalid score), {len(users):,} distinct user(s)"
    )
    stats = {
        "rows_read": n_read,
        "rows_kept": n_kept,
        "rows_dropped_invalid": n_invalid,
        "users_total": len(users),
        "users_toxic": int(users["is_toxic_user"].sum()),
        "toxic_reviews": toxic_total,
        "toxic_reviews_recommended": toxic_recommended,
        "toxic_reviews_recommended_pct": (
            100 * toxic_recommended / toxic_total if toxic_total else None
        ),
    }
    return users, stats


def attach_profiles(users: pd.DataFrame, users_path: Path) -> tuple:
    """Left-joins step01's profile table on the SteamID64 extracted from
    the review URL. Users that do not match keep NaN for the profile
    columns; `matched` marks the ones that did, so the per-panel
    populations stay explicit rather than implicit in a dropna().

    `matched` comes from an explicit indicator carried by the profile
    table, NOT from whether the joined fields are non-null. A collected
    profile can legitimately have a null level or library size - that is
    what a private profile looks like - so reading the payload would score
    those as unmatched and shrink the matched population from 6.20M to
    5.07M, dragging the ban-rate denominator with it.
    """
    profiles = cio.load_user_profiles(users_path)
    profiles = profiles.assign(_profile_found=True)
    users = users.copy()
    users["steam_id"] = cio.extract_steam_id(users["user_url"])

    merged = users.merge(profiles, on="steam_id", how="left")
    merged["matched"] = merged.pop("_profile_found").fillna(False).astype(bool)

    n_with_id = int(users["steam_id"].notna().sum())
    n_matched = int(merged["matched"].sum())
    info(
        f"{n_with_id:,} user(s) carry a SteamID64 URL; {n_matched:,} matched a collected profile "
        f"({100 * n_matched / len(merged):.1f}% of all users)"
    )
    return merged, {"users_with_steamid": n_with_id, "users_matched": n_matched}


def _series_for(users: pd.DataFrame, panel: dict, toxic: bool) -> pd.Series:
    subset = users[users["is_toxic_user"] == toxic]
    if panel["population"] == "matched":
        subset = subset[subset["matched"]]
    return pd.to_numeric(subset[panel["column"]], errors="coerce").dropna()


def empirical_cdf(values: np.ndarray, max_points: int = PLOT_POINTS) -> tuple:
    """(x, y) of the empirical CDF, thinned to at most `max_points` for
    drawing. Thinning happens after sorting, so it subsamples the curve
    rather than the data - the median and the tail shape are unaffected."""
    ordered = np.sort(values)
    y = np.arange(1, len(ordered) + 1) / len(ordered)
    if len(ordered) > max_points:
        idx = np.unique(np.linspace(0, len(ordered) - 1, max_points).astype(int))
        ordered, y = ordered[idx], y[idx]
    return ordered, y


def plot_panel(users: pd.DataFrame, panel: dict, output_path: Path) -> tuple:
    """One CDF panel: both groups, with a dashed vertical line at each
    group's median."""
    # has_mathtext: the log axis prints 10^n, and the exponent is the
    # smallest type in the figure.
    plotting.apply_style(plotting.CDF_FONTSIZE)
    plotting.assert_compliant(
        [plotting.CDF_FONTSIZE], output_path.name, has_mathtext=True
    )
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=plotting.CDF_FIGSIZE, layout="constrained")
    medians = {}

    # `key` is the report/JSON name and `label` the one drawn in the legend;
    # they are kept separate so the key stays a plain identifier, matching
    # ban_rates' `toxic`/`non_toxic` naming.
    for toxic, key, color, style, label in (
        (False, "non_toxic", plotting.NONTOXIC_COLOR,
         plotting.NONTOXIC_LINESTYLE, "Non-toxic users"),
        (True, "toxic", plotting.TOXIC_COLOR,
         plotting.TOXIC_LINESTYLE, "Toxic users"),
    ):
        values = _series_for(users, panel, toxic).to_numpy(dtype="float64")
        if len(values) == 0:
            info(f"[{panel['xlabel']}] no values for the {key} group - skipped")
            continue

        # The median is taken over every value, including any 0, so it is
        # not affected by the log axis dropping them from the drawn curve.
        median = float(np.median(values))
        medians[key] = median
        medians[f"{key}_n"] = int(len(values))
        n_zero = int((values <= 0).sum())
        if n_zero:
            medians[f"{key}_zeros_not_drawn"] = n_zero

        # Dash pattern, not just hue: printed in grayscale the two curves
        # are the same grey - see plotting.py. The median rule is dotted so
        # it stays distinct from the dashed non-toxic curve.
        x, y = empirical_cdf(values)
        ax.plot(x, y, color=color, linestyle=style,
                linewidth=plotting.CDF_LINEWIDTH, label=label, zorder=3)
        ax.axvline(
            median, color=color, linestyle=plotting.MEDIAN_LINESTYLE,
            linewidth=plotting.CDF_LINEWIDTH, zorder=2,
        )

    ax.set_xlabel(panel["xlabel"])
    ax.set_ylabel("CDF")
    # Explicit y ticks: at this panel size matplotlib's automatic locator
    # thins them to 0/0.5/1.0, which reads as a coarser curve than the
    # published panels. Six labels still fit comfortably at 10pt.
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    # Decade labels only. On the real corpus every panel spans four or five
    # decades and matplotlib already labels only decades, so this changes
    # nothing there; it keeps a narrow-range run (a subsample, a smoke test)
    # from collapsing the axis into an unreadable band of minor labels.
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    saved = plotting.save_figure(
        fig, output_path, expected_width=plotting.CDF_FIGSIZE[0]
    )
    info(f"Wrote figure: {saved}  medians={medians}")
    return saved, medians


def plot_legend(output_path: Path) -> Path:
    """The shared legend, as its own file - the three panels sit side by
    side in the paper under one legend, so none of them carries its own.

    Two entries only, matching the published legend: the median rules are
    drawn in each group's own color, so they need no separate key."""
    plotting.apply_style(plotting.CDF_FONTSIZE)
    plotting.assert_compliant([plotting.CDF_FONTSIZE], "cdf-legend")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=plotting.NONTOXIC_COLOR,
               linestyle=plotting.NONTOXIC_LINESTYLE,
               linewidth=plotting.CDF_LINEWIDTH, label="Non-toxic users"),
        Line2D([], [], color=plotting.TOXIC_COLOR,
               linestyle=plotting.TOXIC_LINESTYLE,
               linewidth=plotting.CDF_LINEWIDTH, label="Toxic users"),
    ]
    fig = plt.figure(figsize=plotting.LEGEND_FIGSIZE, layout="constrained")
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False, handlelength=1.8)
    saved = plotting.save_figure(
        fig, output_path, expected_width=plotting.LEGEND_FIGSIZE[0]
    )
    info(f"Wrote figure: {saved}")
    return saved


def ban_rates(users: pd.DataFrame) -> dict:
    """Share of each group with a public ban on record, both ways.

    TWO DENOMINATORS, AND THE PAPER USES THE FIRST ONE. `ban_pct` divides
    by every user in the group, so a user whose profile was never collected
    counts as not banned. That is what the published 1.4% vs. 1.2% reports,
    and it is the conservative reading: an unseen ban cannot inflate the
    gap.

    `ban_pct_matched` divides by the matched profiles only, where the flag
    was actually observed. It is the higher figure (3.12% vs. 2.69% on the
    English corpus) because roughly 44% of users match a profile at all.
    Both are reported because the choice changes the number by more than
    the gap between the two groups does, and a reader comparing this
    artifact to the paper needs to see which one is which.

    Neither denominator changes the finding: toxic users are banned
    slightly more often, and nowhere near in proportion to how much more
    they review.
    """
    out = {}
    for toxic, label in ((True, "toxic"), (False, "non_toxic")):
        group = users[users["is_toxic_user"] == toxic]
        matched = group[group["matched"]]
        banned = int(matched["has_ban"].astype("boolean").fillna(False).sum())

        out[f"{label}_n"] = int(len(group))
        out[f"{label}_matched"] = int(len(matched))
        out[f"{label}_banned"] = banned
        out[f"{label}_ban_pct"] = 100 * banned / len(group) if len(group) else None
        out[f"{label}_ban_pct_matched"] = (
            100 * banned / len(matched) if len(matched) else None
        )
    return out
