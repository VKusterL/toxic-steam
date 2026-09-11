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
    populations stay explicit rather than implicit in a dropna()."""
    profiles = cio.load_user_profiles(users_path)
    users = users.copy()
    users["steam_id"] = cio.extract_steam_id(users["user_url"])

    merged = users.merge(profiles, on="steam_id", how="left")
    merged["matched"] = merged["profile_level"].notna() | merged["library_size"].notna()

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
    plotting.apply_style(plotting.CDF_FONTSIZE)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=plotting.CDF_FIGSIZE)
    medians = {}

    # `key` is the report/JSON name and `label` the one drawn in the legend;
    # they are kept separate so the key stays a plain identifier, matching
    # ban_rates' `toxic`/`non_toxic` naming.
    for toxic, key, color, label in (
        (False, "non_toxic", plotting.NONTOXIC_COLOR, "Non-toxic users"),
        (True, "toxic", plotting.TOXIC_COLOR, "Toxic users"),
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

        x, y = empirical_cdf(values)
        ax.plot(x, y, color=color, linewidth=plotting.CDF_LINEWIDTH, label=label, zorder=3)
        ax.axvline(
            median, color=color, linestyle="--",
            linewidth=plotting.CDF_LINEWIDTH, zorder=2,
        )

    ax.set_xlabel(panel["xlabel"])
    ax.set_ylabel("CDF")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    # Decade labels only. On the real corpus every panel spans four or five
    # decades and matplotlib already labels only decades, so this changes
    # nothing there; it keeps a narrow-range run (a subsample, a smoke test)
    # from collapsing the axis into an unreadable band of minor labels.
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    saved = plotting.save_figure(fig, output_path)
    info(f"Wrote figure: {saved}  medians={medians}")
    return saved, medians


def plot_legend(output_path: Path) -> Path:
    """The shared legend, as its own file - the three panels sit side by
    side in the paper under one legend, so none of them carries its own.

    Two entries only, matching the published legend: the median rules are
    drawn in each group's own color, so they need no separate key."""
    plotting.apply_style(plotting.LEGEND_FONTSIZE)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=plotting.NONTOXIC_COLOR,
               linewidth=plotting.CDF_LINEWIDTH, label="Non-toxic users"),
        Line2D([], [], color=plotting.TOXIC_COLOR,
               linewidth=plotting.CDF_LINEWIDTH, label="Toxic users"),
    ]
    fig = plt.figure(figsize=plotting.LEGEND_FIGSIZE)
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False)
    saved = plotting.save_figure(fig, output_path)
    info(f"Wrote figure: {saved}")
    return saved


def ban_rates(users: pd.DataFrame) -> dict:
    """Share of each group with a public ban on record.

    Restricted to matched profiles, because `has_ban` is a profile field:
    an unmatched user is not an unbanned user, and counting them as one
    would push both rates toward zero by roughly the same unmatched share,
    making the comparison look tighter than it is."""
    matched = users[users["matched"]]
    out = {}
    for toxic, label in ((True, "toxic"), (False, "non_toxic")):
        group = matched[matched["is_toxic_user"] == toxic]
        flags = group["has_ban"].astype("boolean")
        known = flags.notna().sum()
        banned = int(flags.fillna(False).sum())
        out[f"{label}_n"] = int(len(group))
        out[f"{label}_known"] = int(known)
        out[f"{label}_banned"] = banned
        out[f"{label}_ban_pct"] = 100 * banned / known if known else None
    return out
