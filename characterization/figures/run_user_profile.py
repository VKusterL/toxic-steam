"""CLI for user_profile.py - the paper's CDF panels and the descriptive
statistics of the behavioral-profile section.

Usage:
    python run_user_profile.py \\
        --users ../../steam-data/step01-output/users/all_users.parquet \\
        --step02-dir ../../steam-data/step02-output \\
        --output-dir ../../steam-data/figures-output

Writes `cdf-reviews-per-user.pdf`, `cdf-library-size.pdf`,
`cdf-steam-level.pdf` and `cdf-legend.pdf` (all four ready to drop into
`main/images/`), plus `user_profile_report.json` carrying the group
medians, the ban rates, and the share of toxic reviews that still
recommend the game.

This is the heaviest script in this folder: it aggregates every English
review down to one row per user before anything is plotted. Use
`--user-counts` to cache that aggregate and skip the pass on re-runs -
tuning a figure should not mean re-reading 36.8M reviews.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import user_profile as up
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toxic vs. non-toxic user engagement CDFs and behavioral statistics."
    )
    parser.add_argument("--users", required=True, type=Path, help="step01's all_users.parquet")
    parser.add_argument("--step02-dir", required=True, type=Path, help="step02's output directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write outputs to")
    parser.add_argument("--lang", default="en", help="Language to analyze (default: en, what the paper reports)")
    parser.add_argument(
        "--user-counts", type=Path, default=None,
        help="Cache path for the per-user aggregate. Read if it exists, written if it does not.",
    )
    return parser.parse_args()


def load_or_build_counts(args) -> tuple:
    """Per-user counts, from the cache when one exists.

    The review-level statistics travel with the cache in a sidecar JSON.
    They are computed during the corpus pass but are not recoverable from
    the per-user table afterwards - the share of toxic reviews that still
    recommend the game is a property of reviews, not of users - so without
    the sidecar a cached run would quietly report fewer numbers than a cold
    one, which is exactly the kind of difference an artifact reviewer would
    hit and not be able to explain.
    """
    sidecar = args.user_counts.with_suffix(".stats.json") if args.user_counts else None

    if args.user_counts and args.user_counts.exists() and sidecar.exists():
        info(f"Reusing cached per-user counts from {args.user_counts}")
        users = pd.read_parquet(args.user_counts)
        stats = json.loads(sidecar.read_text(encoding="utf-8"))
        stats["user_counts_cached"] = True
        return users, stats

    users, stats = up.per_user_counts(args.step02_dir, args.lang)
    stats["user_counts_cached"] = False
    if args.user_counts:
        args.user_counts.parent.mkdir(parents=True, exist_ok=True)
        users.to_parquet(args.user_counts, index=False)
        sidecar.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        info(f"Cached per-user counts to {args.user_counts} (+ {sidecar.name})")
    return users, stats


def main():
    args = parse_args()

    users, stats = load_or_build_counts(args)
    users, match_stats = up.attach_profiles(users, args.users)

    medians = {}
    figures = {}
    for name, panel in up.PANELS.items():
        path, panel_medians = up.plot_panel(users, panel, args.output_dir / panel["filename"])
        medians[name] = panel_medians
        figures[name] = str(path)
    figures["legend"] = str(up.plot_legend(args.output_dir / "cdf-legend.pdf"))

    bans = up.ban_rates(users)

    info("Group medians:")
    for name, values in medians.items():
        info(f"  {name}: toxic={values.get('toxic')} non-toxic={values.get('non_toxic')}")
    if bans.get("toxic_ban_pct") is not None:
        info(
            f"Ban rate (all users, the paper's denominator): "
            f"toxic {bans['toxic_ban_pct']:.2f}% vs non-toxic {bans['non_toxic_ban_pct']:.2f}%"
        )
        info(
            f"Ban rate (matched profiles only): "
            f"toxic {bans['toxic_ban_pct_matched']:.2f}% vs "
            f"non-toxic {bans['non_toxic_ban_pct_matched']:.2f}%"
        )
    else:
        info("Ban rate: not computable (no matched profiles)")
    if stats.get("toxic_reviews_recommended_pct") is not None:
        info(
            f"{stats['toxic_reviews_recommended_pct']:.1f}% of toxic reviews still recommend the game"
        )

    save_summary(
        {
            "language": args.lang,
            **stats,
            **match_stats,
            "medians": medians,
            "ban_rates": bans,
            "figures": figures,
            "note": (
                "reviews_per_user is computed over every user in the corpus; library_size and "
                "profile_level only over users matched to a collected public profile. Ban rates "
                "are given both ways: *_ban_pct divides by every user in the group, which is the "
                "paper's 1.4% vs 1.2%, and *_ban_pct_matched divides by the matched profiles "
                "where the flag was actually observed."
            ),
        },
        args.output_dir / "user_profile_report.json",
    )


if __name__ == "__main__":
    main()
