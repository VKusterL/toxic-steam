"""CLI for tag_toxicity.py - the paper's top-10 game tags by toxicity rate.

Usage:
    python run_tag_toxicity.py \\
        --games ../../steam-data/step01-output/games/games.parquet \\
        --step02-dir ../../steam-data/step02-output \\
        --output-dir ../../steam-data/figures-output

Writes `top-10-tags-tox.pdf` (the figure, ready to drop into
`main/images/`), `tag_toxicity_all.csv` (every tag, so the low end quoted
in the text is readable too), `tag_toxicity_top.csv` (the ten plotted
tags, which is also what run_tag_tfidf_heatmap.py reads to stay on the
same tag set), and `tag_toxicity_report.json`.
"""
import argparse
from pathlib import Path

import corpus_io as cio
import tag_toxicity as tt
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Top-N game tags by toxicity rate, above a review-volume percentile."
    )
    parser.add_argument("--games", required=True, type=Path, help="step01's games.parquet")
    parser.add_argument("--step02-dir", required=True, type=Path, help="step02's output directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write outputs to")
    parser.add_argument("--lang", default="en", help="Language to analyze (default: en, what the paper reports)")
    parser.add_argument(
        "--volume-quantile", type=float, default=tt.VOLUME_QUANTILE,
        help=f"Review-volume percentile a tag must reach to qualify (default: {tt.VOLUME_QUANTILE})",
    )
    parser.add_argument("--top-n", type=int, default=tt.TOP_N, help=f"Tags to plot (default: {tt.TOP_N})")
    return parser.parse_args()


def main():
    args = parse_args()

    games = cio.load_games(args.games)
    game_counts, stats = tt.per_game_counts(args.step02_dir, args.lang)

    pairs = cio.explode_game_tags(games)
    table = tt.tag_table(game_counts, pairs)
    coverage = tt.tag_coverage(game_counts, pairs)
    qualified, threshold = tt.apply_volume_threshold(table, args.volume_quantile)
    top = qualified.head(args.top_n).reset_index(drop=True)

    info(f"[{args.lang}] Top {len(top)} tag(s) by toxicity rate:")
    print(top[["tag", "n_reviews", "n_toxic", "toxicity_pct"]].to_string(index=False))
    info("Lowest-rate qualifying tags (the contrast the text draws):")
    print(qualified.tail(5)[["tag", "n_reviews", "n_toxic", "toxicity_pct"]].to_string(index=False))

    tt.export_table(table, args.output_dir / "tag_toxicity_all.csv")
    tt.export_table(top, args.output_dir / "tag_toxicity_top.csv")
    figure_path = tt.plot_top_tags(top, args.output_dir / "top-10-tags-tox.pdf")

    save_summary(
        {
            "language": args.lang,
            **stats,
            **coverage,
            "tags_total": len(table),
            "volume_quantile": args.volume_quantile,
            "volume_threshold_reviews": threshold,
            "tags_qualified": len(qualified),
            "top_n": len(top),
            "top_tags": top[["tag", "n_reviews", "n_toxic", "toxicity_pct"]].to_dict("records"),
            "figure_path": str(figure_path),
        },
        args.output_dir / "tag_toxicity_report.json",
    )


if __name__ == "__main__":
    main()
