"""CLI for show_review_examples.py - samples example reviews matching a
language + toxicity + (optionally) a text term / game tag, joining in the
perspective/detoxify scores and the game name, and saves the result as CSV.

This is how the reviews quoted in the paper's characterization section were
pulled (e.g. --contains "ass" / --contains "kill" over a given --game-tag).

Usage (bare minimum - every path defaults to this project's established
--lang <lang> conventions, see _default_paths below):
    python run_show_examples.py --lang en --toxic --n 50

Usage (overriding a default, e.g. a different step02 run):
    python run_show_examples.py \\
        --lang en --toxic --n 50 \\
        --step02-dir ../../steam-data/step02-output-v3/review_lang=en \\
        --contains "trash" --game-tag "Competitive" --seed 42 \\
        --output ../../steam-data/examples/toxic_en_trash.csv
"""
import argparse
from pathlib import Path

import show_review_examples as sre
from pipeline_utils import info

# This project's established data-layout conventions (see step02_run_detoxify/
# detoxify_scoring.py) - every path below is derived from --lang alone, so a
# normal run needs no path flags at all. Override any individual one with its
# own flag when pointing at a non-standard location (e.g. a different step02
# run for comparison).
GAMES_PATH = Path("../../steam-data/step01-output/games/games.parquet")


def _default_paths(lang: str) -> dict:
    return {
        "games": GAMES_PATH,
        "step02_dir": Path(f"../../steam-data/step02-output/review_lang={lang}"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Samples example reviews (language + toxicity + optional term/tag filters) for manual inspection."
    )
    parser.add_argument("--lang", required=True, help="Language code - 'en' for everything reported in the paper")
    toxic_group = parser.add_mutually_exclusive_group(required=True)
    toxic_group.add_argument("--toxic", dest="toxic", action="store_true", default=None, help="Sample toxic reviews")
    toxic_group.add_argument("--non-toxic", dest="toxic", action="store_false", help="Sample non-toxic reviews")
    parser.add_argument("--n", required=True, type=int, help="Number of examples to sample")
    parser.add_argument(
        "--games", type=Path, default=None,
        help=f"Path to step01's games.parquet (default: {GAMES_PATH})",
    )
    parser.add_argument(
        "--step02-dir", type=Path, default=None,
        help="Path to step02's output directory (default: step02-output/review_lang=<lang>)",
    )
    parser.add_argument("--contains", default=None, help="Substring the review text must contain (case-insensitive)")
    parser.add_argument("--game-tag", default=None, help="Game must have this popular_tag (case-insensitive)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--light-mode", action="store_true", help="Filter data while reading to drastically reduce RAM usage")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write the sampled examples CSV to "
        "(default: ../../steam-data/examples/<toxic|non_toxic>_<lang>.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = _default_paths(args.lang)

    games_path = args.games or defaults["games"]
    step02_dir = args.step02_dir or defaults["step02_dir"]
    selector_label = "toxic" if args.toxic else "non_toxic"
    output = args.output or Path(f"../../steam-data/examples/{selector_label}_{args.lang}.csv")

    examples = sre.get_review_examples(
        lang=args.lang,
        toxic=args.toxic,
        n=args.n,
        games_path=games_path,
        step02_dir=step02_dir,
        contains=args.contains,
        game_tag=args.game_tag,
        seed=args.seed,
        light_mode=args.light_mode,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output, index=False)
    info(f"Saved {len(examples)} example(s) to: {output}")


if __name__ == "__main__":
    main()
