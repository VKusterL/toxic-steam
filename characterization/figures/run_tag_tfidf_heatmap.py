"""CLI for tag_tfidf_heatmap.py - mean TF-IDF salience across tag contexts.

Runs AFTER step03 (it reuses step03's cleaned/labeled intermediate and its
lexicon) and AFTER run_tag_toxicity.py (it reads that script's top-tag CSV
so both figures describe the same ten tags).

Usage:
    python run_tag_tfidf_heatmap.py \\
        --labeled ../../steam-data/step03-output/reviews_cleaned_labeled_en.parquet \\
        --lexicon ../../steam-data/step03-output/tfidf_lexicon_en.csv \\
        --games ../../steam-data/step01-output/games/games.parquet \\
        --top-tags ../../steam-data/figures-output/tag_toxicity_top.csv \\
        --output-dir ../../steam-data/figures-output

Writes `heatmap_tfidf_tags.pdf`, `tag_tfidf_means.csv` (the plotted matrix,
so a cell can be quoted without re-reading the figure) and
`tag_tfidf_report.json`.

Re-fitting the vectorizer is the expensive part of this script (it fits on
the full language corpus, matching step03 exactly - see the module
docstring for why that is not optional).
"""
import argparse
from pathlib import Path

import pandas as pd

import corpus_io as cio
import tag_tfidf_heatmap as hm
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heatmap of mean TF-IDF weights for selected terms across game tags."
    )
    parser.add_argument(
        "--labeled", required=True, type=Path,
        help="step03's reviews_cleaned_labeled_<lang>.parquet",
    )
    parser.add_argument("--lexicon", required=True, type=Path, help="step03's tfidf_lexicon_<lang>.csv")
    parser.add_argument("--games", required=True, type=Path, help="step01's games.parquet")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write outputs to")
    parser.add_argument("--lang", default="en", help="Language being analyzed (default: en)")
    parser.add_argument(
        "--top-tags", type=Path, default=None,
        help="run_tag_toxicity.py's tag_toxicity_top.csv - the tags to use as columns",
    )
    parser.add_argument(
        "--tag", action="append", dest="tags", default=None,
        help="Explicit tag to include (repeatable). Overrides --top-tags.",
    )
    parser.add_argument(
        "--term", action="append", dest="terms", default=None,
        help="Explicit term to include as a row (repeatable). Overrides --n-terms.",
    )
    parser.add_argument(
        "--n-terms", type=int, default=hm.DEFAULT_N_TERMS,
        help=f"Top terms by `diferenca` to use as rows (default: {hm.DEFAULT_N_TERMS})",
    )
    parser.add_argument(
        "--step03-code", type=Path, default=Path("../step03_tfidf_analysis"),
        help="Path to step03_tfidf_analysis/ (its tfidf_analysis.py is imported for the shared "
             "cleaning and vectorizer settings; default: ../step03_tfidf_analysis)",
    )
    return parser.parse_args()


def resolve_tags(args) -> list:
    if args.tags:
        info(f"Using {len(args.tags)} explicitly requested tag(s)")
        return list(args.tags)
    if not args.top_tags:
        raise SystemExit("Pass --top-tags (run_tag_toxicity.py's output) or one or more --tag.")
    tags = pd.read_csv(args.top_tags)["tag"].tolist()
    info(f"Using {len(tags)} tag(s) from {args.top_tags}")
    return tags


def main():
    args = parse_args()
    ta = hm.import_step03(args.step03_code)

    tags = resolve_tags(args)
    terms = hm.select_terms(args.lexicon, args.n_terms, args.terms)

    games = cio.load_games(args.games)
    reviews = hm.load_labeled_reviews(args.labeled)

    info(f"[{args.lang}] Re-fitting step03's vectorizer on the full corpus (this is the slow part)...")
    stop_words = ta.build_stop_words(hm.LANG_NLTK_NAME[args.lang], ta.STOPWORD_EXTRAS[args.lang])
    vectorizer = ta.fit_vocabulary(reviews["review_text_clean"].fillna("").astype(str), stop_words)
    info(f"[{args.lang}] Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    means, counts = hm.tag_term_means(reviews, games, tags, terms, vectorizer)

    info("Mean TF-IDF per term per tag:")
    print(means.round(6).to_string())

    hm.export_means(means, args.output_dir / "tag_tfidf_means.csv")
    figure_path = hm.plot_heatmap(means, args.output_dir / "heatmap_tfidf_tags.pdf")

    save_summary(
        {
            "language": args.lang,
            "labeled_path": str(args.labeled),
            "lexicon_path": str(args.lexicon),
            "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
            "tags": tags,
            "terms": list(means.index),
            "toxic_reviews_per_tag": counts,
            "figure_path": str(figure_path),
        },
        args.output_dir / "tag_tfidf_report.json",
    )


if __name__ == "__main__":
    main()
