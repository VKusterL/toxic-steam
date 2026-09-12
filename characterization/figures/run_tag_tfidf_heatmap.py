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
import corpus_vectorizer as cv
import tag_tfidf_heatmap as hm
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heatmap of mean TF-IDF weights for selected terms across game tags."
    )
    parser.add_argument(
        "--labeled", type=Path, default=None,
        help="step03's reviews_cleaned_labeled_<lang>.parquet (small-corpus path; "
             "use --step02-dir instead when the corpus does not fit in memory)",
    )
    parser.add_argument(
        "--step02-dir", type=Path, default=None,
        help="Scored review corpus, streamed rather than loaded (the only path that "
             "works on the full 36.8M-review corpus). Mutually exclusive with --labeled.",
    )
    parser.add_argument(
        "--fit-cache", type=Path, default=None,
        help="Where to cache the streamed corpus fit (.npz) and the cleaned toxic "
             "reviews (.parquet). Reused on later runs, which is what makes tuning "
             "the figure cheap.",
    )
    parser.add_argument(
        "--lexicon", type=Path, default=None,
        help="step03's tfidf_lexicon_<lang>.csv, read only to pick the default terms",
    )
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


def fit_by_streaming(args, ta, terms):
    """Streams the corpus to fit the vectorizer and collect the toxic
    reviews, reusing a cached fit when one is present.

    The cache is what makes this practical: the pass costs about an hour,
    and none of it depends on which terms or tags the figure ends up
    showing, so a second run that only changes the figure reuses it.
    """
    fit_path = args.fit_cache
    toxic_path = fit_path.with_suffix(".toxic.parquet") if fit_path else None

    if fit_path and fit_path.exists() and toxic_path.exists():
        vocabulary, idf, n_docs = cv.load_fit(fit_path)
        reviews = pd.read_parquet(toxic_path)
        reviews["is_toxic"] = True
        info(f"Reusing {len(reviews):,} cached toxic review(s) from {toxic_path}")
    else:
        info(f"[{args.lang}] Streaming the corpus to fit step03's vectorizer "
             "(one pass, ~1h on the full English corpus)...")
        stage = (fit_path.parent / "stage") if fit_path else Path("./_fit_stage")
        streamed = cv.stream_corpus(args.step02_dir, args.lang, ta, stage)
        vocabulary, idf = cv.prune_vocabulary(streamed["counts"], streamed["n_docs"])
        n_docs = streamed["n_docs"]
        reviews = cv.collect_toxic(streamed["toxic_dir"])
        reviews["is_toxic"] = True
        if fit_path:
            cv.save_fit(fit_path, vocabulary, idf, n_docs)
            cv.save_toxic(toxic_path, reviews)

    vectorizer = cv.build_transformer(vocabulary, idf, ta, args.lang)
    return vectorizer, reviews, cv.summarize(vocabulary, idf, n_docs, terms)


def main():
    args = parse_args()
    ta = hm.import_step03(args.step03_code)

    if (args.labeled is None) == (args.step02_dir is None):
        raise SystemExit("Pass exactly one of --labeled or --step02-dir.")
    if args.terms is None and args.lexicon is None:
        raise SystemExit("Pass --lexicon (to pick default terms) or one or more --term.")

    tags = resolve_tags(args)
    terms = hm.select_terms(args.lexicon, args.n_terms, args.terms)
    games = cio.load_games(args.games)

    if args.step02_dir is not None:
        vectorizer, reviews, fit_summary = fit_by_streaming(args, ta, terms)
    else:
        reviews = hm.load_labeled_reviews(args.labeled)
        info(f"[{args.lang}] Re-fitting step03's vectorizer in memory...")
        stop_words = ta.build_stop_words(hm.LANG_NLTK_NAME[args.lang], ta.STOPWORD_EXTRAS[args.lang])
        vectorizer = ta.fit_vocabulary(reviews["review_text_clean"].fillna("").astype(str), stop_words)
        fit_summary = {"vocabulary_size": int(len(vectorizer.get_feature_names_out()))}
    info(f"[{args.lang}] Vocabulary size: {fit_summary['vocabulary_size']:,}")

    means, counts = hm.tag_term_means(reviews, games, tags, terms, vectorizer)

    # Strongest term on the bottom row, as published. Ordering by the data
    # rather than by the order the terms were typed keeps the figure honest
    # when the corpus changes underneath it.
    means = means.loc[means.mean(axis=1).sort_values(ascending=False).index]

    info("Mean TF-IDF per term per tag:")
    print(means.round(6).to_string())

    claims = hm.check_paper_claims(means)
    hm.report_claims(claims)

    hm.export_means(means, args.output_dir / "tag_tfidf_means.csv")
    figure_path = hm.plot_heatmap(means, args.output_dir / "heatmap_tfidf_tags.pdf")

    save_summary(
        {
            "language": args.lang,
            "source": str(args.step02_dir or args.labeled),
            "lexicon_path": str(args.lexicon) if args.lexicon else None,
            "corpus_fit": fit_summary,
            "tags": tags,
            "terms": list(means.index),
            "toxic_reviews_per_tag": counts,
            "paper_claims": claims,
            "figure_path": str(figure_path),
        },
        args.output_dir / "tag_tfidf_report.json",
    )


if __name__ == "__main__":
    main()
