"""Fits step03's TF-IDF vectorizer over the full English corpus without
holding the corpus in memory.

WHY THIS EXISTS. `step03_tfidf_analysis/run_tfidf.py` concatenates every
review of a language into one DataFrame before fitting. That is fine for a
single-language slice on a big machine, but the English corpus is 36.8M
reviews and ~12.8GB of raw text, so on anything under ~32GB of RAM the load
step dies before the vectorizer is reached. The heatmap still needs that
exact vectorizer: a TF-IDF weight means nothing except relative to the
vocabulary and document frequencies it was fitted against, so fitting on a
subset - or on the toxic reviews alone - would put the figure on a
different scale from the paper's TF-IDF table.

WHAT IT DOES INSTEAD. One streaming pass over the corpus that, per file,
counts document and term frequencies with the same analyzer step03 uses and
writes those counts straight to disk; DuckDB then sums them across files.
The totals are pruned exactly the way scikit-learn prunes them (`min_df`,
`max_df`, then the top `max_features` by term frequency), and the IDF is
computed with scikit-learn's smoothed formula. The result is the same
vocabulary and the same IDF vector a full in-memory fit would produce, so
the transform is interchangeable with step03's - `test_equiv` in the
project notes verifies that on a two-file slice, to 1e-07.

The same pass also stages the cleaned text of every toxic review, because
the heatmap averages over exactly those and re-reading 12.8GB to collect
685k rows would double the cost for nothing.

The fit is cached to disk: it takes roughly an hour, and nothing about it
changes when the figure's terms, tags or styling do.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import corpus_io as cio
from pipeline_utils import info

# step03's settings, restated here only as defaults for the CLI. The
# analyzer itself always comes from step03's module, never from a copy.
MIN_DF = 100
MAX_DF = 0.8
MAX_FEATURES = 100_000


def _analyzer_settings(ta, lang: str):
    """The stopword list step03 fits with, for this language."""
    from sklearn.feature_extraction.text import CountVectorizer

    lang_name = {"pt": "portuguese", "en": "english"}[lang]
    stop_words = ta.build_stop_words(lang_name, ta.STOPWORD_EXTRAS[lang])
    probe = CountVectorizer(stop_words=stop_words, ngram_range=(1, 1))
    return stop_words, probe.build_analyzer()


def stream_corpus(step02_dir: Path, lang: str, ta, stage_dir: Path) -> dict:
    """One pass: stage per-file term statistics and the toxic reviews on
    disk, then reduce the statistics with DuckDB.

    NOTHING CORPUS-SIZED IS HELD IN PYTHON. An earlier version accumulated
    the counts in dictionaries and died 17 files in: the distinct-term count
    was already past 1.4M and still climbing, and a `dict` of Python ints is
    an expensive way to hold several million counters. Here each file's
    counts are written out as a small parquet (its own vocabulary only,
    ~170k rows) and the cross-file sum is left to a query engine that spills
    to disk. Peak memory is one file's document-term matrix, which is the
    same footprint the very first file had.

    Counting is still scikit-learn's CountVectorizer per file, so the
    tokenization, lowercasing and stopword handling are the library's and
    not a hand-rolled imitation - only the reduction moved.
    """
    import duckdb
    from sklearn.feature_extraction.text import CountVectorizer

    stop_words, _ = _analyzer_settings(ta, lang)
    stage_dir = Path(stage_dir)
    counts_dir = stage_dir / "term_counts"
    toxic_dir = stage_dir / "toxic"
    for d in (counts_dir, toxic_dir):
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("*.parquet"):
            stale.unlink()

    n_docs = 0
    n_toxic = 0

    for i, (df, _read, _invalid) in enumerate(
        cio.iter_scored_reviews(step02_dir, lang, ["review_text", "game_id"]), start=1
    ):
        if df.empty:
            continue
        cleaned = ta.clean_text_for_tfidf(df["review_text"])

        counter = CountVectorizer(stop_words=stop_words, ngram_range=(1, 1), dtype=np.int32)
        matrix = counter.fit_transform(cleaned)

        pd.DataFrame({
            "term": counter.get_feature_names_out(),
            "tf": np.asarray(matrix.sum(axis=0)).ravel().astype("int64"),
            "df": np.asarray((matrix > 0).sum(axis=0)).ravel().astype("int64"),
        }).to_parquet(counts_dir / f"part.{i:04d}.parquet", index=False)
        n_docs += matrix.shape[0]
        del matrix, counter

        mask = df["is_toxic"].fillna(False).to_numpy(dtype=bool)
        if mask.any():
            # game_id stays a nullable Int64 Series rather than becoming an
            # object array, so the `.isin(tag_games)` the heatmap does later
            # compares integers to integers and actually matches.
            pd.DataFrame({
                "game_id": cio.normalize_game_id(df.loc[mask, "game_id"]).reset_index(drop=True),
                "review_text_clean": pd.Series(cleaned[mask]).reset_index(drop=True),
            }).to_parquet(toxic_dir / f"part.{i:04d}.parquet", index=False)
            n_toxic += int(mask.sum())

        info(f"  file {i}: {n_docs:,} doc(s), {n_toxic:,} toxic so far")

    info("Reducing per-file term counts with DuckDB ...")
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute(f"SET temp_directory='{(stage_dir / 'duckdb_tmp').as_posix()}'")
    counts = con.execute(f"""
        SELECT term, sum(tf) AS tf, sum(df) AS df
        FROM read_parquet('{(counts_dir / "*.parquet").as_posix()}')
        GROUP BY term
    """).fetch_arrow_table()
    con.close()

    info(f"Streamed {n_docs:,} document(s); {counts.num_rows:,} distinct term(s); "
         f"{n_toxic:,} toxic review(s) staged")
    return {
        "counts": counts, "n_docs": n_docs, "n_toxic": n_toxic,
        "toxic_dir": toxic_dir,
    }


def prune_vocabulary(
    counts, n_docs: int,
    min_df: int = MIN_DF, max_df: float = MAX_DF, max_features: int = MAX_FEATURES,
) -> tuple:
    """scikit-learn's `_limit_features`, applied to the reduced counts.

    The order matters and is the library's: absolute `min_df` and
    proportional `max_df` first, then the surviving terms ranked by TERM
    frequency (not document frequency) and cut at `max_features`. Getting
    that order wrong changes which terms survive at the margin.

    `counts` is an Arrow table of (term, tf, df) - the reduction's output.
    """
    table = counts.to_pandas()
    high = max_df * n_docs if isinstance(max_df, float) else max_df
    kept = table[(table["df"] >= min_df) & (table["df"] <= high)]
    info(
        f"{len(table):,} distinct term(s) -> {len(kept):,} after "
        f"min_df={min_df}, max_df={max_df}"
    )

    if max_features and len(kept) > max_features:
        kept = kept.sort_values(["tf", "term"], ascending=[False, True]).head(max_features)
        info(f"Cut to the top {max_features:,} by term frequency")

    # scikit-learn indexes its vocabulary in sorted term order.
    kept = kept.sort_values("term")
    terms = kept["term"].tolist()
    vocabulary = {t: i for i, t in enumerate(terms)}
    # smooth_idf=True, the TfidfTransformer default step03 relies on.
    idf = np.log((1.0 + n_docs) / (1.0 + kept["df"].to_numpy(dtype="float64"))) + 1.0
    return vocabulary, idf


def build_transformer(vocabulary: dict, idf: np.ndarray, ta, lang: str):
    """A TfidfVectorizer that transforms exactly as step03's fitted one
    would, with the streamed vocabulary and IDF injected."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    stop_words, _ = _analyzer_settings(ta, lang)
    vectorizer = TfidfVectorizer(
        stop_words=stop_words, ngram_range=(1, 1), vocabulary=vocabulary, dtype=np.float32,
    )
    # Builds the internal TfidfTransformer against the fixed vocabulary;
    # the idf it computes from this call is then replaced by the streamed one.
    vectorizer.fit([""])
    vectorizer.idf_ = idf
    return vectorizer


def save_fit(path: Path, vocabulary: dict, idf: np.ndarray, n_docs: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    terms = sorted(vocabulary, key=vocabulary.get)
    np.savez_compressed(path, terms=np.array(terms, dtype=object), idf=idf, n_docs=n_docs)
    info(f"Cached corpus fit ({len(terms):,} terms, {n_docs:,} docs) to {path}")
    return path


def load_fit(path: Path) -> tuple:
    data = np.load(Path(path), allow_pickle=True)
    terms = [str(t) for t in data["terms"]]
    vocabulary = {t: i for i, t in enumerate(terms)}
    n_docs = int(data["n_docs"])
    info(f"Loaded cached corpus fit ({len(terms):,} terms, {n_docs:,} docs) from {path}")
    return vocabulary, data["idf"], n_docs


def collect_toxic(toxic_dir: Path) -> pd.DataFrame:
    """The staged toxic reviews, read back as one frame. Only ~685k rows
    with their cleaned text, so this is the one corpus-derived table small
    enough to be resident."""
    parts = sorted(Path(toxic_dir).glob("*.parquet"))
    frame = (
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if parts else pd.DataFrame(columns=["game_id", "review_text_clean"])
    )
    info(f"Collected {len(frame):,} cleaned toxic review(s) from {toxic_dir}")
    return frame


def save_toxic(path: Path, toxic: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    toxic.to_parquet(path, index=False)
    info(f"Cached {len(toxic):,} cleaned toxic review(s) to {path}")
    return path


def summarize(vocabulary: dict, idf: np.ndarray, n_docs: int, terms_of_interest) -> dict:
    """Small, checkable record of the fit, so a reader can confirm the
    vectorizer behind the figure without re-running the hour-long pass."""
    lookup = {}
    for term in terms_of_interest:
        if term in vocabulary:
            lookup[term] = {"index": vocabulary[term], "idf": float(idf[vocabulary[term]])}
    return {
        "n_documents": n_docs,
        "vocabulary_size": len(vocabulary),
        "idf_min": float(idf.min()),
        "idf_max": float(idf.max()),
        "terms": lookup,
    }
