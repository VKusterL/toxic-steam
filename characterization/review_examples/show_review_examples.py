"""Generates example-message files for manual inspection - sampled reviews
matching a language + toxicity + (optionally) a text term / game tag, with
the score data joined in.

These are the examples quoted verbatim in the paper's characterization
section: the reviews illustrating contextual toxicity ("people"),
performative toxicity ("ass", "kill"), and the per-tag lexical signatures.

Pulls from two sources, cross-referenced by review_url/game_id:
  - step02's output (review_text, perspective_score, detoxify_score,
    review_url, game_id, perspective_declared_language, plus every other
    column step01 originally scraped - review_date, hours_played,
    is_recommended, user_url, detection_confidence) - the base table this
    filters and samples from.
  - step01's games.parquet (game title, popular_tags) - joined by game_id,
    for the game_name column and the optional game_tag filter.

step01/step02's output is flat (review_lang is a plain column, not a
directory partition - see step02_run_detoxify/detoxify_scoring.py's
module docstring): every language is scored together in the same file, so
`review_lang == lang AND perspective_declared_language == lang` is what
actually selects this language's rows, not just a defensive double-check -
applied explicitly in load_scored_reviews below, same as step03's
tfidf_analysis.py.

Toxicity uses the same union rule and thresholds as everywhere else in
this project (perspective_score >= 0.7 OR detoxify_score >= 0.9, rows with
an invalid/sentinel score excluded rather than labeled non-toxic).
"""
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import info, list_parquet_files

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9

# Same boilerplate patterns stripped before scoring in step02/step03 -
# review_text itself is never altered anywhere in this project, so this is
# applied to a separate review_text_clean column here, purely so examples
# show what the models actually saw, not a change to review_text.
BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]

OUTPUT_COLUMNS = [
    "game_id", "game_name", "review_url", "user_url", "review_date", "is_recommended",
    "hours_played", "review_text", "review_text_clean", "review_lang", "detection_confidence",
    "perspective_score", "detoxify_score",
]


def clean_review_text(text):
    """Strips known boilerplate phrases from review text (case-insensitive).
    Non-string input (e.g. NaN) passes through unchanged. Same logic as
    detoxify_scoring.py's clean_review_text."""
    if not isinstance(text, str):
        return text
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_games(games_path: Path) -> pd.DataFrame:
    return pd.read_parquet(games_path, columns=["game_id", "title", "popular_tags"])


def _resolve_lang_source(base_dir: Path, lang: str) -> Path:
    """step02's output has been observed in two different layouts across
    this project's lifetime: subfolders (base_dir/review_lang=<lang>/
    *.parquet) and flat (every language together in base_dir directly, with
    review_lang as a column). Rather than hardcode one, check which shape
    is actually present and use that - avoids silently breaking again if
    the layout changes."""
    subfolder = base_dir / f"review_lang={lang}"
    return subfolder if subfolder.is_dir() else base_dir


def load_scored_reviews(step02_dir: Path, lang: str, chunk_filter_fn=None) -> pd.DataFrame:
    """Base table: every step02-scored review for one language. Handles
    both the subfolder layout (read the review_lang=<lang>/ subfolder
    directly - review_lang isn't a real column there) and the flat layout
    (every language together, filtered by the review_lang column) - see
    _resolve_lang_source."""
    source = _resolve_lang_source(step02_dir, lang)
    is_subfolder = source != step02_dir

    files = list_parquet_files(source)
    columns = [
        "review_url", "review_text", "game_id", "perspective_score", "detoxify_score",
        "user_url", "review_date", "is_recommended", "hours_played", "detection_confidence",
    ]
    if not is_subfolder:
        columns += ["review_lang"]
    columns += ["perspective_declared_language"]

    frames = []
    total_excluded = 0
    for f in files:
        df = pd.read_parquet(f, columns=columns)
        if is_subfolder:
            df["review_lang"] = lang

        rows_before_mask = len(df)
        df = df[
            (df["review_lang"] == lang) & (df["perspective_declared_language"] == lang)
        ].copy()
        total_excluded += (rows_before_mask - len(df))
        
        df = df.drop(columns=["perspective_declared_language"])
        
        if chunk_filter_fn:
            df = chunk_filter_fn(df)
            
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[c for c in columns if c != "perspective_declared_language"])

    if total_excluded:
        info(
            f"[{lang}] Excluded {total_excluded} row(s) not matching "
            f"review_lang == perspective_declared_language == '{lang}'"
        )

    return df


def _game_has_tag(tags, game_tag: str) -> bool:
    if tags is None:
        return False
    game_tag = game_tag.lower()
    if isinstance(tags, str):
        return game_tag in tags.lower()
    try:
        return any(game_tag in str(t).lower() for t in tags)
    except TypeError:
        return False  # not a string and not iterable (e.g. NaN)


def filter_reviews(
    df: pd.DataFrame,
    games: pd.DataFrame,
    toxic: bool,
    contains: str = None,
    game_tag: str = None,
) -> pd.DataFrame:
    """Applies toxicity labeling (same union rule/thresholds/invalid-score
    exclusion as elsewhere in this project), then the optional text/tag
    filters. Returns the filtered rows, NOT yet sampled."""
    if toxic is None:
        raise ValueError("`toxic` must be given (True for toxic, False for non-toxic)")

    perspective_valid = df["perspective_score"].between(0, 1)
    detoxify_valid = df["detoxify_score"].between(0, 1)
    df = df[perspective_valid & detoxify_valid].copy()

    is_toxic = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)
    df = df[is_toxic] if toxic else df[~is_toxic]

    if contains:
        df = df[df["review_text"].str.contains(contains, case=False, na=False, regex=False)]

    if game_tag:
        tagged_game_ids = set(
            games.loc[games["popular_tags"].apply(lambda t: _game_has_tag(t, game_tag)), "game_id"]
        )
        df = df[df["game_id"].isin(tagged_game_ids)]

    return df


def sample_reviews(df: pd.DataFrame, n: int, seed: int = None) -> pd.DataFrame:
    if len(df) <= n:
        info(f"Only {len(df)} matching review(s) available, requested {n} - returning all of them")
        return df.copy()
    return df.sample(n=n, random_state=seed)


def attach_game_names(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        games[["game_id", "title"]].rename(columns={"title": "game_name"}), on="game_id", how="left"
    )


def get_review_examples(
    lang: str,
    n: int,
    games_path: Path,
    step02_dir: Path,
    toxic: bool,
    contains: str = None,
    game_tag: str = None,
    seed: int = None,
    light_mode: bool = False,
) -> pd.DataFrame:
    """Main entry point - see module docstring for what's pulled from where.

    Args:
        lang: language code - "en" for everything reported in the paper.
        n: number of examples to sample (returns fewer if not enough match).
        games_path: path to step01's games.parquet.
        step02_dir: path to step02's output directory.
        toxic: True samples toxic reviews, False samples non-toxic ones.
        contains: optional substring the review text must contain
            (case-insensitive).
        game_tag: optional popular_tag the game must have (case-insensitive).
        seed: optional random seed for reproducible sampling.
        light_mode: if True, filters files one by one to drastically reduce RAM usage.
    """
    games = load_games(games_path)

    if light_mode:
        def filter_chunk(chunk_df):
            return filter_reviews(chunk_df, games, toxic=toxic, contains=contains, game_tag=game_tag)
        filtered = load_scored_reviews(step02_dir, lang, chunk_filter_fn=filter_chunk)
    else:
        reviews = load_scored_reviews(step02_dir, lang)
        filtered = filter_reviews(reviews, games, toxic=toxic, contains=contains, game_tag=game_tag)

    info(f"[{lang}] {len(filtered)} review(s) match (toxic={toxic}, contains={contains!r}, game_tag={game_tag!r})")

    sample = sample_reviews(filtered, n=n, seed=seed)
    sample["review_text_clean"] = sample["review_text"].apply(clean_review_text)
    sample = attach_game_names(sample, games)

    available = [c for c in OUTPUT_COLUMNS if c in sample.columns]
    return sample[available].reset_index(drop=True)
