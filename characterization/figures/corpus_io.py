"""Shared readers for step01's games/users tables and step02's scored
reviews, used by every figure script in this folder.

The toxicity rule here is the same union rule and thresholds as everywhere
else in this project (`perspective_score >= 0.7` OR `detoxify_score >=
0.9`), with rows carrying an invalid/sentinel score dropped rather than
labeled non-toxic - see step02_run_detoxify/toxicity_mask.py and
step03_tfidf_analysis/tfidf_analysis.label_toxicity, which apply the
identical rule. Keeping the three in agreement is what lets a figure and a
table in the same paper describe the same population.

Reading is done one file at a time and reduced immediately, because the
English corpus is ~36.8M reviews: no figure in this folder ever needs the
whole corpus resident, only a per-game or per-user aggregate of it.
"""
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import info, list_parquet_files

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9

# Steam profile URLs come in two shapes. Only the numeric one carries the
# SteamID64 that step01's users table is keyed on (`steam_id`); vanity URLs
# (/id/<name>) cannot be joined to a profile without a separate resolution
# step that this project never ran. This is the mechanical reason only
# 43.7% of users match a collected profile - see the paper's user-level
# modeling dataset section.
STEAMID_RE = re.compile(r"profiles/(\d+)")


def resolve_lang_source(base_dir: Path, lang: str) -> Path:
    """step02's output has been observed in two layouts across this
    project's lifetime: subfolders (base_dir/review_lang=<lang>/*.parquet)
    and flat (every language together in base_dir, with review_lang as a
    column). Checks which shape is actually present rather than hardcoding
    one - same helper as review_examples/show_review_examples.py."""
    base_dir = Path(base_dir)
    subfolder = base_dir / f"review_lang={lang}"
    return subfolder if subfolder.is_dir() else base_dir


def label_toxicity(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with a score outside [0, 1] (Detoxify's -1.0 'failed to
    score' sentinel) and adds `is_toxic` from the union rule. Identical to
    tfidf_analysis.label_toxicity, minus the logging, so step03's tables
    and this folder's figures count the same reviews."""
    valid = df["perspective_score"].between(0, 1) & df["detoxify_score"].between(0, 1)
    out = df[valid].copy()
    out["is_toxic"] = (
        (out["perspective_score"] >= PERSPECTIVE_THRESHOLD)
        | (out["detoxify_score"] >= DETOXIFY_THRESHOLD)
    )
    return out


def iter_scored_reviews(step02_dir: Path, lang: str, columns: list):
    """Yields (labeled_frame, n_read, n_dropped_invalid) one step02 file at
    a time, already filtered to `lang` and already carrying `is_toxic`.

    `columns` are the payload columns the caller needs on top of the two
    score columns; the language columns are added and dropped internally.
    """
    source = resolve_lang_source(step02_dir, lang)
    is_subfolder = source != Path(step02_dir)

    read_columns = list(dict.fromkeys(
        list(columns) + ["perspective_score", "detoxify_score", "perspective_declared_language"]
        + ([] if is_subfolder else ["review_lang"])
    ))

    for path in list_parquet_files(source):
        df = pd.read_parquet(path, columns=read_columns)
        if is_subfolder:
            df["review_lang"] = lang

        n_read = len(df)
        df = df[(df["review_lang"] == lang) & (df["perspective_declared_language"] == lang)]
        df = df.drop(columns=["review_lang", "perspective_declared_language"])

        n_after_mask = len(df)
        df = label_toxicity(df)
        yield df, n_read, n_after_mask - len(df)


def load_games(games_path: Path) -> pd.DataFrame:
    """step01's cleaned games table, slimmed to what the tag figures need."""
    games = pd.read_parquet(games_path, columns=["game_id", "popular_tags"])
    info(f"Loaded {len(games)} game(s) from {games_path}")
    return games


def normalize_tags(value) -> list:
    """`popular_tags` reaches disk as a list in the parquet, but has also
    been observed as a comma-separated string depending on how the games
    table was written. Both are accepted, and anything else (NaN, None)
    becomes an empty list, so a game with no tags is simply absent from
    every tag's population rather than crashing the aggregation."""
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    try:
        return [str(t).strip() for t in value if str(t).strip()]
    except TypeError:
        return []  # not a string and not iterable (e.g. NaN)


def explode_game_tags(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, tag) pair. A game carrying ten tags counts
    toward all ten - tags are not mutually exclusive, and the paper's own
    reading of the Free to Play tag depends on that overlap being kept."""
    tags = games.assign(tag=games["popular_tags"].map(normalize_tags))
    tags = tags[["game_id", "tag"]].explode("tag", ignore_index=True)
    tags = tags[tags["tag"].notna() & (tags["tag"] != "")]
    info(f"{len(tags)} (game, tag) pair(s) across {tags['tag'].nunique()} distinct tag(s)")
    return tags


def load_user_profiles(users_path: Path) -> pd.DataFrame:
    """step01's cleaned users table, slimmed to the three engagement
    metrics and the ban flag the behavioral profile reports."""
    users = pd.read_parquet(
        users_path, columns=["steam_id", "profile_level", "library_size", "has_ban"]
    )
    users["steam_id"] = users["steam_id"].astype("string")
    info(f"Loaded {len(users)} user profile(s) from {users_path}")
    return users


def extract_steam_id(user_url: pd.Series) -> pd.Series:
    """SteamID64 out of a profile URL, or <NA> for vanity URLs. See
    STEAMID_RE's comment for why the vanity case is not recoverable here."""
    return user_url.astype("string").str.extract(STEAMID_RE, expand=False)
