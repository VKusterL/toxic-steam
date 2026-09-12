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
import pyarrow.parquet as pq

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


# The Detoxify score reaches disk under two names: `detoxify_score` in the
# step02 scripts' own output, and `toxicity` in the archived corpus under
# data/corpus/reviews_w_detoxify (which keeps Detoxify's original field
# name). They are the same number; whichever is present is renamed to
# `detoxify_score` on read so everything downstream sees one name.
DETOXIFY_COLUMNS = ("detoxify_score", "toxicity")


def resolve_lang_source(base_dir: Path, lang: str) -> Path:
    """step02's output has been observed in three layouts across this
    project's lifetime: `base_dir/review_lang=<lang>/*.parquet`,
    `base_dir/<lang>/*.parquet` (the archived corpus), and flat (every
    language together in base_dir, with review_lang as a column). Checks
    which shape is actually present rather than hardcoding one."""
    base_dir = Path(base_dir)
    for candidate in (base_dir / f"review_lang={lang}", base_dir / lang):
        if candidate.is_dir():
            return candidate
    return base_dir


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
    a time, already restricted to `lang` and already carrying `is_toxic`.

    `columns` are the payload columns the caller needs on top of the two
    score columns; the score and language columns are handled internally.

    The language restriction comes from whichever mechanism the layout on
    disk provides. When the files carry `review_lang` and
    `perspective_declared_language`, both must equal `lang` - the agreement
    mask step02 applies before writing. When the files are partitioned into
    a per-language folder instead, that partition *is* the restriction and
    there is nothing left to filter, so re-applying a mask would only drop
    rows for lacking a column the layout made redundant.
    """
    source = resolve_lang_source(step02_dir, lang)
    available = set(pq.ParquetFile(list_parquet_files(source)[0]).schema_arrow.names)

    detox_column = next((c for c in DETOXIFY_COLUMNS if c in available), None)
    if detox_column is None:
        raise SystemExit(
            f"No Detoxify score column in {source}: expected one of {DETOXIFY_COLUMNS}."
        )

    lang_columns = [c for c in ("review_lang", "perspective_declared_language") if c in available]
    read_columns = list(dict.fromkeys(
        list(columns) + ["perspective_score", detox_column] + lang_columns
    ))

    for path in list_parquet_files(source):
        df = pd.read_parquet(path, columns=read_columns)
        if detox_column != "detoxify_score":
            df = df.rename(columns={detox_column: "detoxify_score"})

        n_read = len(df)
        for column in lang_columns:
            df = df[df[column] == lang]
        df = df.drop(columns=lang_columns)

        n_after_mask = len(df)
        df = label_toxicity(df)
        yield df, n_read, n_after_mask - len(df)


def normalize_game_id(values: pd.Series) -> pd.Series:
    """`game_id` is a Steam AppID, but it is stored as a string in the
    review tables and as an integer in the games table. Joining the two
    without a cast silently matches nothing and every tag comes back empty,
    so both sides are coerced to the same nullable integer here. Anything
    non-numeric becomes <NA> and simply fails to join, rather than joining
    to the wrong game."""
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def load_games(games_path: Path) -> pd.DataFrame:
    """step01's cleaned games table, slimmed to what the tag figures need."""
    games = pd.read_parquet(games_path, columns=["game_id", "popular_tags"])
    games["game_id"] = normalize_game_id(games["game_id"])
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
