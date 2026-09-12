"""Recomputes every number the paper's characterization section states, and
checks each one against the published value.

This exists for artifact evaluation. A reviewer should not have to trust
that the figures and the prose agree with the corpus - they should be able
to run one command and see each claim recomputed and compared:

    python characterization/verify_numbers.py --data data

Every check names the claim, the published value, the recomputed value, and
whether they agree to the precision the paper states them at. A claim the
paper rounds to 1.86% passes when the recomputation rounds to 1.86%; a
claim stated to six decimals (the two correlations) must match to six.

Counting rules, identical to the rest of the pipeline:
  - toxic review  : perspective_score >= 0.7 OR toxicity >= 0.9
  - toxic user    : authored at least one toxic review
  - matched user  : review URL carries a SteamID64 that is present in the
                    collected profile table

DuckDB rather than pandas: the English corpus is 36.8M reviews and ~12.8GB
of text, and these are aggregations, so pushing them into a query engine
that streams and spills keeps the whole check inside a few GB of RAM.
"""
import argparse
import json
from pathlib import Path

import duckdb

THR_DETOX, THR_PERSP = 0.9, 0.7

# What the paper states, and how precisely it states it. `places` is the
# number of decimals the comparison is made at, taken from the paper's own
# formatting - comparing a rounded claim at full precision would fail every
# time for no reason.
CLAIMS = {
    "reviews":                 {"paper": 36_823_127, "places": 0},
    "users":                   {"paper": 14_183_630, "places": 0},
    "games_with_en_review":    {"paper": 67_477,     "places": 0},
    "toxic_reviews":           {"paper": 685_536,    "places": 0},
    "toxic_reviews_pct":       {"paper": 1.86,       "places": 2},
    "toxic_users":             {"paper": 560_079,    "places": 0},
    "toxic_users_pct":         {"paper": 3.95,       "places": 2},
    "users_matched":           {"paper": 6_204_110,  "places": 0},
    "users_matched_pct":       {"paper": 43.7,       "places": 1},
    "pearson":                 {"paper": 0.770234,   "places": 6},
    "spearman":                {"paper": 0.606591,   "places": 6},
    "toxic_recommended_pct":   {"paper": 57.5,       "places": 1},
    "median_reviews_toxic":    {"paper": 3,          "places": 0},
    "median_reviews_nontoxic": {"paper": 1,          "places": 0},
    "median_library_toxic":    {"paper": 102,        "places": 0},
    "median_library_nontoxic": {"paper": 74,         "places": 0},
    "median_level_toxic":      {"paper": 12,         "places": 0},
    "median_level_nontoxic":   {"paper": 10,         "places": 0},
    "ban_pct_toxic":           {"paper": 1.4,        "places": 1},
    "ban_pct_nontoxic":        {"paper": 1.2,        "places": 1},
    "tag_team_based_pct":      {"paper": 2.93,       "places": 2},
    "tag_competitive_pct":     {"paper": 2.90,       "places": 2},
    "tag_pvp_pct":             {"paper": 2.57,       "places": 2},
    "tag_exploration_pct":     {"paper": 1.22,       "places": 2},
    "tag_puzzle_pct":          {"paper": 1.26,       "places": 2},
    "tag_cute_pct":            {"paper": 1.53,       "places": 2},
}


def connect(data: Path, threads: int, memory: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory}'")
    tmp = data / "_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def compute(con, data: Path) -> dict:
    en = (data / "corpus/reviews_w_detoxify/en/*.parquet").as_posix()
    games = (data / "corpus/games/games.parquet").as_posix()
    users = (data / "corpus/users/all_users.parquet").as_posix()
    tox = f"(r.toxicity >= {THR_DETOX} OR r.perspective_score >= {THR_PERSP})"
    got = {}

    print("[1/4] corpus size, labels, recommendation split ...", flush=True)
    row = con.execute(f"""
        SELECT count(*), count(DISTINCT r.user_url), count(DISTINCT r.game_id),
               sum(CASE WHEN {tox} THEN 1 ELSE 0 END),
               count(DISTINCT CASE WHEN {tox} THEN r.user_url END),
               sum(CASE WHEN {tox} AND r.is_recommended THEN 1 ELSE 0 END)
        FROM read_parquet('{en}') r
    """).fetchone()
    reviews, n_users, n_games, tox_rev, tox_users, tox_rec = row
    got.update({
        "reviews": reviews, "users": n_users, "games_with_en_review": n_games,
        "toxic_reviews": tox_rev, "toxic_reviews_pct": 100 * tox_rev / reviews,
        "toxic_users": tox_users, "toxic_users_pct": 100 * tox_users / n_users,
        "toxic_recommended_pct": 100 * tox_rec / tox_rev,
    })

    print("[2/4] Perspective/Detoxify agreement ...", flush=True)
    valid = f"""FROM read_parquet('{en}')
                WHERE toxicity BETWEEN 0 AND 1 AND perspective_score BETWEEN 0 AND 1"""
    got["pearson"] = con.execute(
        f"SELECT corr(toxicity, perspective_score) {valid}"
    ).fetchone()[0]
    # Average ranks for ties, which is what scipy.stats.spearmanr computes.
    got["spearman"] = con.execute(f"""
        SELECT corr(rt, rp) FROM (
          SELECT rank() OVER (ORDER BY toxicity) +
                 (count(*) OVER (PARTITION BY toxicity) - 1) / 2.0 AS rt,
                 rank() OVER (ORDER BY perspective_score) +
                 (count(*) OVER (PARTITION BY perspective_score) - 1) / 2.0 AS rp
          {valid}
        )
    """).fetchone()[0]

    print("[3/4] per-tag toxicity rates ...", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TABLE tag_counts AS
        SELECT t.tag, sum(c.n_reviews) AS n_reviews, sum(c.n_toxic) AS n_toxic
        FROM (SELECT TRY_CAST(r.game_id AS BIGINT) AS gid, count(*) AS n_reviews,
                     sum(CASE WHEN {tox} THEN 1 ELSE 0 END) AS n_toxic
              FROM read_parquet('{en}') r GROUP BY 1) c
        JOIN (SELECT game_id AS gid, unnest(popular_tags) AS tag
              FROM read_parquet('{games}')) t USING (gid)
        GROUP BY 1
    """)
    for key, tag in [
        ("tag_team_based_pct", "Team-Based"), ("tag_competitive_pct", "Competitive"),
        ("tag_pvp_pct", "PvP"), ("tag_exploration_pct", "Exploration"),
        ("tag_puzzle_pct", "Puzzle"), ("tag_cute_pct", "Cute"),
    ]:
        got[key] = con.execute(
            f"SELECT 100.0*n_toxic/n_reviews FROM tag_counts WHERE tag = '{tag}'"
        ).fetchone()[0]

    print("[4/4] user engagement, matching and bans ...", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TABLE user_joined AS
        SELECT u.n_reviews, (u.n_toxic >= 1) AS is_toxic_user,
               p.profile_level, p.library_size, p.has_ban,
               (p.steam_id IS NOT NULL) AS matched
        FROM (SELECT r.user_url, count(*) AS n_reviews,
                     sum(CASE WHEN {tox} THEN 1 ELSE 0 END) AS n_toxic,
                     nullif(regexp_extract(r.user_url, 'profiles/([0-9]+)', 1), '') AS steam_id
              FROM read_parquet('{en}') r GROUP BY 1) u
        LEFT JOIN read_parquet('{users}') p USING (steam_id)
    """)
    matched = con.execute("SELECT count(*) FROM user_joined WHERE matched").fetchone()[0]
    got["users_matched"] = matched
    got["users_matched_pct"] = 100 * matched / got["users"]

    for flag, suffix in ((True, "toxic"), (False, "nontoxic")):
        row = con.execute(f"""
            SELECT median(n_reviews),
                   median(library_size) FILTER (matched),
                   median(profile_level) FILTER (matched),
                   count(*),
                   count(*) FILTER (matched AND has_ban)
            FROM user_joined WHERE is_toxic_user = {str(flag).lower()}
        """).fetchone()
        med_rev, med_lib, med_lvl, n_group, n_banned = row
        got[f"median_reviews_{suffix}"] = med_rev
        got[f"median_library_{suffix}"] = med_lib
        got[f"median_level_{suffix}"] = med_lvl
        # The paper's denominator is every user in the group: a profile that
        # was never collected cannot show a ban, and is counted as unbanned.
        got[f"ban_pct_{suffix}"] = 100 * n_banned / n_group
    return got


def report(got: dict) -> bool:
    print(f"\n{'claim':26s} {'paper':>14s} {'recomputed':>16s}   status")
    print("-" * 68)
    ok = True
    for key, spec in CLAIMS.items():
        value = got.get(key)
        if value is None:
            print(f"{key:26s} {'-':>14} {'NOT COMPUTED':>16}   MISSING")
            ok = False
            continue
        places = spec["places"]
        agrees = round(float(value), places) == round(float(spec["paper"]), places)
        ok &= agrees
        fmt = f"{{:>16.{places}f}}" if places else "{:>16,.0f}"
        paper_fmt = f"{{:>14.{places}f}}" if places else "{:>14,.0f}"
        print(
            f"{key:26s} {paper_fmt.format(float(spec['paper']))} "
            f"{fmt.format(float(value))}   {'OK' if agrees else 'MISMATCH'}"
        )
    print("-" * 68)
    print("ALL CLAIMS REPRODUCED" if ok else "SOME CLAIMS DO NOT REPRODUCE")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Recompute and check the paper's characterization numbers."
    )
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="Corpus root (default: data)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the recomputed values as JSON")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="11GB")
    args = parser.parse_args()

    con = connect(args.data, args.threads, args.memory)
    got = compute(con, args.data)
    ok = report(got)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            k: {"paper": CLAIMS[k]["paper"], "recomputed": got.get(k),
                "decimals_compared": CLAIMS[k]["places"]}
            for k in CLAIMS
        }
        args.output.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nWrote {args.output}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
