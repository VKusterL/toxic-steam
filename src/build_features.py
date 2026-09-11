#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_features.py  -  Building the feature tables

Materializes, from the parquet files, three artifacts ready for modeling:

  1) reviews_resolved/        (Task A, review level) - English-only SCOPE (--langs en)
        - deduplicated by review_url
        - join reviews<->users by steam_id extracted from the URL (flag `joinable`)
        - label y_review = (toxicity>=0.9 OR perspective_score>=0.7)
        - surface features (RE2-safe) + game context (via games)
        - split by USER (stable key; group = steam_id or 'v:'+vanity)

  2) review_train_sample.parquet  - balanced sample of split=train (all positives
        + neg_ratio x negatives) to train a transformer/LLM without loading 40M rows.
        (The test set keeps the real prevalence for an honest evaluation.)

  3) user_features.parquet    (user level) - 1 row per joinable user
        - y_user = MAX(y_review) - "produced >=1 toxic review"
        - ONLY behavioral features/metadata (no score; no review text)
        - NULL becomes 0 in the counts; missingness flags where NULL != 0
        - genre/tag profile 

Also writes: feature_dictionary.csv, feature_cols.py (lists for the modeler),
build_report.txt.

Usage:
    python src/build_features.py --root <root (D:\...)> --out data/features
    python src/build_features.py --root . --neg-ratio 4 --memory-limit 8GB --threads 4

NOTE: dedup is done by physical key (file_row_number),
so the window operation does not load the text, hence low memory. The pipeline writes 1-2
intermediate parquets; reserve ~25GB of free disk. Out-of-memory mitigations already built in:
dedup by narrow key, separate stages, preserve_insertion_order=false and threads=4 (default). If
you still run short on RAM, use --threads 2 and make sure --temp-dir points to a disk with space
(DuckDB spills there).

Dependencies: duckdb, pandas, pyarrow
"""
from __future__ import annotations
import argparse, os, sys, glob, re, io

import duckdb
import pandas as pd

# label thresholds (inherited from the previous work)
THR_DETOX = 0.9
THR_PERSP = 0.7

_BUF = io.StringIO()
def out(*a, **k):
    print(*a, **k); print(*a, **k, file=_BUF)
def h1(t): out("\n" + "=" * 78); out(t); out("=" * 78)
def h2(t): out("\n" + "-" * 78); out(t); out("-" * 78)
def show(df, n=60):
    if df is None or len(df) == 0: out("(empty)"); return
    with pd.option_context("display.max_rows", n, "display.width", 200, "display.max_colwidth", 70):
        out(df.to_string(index=False))

# ----------------------------------------------------------------------------- 
def discover(root):
    pq = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    users, games, detox = [], [], {}
    for p in pq:
        rel = os.path.relpath(p, root).replace("\\", "/").lower()
        if "data/corpus/reviews_w_detoxify" in rel:
            detox[p] = "pt" if ("/pt" in rel or "lang=pt" in rel) else ("en" if ("/en" in rel or "lang=en" in rel) else "?")
        elif "all_users" in rel or "/users/" in rel:
            users.append(p)
        elif os.path.basename(rel) == "games.parquet" or "/games/" in rel:
            games.append(p)
    return users, games, detox

def lst(files):  # list of files for the read_parquet argument
    return "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in files) + "]"

# ----------------------------------------------------------------------------- 
def build_reviews_resolved(con, detox, users, games, out_dir, gold_csv, keep_stage=False, sleep_between=0.0, dedup_buckets=16):
    import time, shutil
    h1("STAGE 1  reviews_resolved  (global dedup + PER-FILE processing, resumable)")
    detox_items = list(detox.items())          # [(path, lang), ...]
    gusers, ggames = lst(users), lst(games)

    # view only for the global dedup (1a-i): reads all files, with file_id + file_row_number
    union = " UNION ALL BY NAME ".join(
        f"SELECT review_url, review_date, {i} AS file_id, file_row_number "
        f"FROM read_parquet('{p.replace(chr(39),chr(39)*2)}', file_row_number=true)"
        for i, (p, _lang) in enumerate(detox_items))
    con.sql(f"CREATE OR REPLACE TEMP VIEW rev_keys AS {union}")
    n_raw = con.sql("SELECT count(*) FROM rev_keys").fetchone()[0]
    out(f"raw reviews (with duplicates): {n_raw:,}")

    # optional gold (review_url annotated by a human)
    gold_ok = bool(gold_csv and os.path.exists(gold_csv))
    if gold_ok:
        con.sql(f"CREATE OR REPLACE TEMP VIEW gold AS SELECT DISTINCT review_url, y_review_gold FROM read_csv_auto('{gold_csv}')")
        out(f"gold set: {con.sql('SELECT count(*) FROM gold').fetchone()[0]:,} annotated review_url")
    gold_sel  = "gd.y_review_gold AS y_review_gold," if gold_ok else ""
    gold_join = "LEFT JOIN gold gd ON r.review_url = gd.review_url" if gold_ok else ""

    # ---- 1a-i: GLOBAL dedup keys processed in BUCKETS by hash (bounded memory) ----
    # Deduplicating ~44M distinct review_url does not fit in RAM if all the URLs (long
    # strings) are resident at once - both the window and a single GROUP BY review_url
    # blow up (the keys alone already exceed 3GB). Solution: partition by
    # hash(review_url) % N_BUCKETS and deduplicate ONE bucket per pass - each pass sees only
    # ~1/N of the groups, so memory is bounded by construction. RESUMABLE per bucket.
    # Cost: re-reads the key columns N times (the parquet reads only review_url/review_date,
    # never the text nor the scores, so each pass is cheap).
    keep = os.path.join(out_dir, "_stage_keep.parquet")
    keep_dir = os.path.join(out_dir, "_stage_keep_parts")
    if os.path.exists(keep):
        n_dedup = con.sql(f"SELECT count(*) FROM read_parquet('{keep}')").fetchone()[0]
        out(f"[1a-i] reusing already-computed keys: {n_dedup:,}  (delete {os.path.basename(keep)} to redo)")
    else:
        os.makedirs(keep_dir, exist_ok=True)
        nb = max(1, int(dedup_buckets))
        out(f"[1a-i] global dedup in {nb} bucket(s) by hash(review_url)  (resumable per bucket)")
        for b in range(nb):
            part = os.path.join(keep_dir, f"keep_{b:03d}.parquet")
            if os.path.exists(part):
                out(f"   bucket {b+1}/{nb}: already exists, skipping"); continue
            # arg_min over a struct guarantees that file_id and file_row_number come from the
            # same winning row (earliest date; null dates become 9999-12-31 = chosen last).
            con.sql(f"""
            COPY (
              SELECT (d.s).fid AS file_id, (d.s).frn AS file_row_number
              FROM (
                SELECT arg_min(
                         struct_pack(fid := file_id, frn := file_row_number),
                         coalesce(TRY_CAST(review_date AS DATE), DATE '9999-12-31')
                       ) AS s
                FROM rev_keys
                WHERE hash(review_url) % {nb} = {b}
                GROUP BY review_url
              ) d
            ) TO '{part}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)""")
            out(f"   bucket {b+1}/{nb}: ok")
        # consolidate the buckets into a single _stage_keep.parquet
        con.sql(f"COPY (SELECT * FROM read_parquet('{keep_dir}/*.parquet')) TO '{keep}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)")
        n_dedup = con.sql(f"SELECT count(*) FROM read_parquet('{keep}')").fetchone()[0]
        out(f"[1a-i] global dedup: {n_dedup:,}  (removed: {n_raw-n_dedup:,})")

    # lookups loaded ONCE (avoids re-reading users/games for each file)
    con.sql(f"CREATE OR REPLACE TEMP TABLE usr_lkp AS SELECT steam_id, country FROM read_parquet({gusers}, union_by_name=true)")
    con.sql(f"CREATE OR REPLACE TEMP TABLE gm_lkp  AS SELECT game_id, popular_tags, has_review_bombing FROM read_parquet({ggames}, union_by_name=true)")

    # reused expressions
    sid = "nullif(regexp_extract(r.user_url,'profiles/([0-9]+)',1),'')"
    gk  = f"coalesce({sid}, 'v:'||nullif(regexp_extract(r.user_url,'id/([^/]+)',1),''))"

    # ---- 1b: process ONE FILE at a time, write reviews_resolved/rr_NNN.parquet (RESUMABLE) ----
    rr_dir = os.path.join(out_dir, "reviews_resolved")
    os.makedirs(rr_dir, exist_ok=True)
    ntot = len(detox_items)
    for i, (p, lang) in enumerate(detox_items):
        out_i = os.path.join(rr_dir, f"rr_{i:03d}.parquet")
        if os.path.exists(out_i):
            out(f"  [{i+1}/{ntot}] already exists, skipping {os.path.basename(out_i)}")
            continue
        p_esc = p.replace(chr(39), chr(39)*2)
        con.sql(rf"""
        COPY (
          SELECT
            r.review_url AS review_id,
            {sid}  AS user_key,
            {gk}   AS group_key,
            CASE WHEN lower(r.user_url) LIKE '%/profiles/%' THEN 'profiles'
                 WHEN lower(r.user_url) LIKE '%/id/%'       THEN 'vanity' ELSE 'outro' END AS url_type,
            '{lang}' AS review_lang,
            CASE WHEN r.toxicity>={THR_DETOX} OR r.perspective_score>={THR_PERSP} THEN 1 ELSE 0 END AS y_review,
            CASE WHEN r.toxicity>={THR_DETOX} AND r.perspective_score>={THR_PERSP} THEN 'both'
                 WHEN r.toxicity>={THR_DETOX} THEN 'detox'
                 WHEN r.perspective_score>={THR_PERSP} THEN 'persp' ELSE 'none' END AS tox_source,
            {gold_sel}
            r.toxicity, r.perspective_score, r.severe_toxicity, r.obscene,
            r.identity_attack, r.insult, r.threat, r.sexual_explicit,
            r.review_text,
            length(r.review_text) AS len_chars,
            CASE WHEN r.review_text IS NULL OR trim(r.review_text)='' THEN 0
                 ELSE len(string_split_regex(trim(r.review_text), '\s+')) END AS len_words,
            CASE WHEN length(regexp_replace(coalesce(r.review_text,''),'[^A-Za-z]','','g'))=0 THEN 0.0
                 ELSE length(regexp_replace(coalesce(r.review_text,''),'[^A-Z]','','g'))::DOUBLE
                      / length(regexp_replace(coalesce(r.review_text,''),'[^A-Za-z]','','g')) END AS caps_ratio,
            len(regexp_extract_all(coalesce(r.review_text,''), '\b[A-Z]{{3,}}\b')) AS upper_word_count,
            len(regexp_extract_all(lower(coalesce(r.review_text,'')), 'r{{3,}}'))  AS rrun_count,
            len(regexp_extract_all(coalesce(r.review_text,''), '!'))               AS excl_count,
            len(regexp_extract_all(coalesce(r.review_text,''), '\?'))              AS ques_count,
            r.is_recommended, r.hours_played,
            TRY_CAST(r.review_date AS DATE)        AS review_dt,
            year(TRY_CAST(r.review_date AS DATE))  AS review_year,
            TRY_CAST(r.game_id AS BIGINT)          AS game_id_int,
            (u.steam_id IS NOT NULL)               AS joinable,
            coalesce(list_contains(g.popular_tags,'PvP') OR list_contains(g.popular_tags,'Team-Based')
                     OR list_contains(g.popular_tags,'Online Co-Op'), false) AS g_competitive,
            coalesce(list_contains(g.popular_tags,'Gore') OR list_contains(g.popular_tags,'Violent')
                     OR list_contains(g.popular_tags,'Shooter') OR list_contains(g.popular_tags,'FPS'), false) AS g_violent,
            coalesce(list_contains(g.popular_tags,'Comedy'), false) AS g_comedy,
            coalesce(g.has_review_bombing, 0)                       AS g_review_bombing,
            CASE WHEN hash({gk}) % 100 < 70 THEN 'train'
                 WHEN hash({gk}) % 100 < 85 THEN 'val' ELSE 'test' END AS split
          FROM read_parquet('{p_esc}', file_row_number=true) r
          JOIN (SELECT file_row_number FROM read_parquet('{keep}') WHERE file_id={i}) k USING (file_row_number)
          LEFT JOIN usr_lkp u ON {sid} = u.steam_id
          LEFT JOIN gm_lkp  g ON TRY_CAST(r.game_id AS BIGINT) = g.game_id
          {gold_join}
        ) TO '{out_i}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)
        """)
        out(f"  [{i+1}/{ntot}] wrote {os.path.basename(out_i)}")
        if sleep_between > 0:
            time.sleep(sleep_between)        # let the machine breathe/cool down between files

    out(f"-> wrote: {rr_dir}  ({ntot} rr_NNN.parquet files)")
    if not keep_stage:
        if os.path.exists(keep): os.remove(keep)
        if os.path.isdir(keep_dir): shutil.rmtree(keep_dir, ignore_errors=True)
        out("   [_stage_keep* removed; use --keep-stage to keep the dedup keys]")
    return rr_dir, n_dedup

# ----------------------------------------------------------------------------- 
def build_user_features(con, rr_dir, users, out_dir, join_buckets=8, lang_sql=None):
    import shutil
    lang_and = (" AND " + lang_sql) if lang_sql else ""
    h1("STAGE 2  user_features  (TASK B -- behavioral; no score; no text)")
    rr = f"read_parquet('{rr_dir}/*.parquet', union_by_name=true)"
    gusers = lst(users)

    # imputation medians materialized ONCE (1 row) so users is not re-read per bucket
    con.sql(f"""CREATE OR REPLACE TEMP TABLE med AS
                SELECT approx_quantile(profile_level, 0.5) AS pl_med,
                       approx_quantile(library_size, 0.5) AS ls_med
                FROM read_parquet({gusers}, union_by_name=true)""")

    # ---- assembly in BUCKETS by hash(user_key) (bounded memory; resumable per bucket) ----
    # All of a user's reviews share the same user_key, hence the same bucket, so the per-user
    # aggregation is complete within a single bucket. The join to `users` is also filtered by
    # the same hash, staying bounded to ~1/N. The consolidated result is identical to that of a
    # single query (the split uses hash%100, independent of the number of buckets).
    path = os.path.join(out_dir, "user_features.parquet")
    parts_dir = os.path.join(out_dir, "_uf_parts")
    os.makedirs(parts_dir, exist_ok=True)
    nb = max(1, int(join_buckets))
    out(f"assembling user_features in {nb} bucket(s) by hash(user_key)  (agg + join, resumable)")
    for b in range(nb):
        part = os.path.join(parts_dir, f"uf_{b:03d}.parquet")
        if os.path.exists(part):
            out(f"   bucket {b+1}/{nb}: already exists, skipping"); continue
        con.sql(f"""
        COPY (
          WITH agg AS (
            SELECT user_key,
                   max(y_review)                                              AS y_user,
                   sum(y_review)                                              AS n_toxic,
                   count(*)                                                   AS n_reviews,
                   approx_count_distinct(game_id_int)                         AS n_distinct_games,
                   avg(CASE WHEN is_recommended THEN 1.0 WHEN NOT is_recommended THEN 0.0 END) AS pct_recommended,
                   approx_quantile(hours_played, 0.5)                         AS hours_med,
                   approx_quantile(hours_played, 0.9)                         AS hours_p90,
                   sum(hours_played)                                          AS hours_total,
                   date_diff('day', min(review_dt), max(review_dt))           AS tenure_days,
                   date_diff('day', max(review_dt), current_date)             AS days_since_last_review,
                   avg(CASE WHEN g_competitive THEN 1.0 ELSE 0.0 END)         AS pct_competitive,
                   avg(CASE WHEN g_violent     THEN 1.0 ELSE 0.0 END)         AS pct_violent,
                   avg(CASE WHEN g_comedy      THEN 1.0 ELSE 0.0 END)         AS pct_comedy
            FROM {rr}
            WHERE user_key IS NOT NULL AND joinable AND hash(user_key) % {nb} = {b}{lang_and}
            GROUP BY user_key
          )
          SELECT
            a.user_key, a.y_user,
            a.n_toxic, a.n_reviews AS n_reviews_diag,                      -- DIAGNOSTIC: drop from X
            a.n_reviews, a.n_distinct_games,
            coalesce(a.pct_recommended, 0.0)                               AS pct_recommended,
            a.hours_med, a.hours_p90, a.hours_total,
            a.tenure_days, a.days_since_last_review,
            coalesce(a.n_reviews::DOUBLE / nullif(a.tenure_days,0) * 365, 0) AS reviews_per_year,
            a.pct_competitive, a.pct_violent, a.pct_comedy,
            coalesce(us.profile_level, m.pl_med)        AS profile_level,
            (us.profile_level IS NULL)::INT             AS profile_level_missing,
            coalesce(us.library_size, m.ls_med)         AS library_size,
            (us.library_size IS NULL)::INT              AS library_size_missing,
            coalesce(us.awards, 0)         AS awards,
            coalesce(us.insignias, 0)      AS insignias,
            coalesce(us.screenshots, 0)    AS screenshots,
            coalesce(us.workshop_items, 0) AS workshop_items,
            coalesce(us.guides, 0)         AS guides,
            coalesce(us.arts, 0)           AS arts,
            coalesce(us.groups, 0)         AS groups,
            CASE WHEN us.reviews_langs IS NULL THEN 0 ELSE len(string_split(us.reviews_langs, ',')) END AS n_review_langs,
            ( (us.country IS NOT NULL)::INT + (us.profile_level IS NOT NULL)::INT
            + (us.library_size IS NOT NULL)::INT + (us.screenshots IS NOT NULL)::INT
            + (us.groups IS NOT NULL)::INT + (us.insignias IS NOT NULL)::INT )::DOUBLE / 6 AS profile_visibility,
            us.country,
            (us.country='Brazil')::INT                  AS country_is_brazil,
            us.has_ban::INT                             AS has_ban,
            coalesce(us.days_since_last_ban, 9999)      AS ban_recency_days,
            CASE WHEN hash(a.user_key) % 100 < 70 THEN 'train'
                 WHEN hash(a.user_key) % 100 < 85 THEN 'val' ELSE 'test' END AS split
          FROM agg a
          LEFT JOIN (SELECT steam_id, profile_level, library_size, awards, insignias,
                            screenshots, workshop_items, guides, arts, groups,
                            reviews_langs, country, has_ban, days_since_last_ban
                     FROM read_parquet({gusers}, union_by_name=true)
                     WHERE hash(steam_id) % {nb} = {b}) us
            ON a.user_key = us.steam_id
          CROSS JOIN med m
        ) TO '{part}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE)""")
        out(f"   bucket {b+1}/{nb}: ok")

    # consolidate the buckets into a single user_features.parquet (lightweight streaming copy)
    con.sql(f"COPY (SELECT * FROM read_parquet('{parts_dir}/*.parquet')) TO '{path}' (FORMAT PARQUET)")
    n_u = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    out(f"users (joinable, with >=1 review in the study scope{(' [' + lang_sql + ']') if lang_sql else ''}): {n_u:,}")
    shutil.rmtree(parts_dir, ignore_errors=True)
    out(f"-> wrote: {path}")
    return path, n_u

# ----------------------------------------------------------------------------- 
def build_train_sample(con, rr_dir, out_dir, neg_ratio, seed=42, lang_sql=None):
    h1("STAGE 3  review_train_sample  (balanced sample of split=train)")
    rr = f"read_parquet('{rr_dir}/*.parquet', union_by_name=true)"
    lang_and = (" AND " + lang_sql) if lang_sql else ""
    npos = con.sql(f"SELECT count(*) FROM {rr} WHERE split='train' AND y_review=1{lang_and}").fetchone()[0]
    nneg_keep = int(npos * neg_ratio)
    out(f"positives in train: {npos:,}   | sampled negatives ({neg_ratio}x): {nneg_keep:,}")
    if npos == 0:
        out("no positives in train -- skipping sample."); return None
    path = os.path.join(out_dir, "review_train_sample.parquet")
    con.sql(f"""
      COPY (
        SELECT * FROM {rr} WHERE split='train' AND y_review=1{lang_and}
        UNION ALL
        SELECT * FROM (
          SELECT * FROM {rr} WHERE split='train' AND y_review=0{lang_and}
        ) USING SAMPLE {nneg_keep} ROWS (reservoir, {seed})
      ) TO '{path}' (FORMAT PARQUET)
    """)
    tot = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    out(f"-> wrote: {path}   ({tot:,} rows; prevalence ~{100*npos/tot:.1f}%)")
    out("   (reminder: evaluate on split=test with the REAL prevalence, not on this sample)")
    return path

# ----------------------------------------------------------------------------- 
def sanity_and_findings(con, rr_dir, uf_path, out_dir):
    h1("STAGE 4  Sanity checks, findings, and feature dictionary")
    rr = f"read_parquet('{rr_dir}/*.parquet', union_by_name=true)"

    h2("Prevalence of y_review by split  (should be similar across splits)")
    d = con.sql(f"""SELECT split, count(*) n, sum(y_review) toxic_n,
                    round(100.0*sum(y_review)/count(*),3) pct_toxic
                    FROM {rr} GROUP BY 1 ORDER BY 1""").df()
    show(d)

    h2("Reviews by language x toxicity")
    d = con.sql(f"""SELECT review_lang, count(*) n, sum(y_review) toxic_n,
                    round(100.0*sum(y_review)/count(*),3) pct_toxic
                    FROM {rr} GROUP BY 1 ORDER BY n DESC""").df()
    show(d)

    h2("FINDING: % of toxic reviews that RECOMMEND, by language  (paper's PT ~70.3%)")
    d = con.sql(f"""SELECT review_lang,
                    round(100.0*avg(CASE WHEN is_recommended THEN 1.0 WHEN NOT is_recommended THEN 0.0 END),2) AS pct_recommend,
                    sum(y_review) AS n_toxic
                    FROM {rr} WHERE y_review=1 GROUP BY 1 ORDER BY n_toxic DESC""").df()
    show(d); save_csv(d, out_dir, "recommend_finding_by_language.csv")

    h2("Join coverage (reviews)  --  English-only scope")
    d = con.sql(f"""SELECT
        count(*) AS reviews,
        count(*) FILTER (WHERE joinable) AS joinable_n,
        round(100.0*count(*) FILTER (WHERE joinable)/count(*),1) AS pct_joinable
        FROM {rr}""").df()
    show(d)

    if uf_path and os.path.exists(uf_path):
        h2("user_features: prevalence of y_user  (English-only scope)")
        d = con.sql(f"""SELECT count(*) users,
                        sum(y_user) toxic_n, round(100.0*avg(y_user),2) pct_toxic
                        FROM read_parquet('{uf_path}')""").df()
        show(d)
        h2("user_features: prevalence by split")
        d = con.sql(f"""SELECT split, count(*) n, round(100.0*avg(y_user),2) pct_toxic
                        FROM read_parquet('{uf_path}') GROUP BY 1 ORDER BY 1""").df()
        show(d)

    # ---- feature dictionary + lists for the modeler ----
    write_feature_dictionary(con, rr_dir, uf_path, out_dir)

def save_csv(df, out_dir, name):
    if df is not None and len(df):
        df.to_csv(os.path.join(out_dir, name), index=False)

def write_feature_dictionary(con, rr_dir, uf_path, out_dir):
    h2("Feature dictionary + lists (feature_cols.py)")
    # roles per column
    review_roles = {
        "review_id":"key","user_key":"key","group_key":"key","url_type":"meta","review_lang":"meta",
        "joinable":"meta","split":"meta","review_dt":"meta","game_id_int":"meta",
        "y_review":"label","tox_source":"analysis","y_review_gold":"label",
        "toxicity":"FORBIDDEN","perspective_score":"FORBIDDEN","severe_toxicity":"FORBIDDEN",
        "obscene":"FORBIDDEN","identity_attack":"FORBIDDEN","insult":"FORBIDDEN","threat":"FORBIDDEN",
        "sexual_explicit":"FORBIDDEN",
        "review_text":"feature_text",
        "len_chars":"feature","len_words":"feature","caps_ratio":"feature","upper_word_count":"feature",
        "rrun_count":"feature","excl_count":"feature","ques_count":"feature",
        "is_recommended":"gray","hours_played":"gray","review_year":"gray",
        "g_competitive":"gray","g_violent":"gray","g_comedy":"gray","g_review_bombing":"gray",
    }
    # English-only study: the language/Brazil columns (pct_pt, pct_en, has_*_reviews,
    # is_br_pt) were REMOVED from the materialized tables because they are degenerate (a single language).
    user_roles_override = {
        "user_key":"key","split":"meta","country_is_brazil":"gray","country":"gray",
        "y_user":"label","n_toxic":"DIAGNOSTIC(drop)","n_reviews_diag":"DIAGNOSTIC(drop)",
        "has_ban":"gray","ban_recency_days":"gray",
    }
    def schema_of(src):
        return con.sql(f"DESCRIBE SELECT * FROM {src}").df()[["column_name","column_type"]]

    rr = f"read_parquet('{rr_dir}/*.parquet', union_by_name=true)"
    rows = []
    for _, r in schema_of(rr).iterrows():
        c = r["column_name"]
        rows.append({"table":"reviews_resolved","column":c,"type":r["column_type"],
                     "role":review_roles.get(c,"feature")})
    if uf_path and os.path.exists(uf_path):
        for _, r in schema_of(f"read_parquet('{uf_path}')").iterrows():
            c = r["column_name"]
            rows.append({"table":"user_features","column":c,"type":r["column_type"],
                         "role":user_roles_override.get(c,"feature")})
    fd = pd.DataFrame(rows)
    save_csv(fd, out_dir, "feature_dictionary.csv")
    out(f"feature_dictionary.csv: {len(fd)} columns cataloged")

    # explicit lists for direct use in Python
    rev_feats = fd[(fd.table=="reviews_resolved") & (fd.role.isin(["feature","gray"]))]["column"].tolist()
    rev_forb  = fd[(fd.table=="reviews_resolved") & (fd.role.str.contains("FORBIDDEN"))]["column"].tolist()
    usr_feats = fd[(fd.table=="user_features") & (fd.role.isin(["feature","gray"]))]["column"].tolist()
    usr_drop  = fd[(fd.table=="user_features") & (fd.role.str.contains("DIAGNOSTIC"))]["column"].tolist()
    py = os.path.join(out_dir, "feature_cols.py")
    with open(py, "w", encoding="utf-8") as f:
        f.write("# Generated by build_features.py -- select X with these lists.\n")
        f.write("# TASK A (review-level): text + surface/context. NEVER include REVIEW_FORBIDDEN.\n")
        f.write(f"REVIEW_TEXT_COL = 'review_text'\n")
        f.write(f"REVIEW_SURFACE_CONTEXT = {rev_feats!r}\n")
        f.write(f"REVIEW_FORBIDDEN = {rev_forb!r}\n\n")
        f.write("# TASK B (user-level): ONLY behavioral. Drop USER_DROP_BEFORE_FIT from X.\n")
        f.write(f"USER_FEATURES = {usr_feats!r}\n")
        f.write(f"USER_DROP_BEFORE_FIT = {usr_drop!r}\n")
    out(f"feature_cols.py: REVIEW_SURFACE_CONTEXT ({len(rev_feats)}), USER_FEATURES ({len(usr_feats)})")
    out(f"  forbidden (review): {rev_forb}")
    out(f"  diagnostic to drop (user): {usr_drop}")

# ----------------------------------------------------------------------------- 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--neg-ratio", type=int, default=4, help="negatives per positive in the training sample")
    ap.add_argument("--memory-limit", default="6GB", help="DuckDB RAM limit (e.g. 6GB)")
    ap.add_argument("--threads", type=int, default=2, help="threads (fewer = less heat/power/memory)")
    ap.add_argument("--temp-dir", default=None, help="DuckDB spill dir (disk)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds of pause between files (lets the machine cool down)")
    ap.add_argument("--gold-csv", default=None, help="CSV with review_url,y_review_gold (human annotation)")
    ap.add_argument("--keep-stage", action="store_true", help="keep _stage_keep.parquet (dedup keys)")
    ap.add_argument("--dedup-buckets", type=int, default=16,
                    help="hash buckets for the global dedup (more buckets = less RAM per pass, more re-reads)")
    ap.add_argument("--user-buckets", type=int, default=8,
                    help="hash buckets for assembling user_features (more buckets = less RAM per pass)")
    ap.add_argument("--langs", default="en",
                    help="languages in the study scope (csv). DEFAULT 'en' = English-only study; "
                         "use 'en,pt' to reproduce the old bilingual corpus.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)
    con = duckdb.connect()
    # --- scale/stability tuning (real data ~43M; reduces peak RAM and load) ---
    con.sql("SET preserve_insertion_order=false")          # does not buffer order, so it spills better
    con.sql(f"SET temp_directory='{args.temp_dir or os.path.join(args.out,'_tmp')}'")
    try: con.sql("SET max_temp_directory_size='500GB'")    # allows a large spill (uses only what is needed)
    except Exception: pass
    if args.threads: con.sql(f"PRAGMA threads={args.threads}")
    if args.memory_limit: con.sql(f"PRAGMA memory_limit='{args.memory_limit}'")
    try: con.sql("PRAGMA enable_progress_bar")
    except Exception: pass

    h1("build_features.py  --  Final AI Project")
    out(f"root: {root}   | threads={args.threads}  memory_limit={args.memory_limit}  sleep={args.sleep}s")
    users, games, detox = discover(root)
    # --- STUDY SCOPE: filter the languages (English-only by default) ---------------
    # `detox` already comes with each file marked 'en'/'pt'. We keep only the requested languages.
    # Since the 'en' files precede the 'pt' ones in discover()'s sorted order, filtering 'pt'
    # does not shift the 'en' indices, so any already-existing rr_NNN partitions remain valid.
    langs = [s.strip().lower() for s in args.langs.split(",") if s.strip()]
    detox = {p: lg for p, lg in detox.items() if lg in langs}
    lang_sql = "review_lang IN (" + ",".join("'" + l.replace("'", "''") + "'" for l in langs) + ")"
    out(f"files: users={len(users)}  games={len(games)}  data/corpus/reviews_w_detoxify={len(detox)}  (languages={langs})")
    out(f"study scope: {'/'.join(langs).upper()}" + ("  [English-only]" if langs == ['en'] else ""))
    if not detox or not users or not games:
        out("Missing essential files (users/games/data/corpus/reviews_w_detoxify). Check --root/--langs.")
        _flush(args.out); return

    rr_dir, _ = build_reviews_resolved(con, detox, users, games, args.out, args.gold_csv,
                                       args.keep_stage, args.sleep, args.dedup_buckets)
    uf_path, _ = build_user_features(con, rr_dir, users, args.out, args.user_buckets, lang_sql=lang_sql)
    build_train_sample(con, rr_dir, args.out, args.neg_ratio, lang_sql=lang_sql)
    sanity_and_findings(con, rr_dir, uf_path, args.out)

    h1("DONE")
    out("Artifacts in: " + os.path.abspath(args.out))
    out(" - reviews_resolved/        (Task A, partitioned by split/lang)")
    out(" - review_train_sample.parquet  (balanced sample of train)")
    out(" - user_features.parquet    (Task B)")
    out(" - feature_dictionary.csv, feature_cols.py, build_report.txt")
    _flush(args.out)

def _flush(out_dir):
    with open(os.path.join(out_dir, "build_report.txt"), "w", encoding="utf-8") as f:
        f.write(_BUF.getvalue())

if __name__ == "__main__":
    main()
