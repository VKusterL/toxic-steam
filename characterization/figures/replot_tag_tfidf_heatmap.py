"""Redraws `heatmap_tfidf_tags.pdf` from an existing `tag_tfidf_means.csv`.

WHY THIS EXISTS. run_tag_tfidf_heatmap.py has to stream the whole language
corpus to fit step03's vectorizer - about an hour on the full English
corpus - and none of that work depends on how the figure is DRAWN. The
matrix it produces is already exported as `tag_tfidf_means.csv` (see
`export_means`), and that CSV is the figure's entire input. So a change to
type sizes, canvas width or cell formatting can be re-rendered from it in
under a second, and the rendered PDF is still a product of the pipeline
rather than something drawn by hand next to it.

This script therefore recomputes NOTHING. If the corpus, the tag set or
the term set changes, run run_tag_tfidf_heatmap.py; this one only replaces
the picture drawn from a matrix that is already on disk.

Usage:
    python replot_tag_tfidf_heatmap.py \
        --means ../../results/characterization/tag_tfidf_means.csv \
        --output ../../AIIDE26___Toxic_Steam__A_Large_Scale_Characterization_and_Prediction_of_Toxic_Users_in_a_Gaming_Platform/main/images/heatmap_tfidf_tags.pdf

The row order in the CSV is the published order (strongest term first),
and `pcolormesh` puts row 0 at the bottom, so the file is read and drawn
as-is: no re-sorting here, or the figure would flip relative to the
version the paper describes.
"""
import argparse
from pathlib import Path

import pandas as pd

import plotting
import tag_tfidf_heatmap as hm
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Redraw the TF-IDF tag heatmap from a saved means matrix."
    )
    parser.add_argument(
        "--means", required=True, type=Path,
        help="tag_tfidf_means.csv, as written by run_tag_tfidf_heatmap.py",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Where to write heatmap_tfidf_tags.pdf",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.means.exists():
        raise SystemExit(f"No such means matrix: {args.means}")

    means = pd.read_csv(args.means, index_col="termo")
    info(f"Read {means.shape[0]} term(s) x {means.shape[1]} tag(s) from {args.means}")

    saved = hm.plot_heatmap(means, args.output)

    # The type sizes only survive if LaTeX includes the PDF at scale 1.0,
    # which it does exactly when the canvas is the width the include asks
    # for. Reporting it here means a mismatch shows up on this run rather
    # than in an editor's review.
    expected_pt = plotting.HEATMAP_FIGSIZE[0] * 72
    info(
        f"Canvas: {plotting.HEATMAP_FIGSIZE[0]:.4f}in ({expected_pt:.1f}pt) wide - "
        f"include with width=\\columnwidth inside a single-column `figure`, "
        f"where \\columnwidth is {plotting.COLUMN_WIDTH:.4f}in."
    )
    info(f"Smallest type in this figure: {plotting.HEATMAP_FONTSIZE}pt "
         f"(AAAI floor: {plotting.MIN_ALLOWED_FONTSIZE}pt)")
    return saved


if __name__ == "__main__":
    main()
