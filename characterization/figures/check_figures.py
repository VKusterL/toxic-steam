"""Checks the generated figure PDFs against AAAI's layout rules.

The rules the editors enforce are properties of the PRINTED page, not of
the plotting code, so this reads them back out of the PDFs themselves:
canvas size, every font size actually used, and what each label will
measure once LaTeX has applied the include width.

    python characterization/figures/check_figures.py --dir data/figures-output

A figure passes when it is authored at the width its `\\includegraphics`
uses - so LaTeX scales it by 1.0 - and no text in it falls below 7pt.

This catches what `plotting.assert_compliant` cannot. That guard checks the
sizes the code *declares*; this one checks the sizes that reached the file,
including the ones nobody declared - mathtext superscripts, which matplotlib
silently sets to 0.7 of the base, and are the smallest type in any figure
with a log axis.
"""
import argparse
import re
import zlib
from collections import Counter
from pathlib import Path

TEXT_WIDTH = 7.0
COLUMN_WIDTH = (TEXT_WIDTH - 0.375) / 2
MIN_PT = 7.0

# What each figure is authored for, and the include that has to match it.
EXPECTED = {
    "top-10-tags-tox.pdf":      (COLUMN_WIDTH,        r"width=\columnwidth (figure)"),
    "heatmap_tfidf_tags.pdf":   (TEXT_WIDTH,          r"width=\textwidth (figure*)"),
    "cdf-reviews-per-user.pdf": (0.30 * TEXT_WIDTH,   r"width=\linewidth in a 0.30\linewidth subfigure"),
    "cdf-library-size.pdf":     (0.30 * TEXT_WIDTH,   r"width=\linewidth in a 0.30\linewidth subfigure"),
    "cdf-steam-level.pdf":      (0.30 * TEXT_WIDTH,   r"width=\linewidth in a 0.30\linewidth subfigure"),
    "cdf-legend.pdf":           (0.45 * TEXT_WIDTH,   r"width=0.45\linewidth"),
}


def probe(path: Path) -> tuple:
    data = path.read_bytes()
    box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", data)
    nums = [float(x) for x in box.group(1).split()]
    width = (nums[2] - nums[0]) / 72

    sizes = Counter()
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            chunk = zlib.decompress(m.group(1))
        except zlib.error:
            chunk = m.group(1)
        for tf in re.finditer(r"/F\d+\s+([\d.]+)\s+Tf", chunk.decode("latin-1", "ignore")):
            sizes[round(float(tf.group(1)), 2)] += 1
    return width, sizes


def main():
    parser = argparse.ArgumentParser(description="Check figure PDFs against AAAI's layout rules.")
    parser.add_argument("--dir", type=Path, required=True, help="Directory holding the figure PDFs")
    args = parser.parse_args()

    print(f"{'figure':28s} {'width in':>9s} {'expected':>9s} {'scale':>6s} {'min pt':>7s}  status")
    print("-" * 76)
    ok = True
    for name, (expected_width, include) in EXPECTED.items():
        path = args.dir / name
        if not path.exists():
            print(f"{name:28s} {'-':>9} {expected_width:9.4f} {'-':>6} {'-':>7}  MISSING")
            ok = False
            continue

        width, sizes = probe(path)
        scale = expected_width / width
        smallest = min(sizes) if sizes else float("inf")
        printed = smallest * scale

        problems = []
        if abs(scale - 1.0) > 1e-4:
            problems.append(f"scaled {scale:.3f}x by the include")
        if printed < MIN_PT - 1e-6:
            problems.append(f"{printed:.2f}pt on the page")
        ok &= not problems
        print(
            f"{name:28s} {width:9.4f} {expected_width:9.4f} {scale:6.3f} "
            f"{printed:7.2f}  {'OK' if not problems else '; '.join(problems)}"
        )
        if not problems:
            print(f"{'':28s} include as: {include}")

    print("-" * 76)
    print("ALL FIGURES COMPLY" if ok else "SOME FIGURES DO NOT COMPLY")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
