"""Shared matplotlib styling for the paper's characterization figures.

THE ONE RULE: A FIGURE IS AUTHORED AT THE SIZE IT WILL BE PRINTED.

A figure drawn on a 16-inch canvas and then pulled into a 3.31-inch column
by LaTeX is scaled by 0.21, and its 20pt labels reach the page at 4.2pt.
That is how the first submission's Figures 1 and 2 arrived at 4.17pt and
4.99pt, under AAAI's 7pt floor, and it is what the editors rejected. The
numbers in the figure were right; only the geometry was wrong.

So every canvas here is declared in the units LaTeX will use:

  AAAI two-column, from aaai24.sty:
    \\textwidth    = 7.0in         (\\columnsep = 0.375in)
    \\columnwidth  = 3.3125in      = (7.0 - 0.375) / 2

A figure sized COLUMN_WIDTH and included with `width=\\columnwidth` is
scaled by exactly 1.0, so a 9pt label is 9pt on the page. Nothing has to be
back-computed, and the guarantee survives anyone re-running these scripts.

TWO THINGS THAT QUIETLY BREAK THAT GUARANTEE, both avoided below:

1. `bbox_inches='tight'` trims the canvas to its content, so the saved PDF
   is NARROWER than the figsize that was asked for. LaTeX then scales it
   back UP to \\columnwidth and the fonts land somewhere unintended. Every
   figure here is therefore saved WITHOUT a tight bbox, and uses
   constrained layout so the content fits the declared canvas instead of
   the canvas shrinking to the content. `save_figure` enforces this and
   reports the delivered size.

2. Including a figure at a width other than the one it was authored for.
   The sizes below are the contract; the README records which
   `\\includegraphics` width each one expects.

Design (colors, structure, labels) is unchanged from the published
figures - it was read out of their PDF content streams and is reproduced
exactly. Only the geometry and the type sizes move, because only those
were out of spec:

  green  #2ca02c  (matplotlib tab:green)  non-toxic
  red    #d62728  (matplotlib tab:red)    toxic
  font   Times New Roman, embedded as TrueType (pdf.fonttype 42)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; these scripts run on servers
import matplotlib.pyplot as plt  # noqa: E402

# AAAI page geometry, in inches. See aaai24.sty lines 32-34.
TEXT_WIDTH = 7.0
COLUMN_SEP = 0.375
COLUMN_WIDTH = (TEXT_WIDTH - COLUMN_SEP) / 2  # 3.3125

# Toxic vs. non-toxic, everywhere in the paper.
NONTOXIC_COLOR = "#2ca02c"
TOXIC_COLOR = "#d62728"
RATE_LINE_COLOR = "black"

# COLOR ALONE DOES NOT SURVIVE A BLACK-AND-WHITE PRINTER, AND AAAI ASKS
# THAT IT DOES: "graph's bars/lines are distinguishable enough when printed
# in grayscale" is part of the same rule as the type sizes. These two
# colors are 112 and 91 out of 255 in luminance - 8% apart, where about 20%
# is the threshold for telling two greys apart - so printed in grayscale
# the toxic slice disappears into the non-toxic one and the two legend
# swatches become the same box.
#
# The fix is a second channel rather than a new palette: the colors stay
# exactly as published, and shape carries the distinction when hue cannot.
# Bars get a hatch, lines get a dash pattern. The median rules are dotted
# rather than dashed so they stay distinct from the non-toxic curve, which
# is the pair most easily confused once both are grey.
TOXIC_HATCH = "///"
NONTOXIC_LINESTYLE = (0, (3.5, 1.5))
TOXIC_LINESTYLE = "-"
MEDIAN_LINESTYLE = (0, (1, 1.5))

# Canvases, in the width LaTeX will give each figure.
#   BAR      one column,  included at width=\columnwidth
#   HEATMAP  both columns, included at width=\textwidth inside figure*
#   CDF      one panel of three, included at width=\linewidth inside a
#            0.30\linewidth subfigure of a figure* -> 0.30 * 7.0 = 2.1in
#   LEGEND   its own line above the three panels, included at
#            width=0.45\linewidth -> 0.45 * 7.0 = 3.15in
#
# HEIGHT IS A PAGE BUDGET, NOT A FREE PARAMETER. The body of the paper
# fills its 9-page allowance exactly, so every inch a figure grows has to
# come out of the text. Authoring at print size costs height - the original
# Figure 1 was a 2:1 canvas squeezed into 1.64in of column, which is
# precisely why its labels ended up at 4.17pt - so each canvas below is the
# shortest one that still clears the 7pt floor, not the most comfortable.
BAR_FIGSIZE = (COLUMN_WIDTH, 2.25)
HEATMAP_FIGSIZE = (TEXT_WIDTH, 2.45)
CDF_FIGSIZE = (0.30 * TEXT_WIDTH, 1.50)
LEGEND_FIGSIZE = (0.45 * TEXT_WIDTH, 0.24)

# Type sizes, in points, landing on the page at these exact values because
# every canvas above is included at scale 1.0. AAAI asks for 10pt where
# possible and forbids anything below 7pt (\scriptsize); the body size here
# is 9pt, with 8pt reserved for dense secondary text (legend entries, cell
# annotations, colorbar ticks) that would otherwise collide.
BODY_FONTSIZE = 9
SMALL_FONTSIZE = 8
MIN_ALLOWED_FONTSIZE = 7  # AAAI's floor; assert_compliant checks against it

# Figure 1 carries ten tag labels stacked vertically, and every point of
# type there is a point of column height. AAAI allows smaller-than-body
# text where necessary ("Smaller text can be used if necessary, but it
# should stay readable and not go below 7pt"), so it runs at 8pt: two
# points clear of the floor, and about 0.7in shorter than it would be at
# 9pt on a page that has no spare inches.
BAR_FONTSIZE = 8

# MATHTEXT SHRINKS SUPERSCRIPTS TO 0.7 OF THE BASE, AND THE FLOOR APPLIES
# TO THE SUPERSCRIPT TOO. A log axis labels its ticks "10^0, 10^1, ...",
# and the exponent is the smallest type in the whole figure: at a 9pt base
# it lands at 6.3pt, under the floor, in a figure whose every other label
# passes. The CDF panels are the only figures here with a log axis, so they
# run at 10pt, putting the exponent at exactly 7.0pt - and incidentally at
# the 10pt body size AAAI asks for. Plain integer ticks would avoid the
# shrink entirely, but "10000" at 10pt does not fit a 2.1in panel five
# times over.
MATHTEXT_SHRINK = 0.7
CDF_FONTSIZE = 10

# Line weights, unchanged from the published figures.
CDF_LINEWIDTH = 1.2
RATE_LINEWIDTH = 1.2
GRID_LINEWIDTH = 0.5

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,  # embed TrueType, so the PDF is not font-dependent
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}


def apply_style(fontsize: int = BODY_FONTSIZE) -> None:
    """Resets to the publication style at `fontsize`, in points as printed."""
    plt.rcdefaults()
    plt.rcParams.update(RC_PARAMS)
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })


def save_figure(fig, output_path: Path, expected_width: float = None) -> Path:
    """Saves at the declared canvas size and reports what was delivered.

    No `bbox_inches='tight'`: trimming would hand LaTeX a narrower PDF than
    the figure was designed for, and the scale-up would undo the type sizes
    this module exists to guarantee. See the module docstring.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    width, height = fig.get_size_inches()
    plt.close(fig)

    if expected_width is not None and abs(width - expected_width) > 1e-6:
        raise SystemExit(
            f"{output_path.name}: saved at {width:.4f}in but the include "
            f"expects {expected_width:.4f}in - type sizes would not survive."
        )
    return output_path


def assert_compliant(fontsizes, label: str, has_mathtext: bool = False) -> None:
    """Guards AAAI's 7pt floor at the point of authorship, so a future edit
    that shrinks a label fails here rather than in an editor's review.

    `has_mathtext=True` also checks the shrunk superscript size, which is
    what a log axis actually prints and is easy to miss - see
    MATHTEXT_SHRINK.
    """
    checked = dict.fromkeys(fontsizes)
    if has_mathtext:
        checked.update({round(f * MATHTEXT_SHRINK, 2): None for f in fontsizes})

    too_small = sorted(f for f in checked if f < MIN_ALLOWED_FONTSIZE)
    if too_small:
        raise SystemExit(
            f"{label}: {too_small} pt is below AAAI's {MIN_ALLOWED_FONTSIZE}pt floor"
            + (" (superscripts included)" if has_mathtext else "")
        )
