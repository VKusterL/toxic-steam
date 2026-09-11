"""Shared matplotlib styling for the paper's characterization figures.

EVERY CONSTANT HERE WAS READ BACK OUT OF THE PUBLISHED PDFs, not chosen.
The figures in `main/images/` were produced by matplotlib 3.10.7, so their
content streams still carry the exact colors, font, font sizes, line
widths, dash patterns and canvas sizes that produced them. Those values
were extracted and are reproduced below so a regenerated figure is
indistinguishable in design from the one already in the paper.

Do not "improve" these. A regenerated figure that differs from its
neighbours in the same paper - a slightly different red, a thinner line, a
sans-serif tick label - reads as an error to a reviewer even when the
numbers behind it are right.

Extracted from the PDFs:
  green  #2ca02c  (matplotlib tab:green)  non-toxic
  red    #d62728  (matplotlib tab:red)    toxic
  font   TimesNewRomanPSMT, embedded as TrueType (pdf.fonttype 42)
  sizes  bar chart 20pt text / 18pt legend
         CDF panels 25pt text, legend file 23pt
         heatmap 10pt cells+ticks / 9pt axis label / 8pt colorbar
  lines  CDF curves and median rules 1.5pt; bar-chart rate line 2.0pt
  dashes bar-chart grid '--' at 0.5pt  -> [1.85 0.8]
         CDF median rules '--' at 1.5pt -> [5.55 2.4]
  canvas bar chart 16x8in, CDF panel 8x6in, heatmap 7x5in, legend 8x1in
         (the PDFs are slightly smaller because of bbox_inches='tight')
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; these scripts run on servers
import matplotlib.pyplot as plt  # noqa: E402

# Toxic vs. non-toxic, everywhere in the paper.
NONTOXIC_COLOR = "#2ca02c"
TOXIC_COLOR = "#d62728"
RATE_LINE_COLOR = "black"

# Canvas sizes, in inches, before bbox_inches='tight' trims the margins.
BAR_FIGSIZE = (16, 8)
CDF_FIGSIZE = (8, 6)
HEATMAP_FIGSIZE = (7, 5)
LEGEND_FIGSIZE = (8, 1)

# Font sizes, per figure family.
BAR_FONTSIZE = 20
BAR_LEGEND_FONTSIZE = 18
CDF_FONTSIZE = 25
LEGEND_FONTSIZE = 23
HEATMAP_FONTSIZE = 10
HEATMAP_LABEL_FONTSIZE = 9
HEATMAP_CBAR_FONTSIZE = 8

# Line weights.
CDF_LINEWIDTH = 1.5
RATE_LINEWIDTH = 2.0
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
}


def apply_style(fontsize: int) -> None:
    """Resets to the published style at `fontsize`. Called per figure, since
    the three figure families use different sizes."""
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


def save_figure(fig, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
