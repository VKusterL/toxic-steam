#!/usr/bin/env python3
"""Audit this paper's LaTeX source and compiled PDF without modifying either.

Usage (from the repository root; requires the existing PyMuPDF dependency)::

    python tools/check_paper_layout.py --tex paper/main/main.tex \
        --pdf paper/main.pdf --json paper/layout-audit.json

Optional --render-dir writes one PNG preview per page, for visual review.
Exit status is 1 for a failed reliable check and 2 for an input/dependency error.

Scope and limits:
* Figure geometry is checked against the six characterization assets' actual
  canvas widths and explicit include/subfigure widths. Text in their embedded
  font families is also measured in the final PDF, after LaTeX transformations.
  Shared fonts are pooled: this does not assign every span to an individual
  figure. Converted-to-outline labels cannot be checked as text.
* Caption identification uses a PDF text block beginning with ``Table n.`` (or
  a colon, which is detected and rejected). Source order checks caption position;
  the PDF body-region measurement uses nearby booktabs rules and is heuristic.
  Ambiguous/missing table regions are warnings, not fabricated measurements.
* This is not a complete publisher validator. Readability, grayscale contrast,
  page composition, float proximity, and semantic correctness need visual review.
* PDF distances are big points (1/72 inch). LaTeX text sizes use TeX points
  (1/72.27 inch); caption/table measurements account for that difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

try:
    import pymupdf
except ImportError:
    raise SystemExit("PyMuPDF is required. Use the Python environment containing pymupdf.")


TEX_TO_PDF_PT = 72 / 72.27
MIN_FIGURE_PT = 7.0
MIN_TABLE_PT = 7 * TEX_TO_PDF_PT
CAPTION_PT = 10 * TEX_TO_PDF_PT
TEXT_WIDTH_IN = 7.0
COLUMN_WIDTH_IN = 3.3125
EXPECTED = {
    "top-10-tags-tox.pdf": COLUMN_WIDTH_IN,
    "heatmap_tfidf_tags.pdf": TEXT_WIDTH_IN,
    "cdf-reviews-per-user.pdf": 0.30 * TEXT_WIDTH_IN,
    "cdf-library-size.pdf": 0.30 * TEXT_WIDTH_IN,
    "cdf-steam-level.pdf": 0.30 * TEXT_WIDTH_IN,
    "cdf-legend.pdf": 0.45 * TEXT_WIDTH_IN,
}


def no_comments(source: str) -> str:
    """Remove unescaped LaTeX comments; preserve line numbering."""
    cleaned = []
    for line in source.splitlines():
        for index, character in enumerate(line):
            if character != "%":
                continue
            before = line[:index]
            backslashes = len(before) - len(before.rstrip("\\"))
            if backslashes % 2 == 0:
                line = line[:index]
                break
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_font(font: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", font)


def span_list(page) -> list[dict]:
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]


def add_check(report: dict, name: str, ok: bool, **details) -> None:
    report["checks"].append({"check": name, "status": "pass" if ok else "fail", **details})


def graphics_paths(source: str) -> list[str]:
    """The directory prefixes declared in \\graphicspath, in declared order.

    A figure written as \\includegraphics{fig.pdf} is only findable through
    that declaration, so an auditor that ignores it reports every asset as
    missing on a document that compiles perfectly.
    """
    declaration = re.search(r"\\graphicspath\s*\{(.+?)\}\s*$", source, re.M)
    if not declaration:
        return []
    return re.findall(r"\{([^{}]*)\}", declaration[1])


def locate_asset(tex: Path, name: str, prefixes: Sequence[str] = ()) -> Path | None:
    """Allow compilation from main/ or its parent (the original archive layout)."""
    for base in (tex.parent, tex.parent.parent):
        for prefix in ("", *prefixes):
            candidate = base / prefix / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def width_inches(expression: str, line_width: float, column_width: float) -> float | None:
    expression = re.sub(r"\s+", "", expression)
    match = re.fullmatch(r"(\d*\.?\d*)\\(linewidth|textwidth|columnwidth)", expression)
    if match:
        factor = float(match[1]) if match[1] else 1.0
        return factor * {"linewidth": line_width, "textwidth": TEXT_WIDTH_IN,
                         "columnwidth": column_width}[match[2]]
    match = re.fullmatch(r"(\d*\.?\d+)(in|pt|bp|cm|mm)", expression)
    if match:
        return float(match[1]) * {"in": 1, "pt": 1 / 72.27, "bp": 1 / 72,
                                 "cm": 1 / 2.54, "mm": 1 / 25.4}[match[2]]
    return None


def source_audit(tex: Path, source: str, report: dict) -> tuple[set[str], int]:
    clean = no_comments(source)
    prefixes = graphics_paths(clean)
    # Stop at whatever actually follows the section: the bibliography may be
    # wrapped in a conditional, and matching \bibliography alone would then
    # swallow that wrapper into the marker and stop finding it in the PDF.
    acknowledgments = re.search(
        r"\\section\{Acknowledg(?:e)?ments\}(.*?)"
        r"(?:\\IfFileExists|\\bibliography|\\end\{document\})", clean, re.S)
    add_check(report, "acknowledgments_last_content_section", acknowledgments is not None
              and r"\section" not in acknowledgments[1])
    if acknowledgments:
        # The final words of the source paragraph locate its END in the PDF,
        # rather than only checking which page holds the section heading.
        report["acknowledgments_end_marker"] = " ".join(re.findall(r"\w+", acknowledgments[1])[-8:]).lower()
    tables = list(re.finditer(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", clean, re.S))
    for index, match in enumerate(tables, 1):
        body = match[2]
        captions = list(re.finditer(r"\\caption(?:\[[^]]*\])?\s*\{", body))
        tabular_ends = list(re.finditer(r"\\end\{tabular(?:\*|x)?\}", body))
        below = (len(captions) == 1 and bool(tabular_ends)
                 and captions[0].start() > tabular_ends[-1].end())
        add_check(report, f"source_table_{index}_caption_below", below,
                  line=clean.count("\n", 0, match.start()) + 1)
        add_check(report, f"source_table_{index}_no_scaling",
                  re.search(r"\\(?:resizebox|scalebox)\b", body) is None)
        add_check(report, f"source_table_{index}_no_tiny", r"\tiny" not in body)

    # Follow explicit environment widths rather than assuming that \linewidth
    # is a page width inside a nested subfigure.
    tokens = re.compile(
        r"\\(?P<action>begin|end)\{(?P<env>figure\*?|subfigure)\}"
        r"(?:\[[^]]*\])?(?:\{(?P<subwidth>[^{}]+)\})?"
        r"|\\includegraphics(?:\[(?P<options>[^]]*)\])?\{(?P<file>[^{}]+)\}"
    )
    stack: list[tuple[str, float, float]] = []
    seen = Counter()
    figure_fonts: set[str] = set()
    for token in tokens.finditer(clean):
        env = token["env"]
        if env:
            if token["action"] == "end":
                if stack and stack[-1][0] == env:
                    stack.pop()
                continue
            parent_line, parent_col = (stack[-1][1:] if stack else
                                       (COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
            if env == "figure*":
                stack.append((env, TEXT_WIDTH_IN, COLUMN_WIDTH_IN))
            elif env == "figure":
                stack.append((env, COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
            else:
                width = width_inches(token["subwidth"] or "", parent_line, parent_col)
                stack.append((env, width if width is not None else float("nan"),
                              width if width is not None else float("nan")))
            continue
        name = Path(token["file"]).name
        if name not in EXPECTED:
            continue
        seen[name] += 1
        options = token["options"] or ""
        width_option = re.search(r"(?:^|,)\s*width\s*=\s*([^,]+)", options)
        current_line, current_col = (stack[-1][1:] if stack else
                                     (COLUMN_WIDTH_IN, COLUMN_WIDTH_IN))
        width = width_inches(width_option[1], current_line, current_col) if width_option else None
        expected = EXPECTED[name]
        add_check(report, f"include_width:{name}",
                  width is not None and abs(width - expected) < 0.0001,
                  expected_inches=expected, measured_inches=width,
                  options=options, environments=[entry[0] for entry in stack])
        # Extra independent transformations defeat the width contract.
        add_check(report, f"include_no_extra_transform:{name}",
                  re.search(r"(?:^|,)\s*(?:scale|height|totalheight|angle|trim|viewport)\s*=", options) is None)
        asset = locate_asset(tex, token["file"], prefixes)
        add_check(report, f"figure_asset_exists:{name}", asset is not None)
        if asset is None:
            continue
        with pymupdf.open(asset) as document:
            page = document[0]
            actual_width = page.rect.width / 72
            spans = span_list(page)
            fonts = {normalize_font(span["font"]) for span in spans}
            figure_fonts.update(fonts)
            minimum = min((span["size"] for span in spans), default=None)
            add_check(report, f"figure_canvas:{name}", abs(actual_width - expected) < .0001,
                      canvas_inches=[round(page.rect.width / 72, 4), round(page.rect.height / 72, 4)])
            add_check(report, f"figure_source_font_floor:{name}",
                      minimum is not None and minimum >= MIN_FIGURE_PT - .01,
                      smallest_pdf_pt=minimum, fonts=sorted(fonts))
    for name in EXPECTED:
        add_check(report, f"include_once:{name}", seen[name] == 1, occurrences=seen[name])
    report["source_table_count"] = len(tables)
    return figure_fonts, len(tables)


def horizontal_rules(page) -> list[dict]:
    rules = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.width > 60 and rect.height < 1.5:
            rules.append({"x0": rect.x0, "x1": rect.x1,
                          "y": (rect.y0 + rect.y1) / 2,
                          "weight": max(drawing.get("width") or 0, rect.height)})
    return sorted(rules, key=lambda rule: rule["y"])


def inferred_table(page, caption: dict, rules: list[dict]) -> dict | None:
    """Find a nearby bottom rule and its matching heavier booktabs top rule."""
    x0, y0, x1, _ = caption["bbox"]
    center = (x0 + x1) / 2
    bottoms = [rule for rule in rules if 0 < y0 - rule["y"] <= 32
               and rule["x0"] - 2 <= center <= rule["x1"] + 2]
    if not bottoms:
        return None
    bottom = max(bottoms, key=lambda rule: rule["y"])
    matching = [rule for rule in rules
                if 10 < bottom["y"] - rule["y"] <= 450
                and abs(rule["x0"] - bottom["x0"]) < 1
                and abs(rule["x1"] - bottom["x1"]) < 1
                and abs(rule["weight"] - bottom["weight"]) < .08]
    if not matching:
        return None
    top = max(matching, key=lambda rule: rule["y"])
    # A top/bottom pair should enclose at least one thinner header rule.
    middle = [rule for rule in rules if top["y"] < rule["y"] < bottom["y"]
              and abs(rule["x0"] - bottom["x0"]) < 1
              and abs(rule["x1"] - bottom["x1"]) < 1
              and rule["weight"] < bottom["weight"] - .1]
    if not middle:
        return None
    rect = pymupdf.Rect(top["x0"] - 1, top["y"], bottom["x1"] + 1, bottom["y"])
    spans = [span for span in span_list(page)
             if rect.contains(pymupdf.Rect(span["bbox"]).tl)
             and rect.contains(pymupdf.Rect(span["bbox"]).br)]
    if not spans:
        return None
    return {"bbox": list(rect), "caption_gap_pdf_pt": y0 - bottom["y"],
            "smallest_pdf_pt": min(span["size"] for span in spans),
            "font_sizes_tex_pt": sorted({round(span["size"] / TEX_TO_PDF_PT, 3) for span in spans}),
            "span_count": len(spans), "method": "heuristic: matching booktabs top/bottom rules"}


def pdf_audit(pdf: Path, figure_fonts: set[str], source_tables: int, report: dict,
              render_dir: Path | None, dpi: int) -> None:
    with pymupdf.open(pdf) as document:
        report["pages"] = len(document)
        marker = report.get("acknowledgments_end_marker", "")
        end_pages = [index + 1 for index, page in enumerate(document)
                     if marker and marker in " ".join(re.findall(r"\w+", page.get_text())).lower()]
        report["content_end_page"] = end_pages[0] if len(end_pages) == 1 else None
        reference_pages = [index + 1 for index, page in enumerate(document)
                           if "References" in [line.strip() for line in page.get_text().splitlines()]]
        report["references_start_page"] = reference_pages[0] if len(reference_pages) == 1 else None
        add_check(report, "content_through_acknowledgments_within_9_pages",
                  len(end_pages) == 1 and end_pages[0] <= 9, end_pages=end_pages, limit=9)
        add_check(report, "references_heading_found_once", len(reference_pages) == 1,
                  pages=reference_pages)
        checked_fonts = set()
        figure_sizes = Counter()
        found_captions = Counter()
        caption_measurements = []
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            add_check(report, f"letter_page_{page_number}",
                      abs(page.mediabox.width - 612) < .1 and abs(page.mediabox.height - 792) < .1
                      and abs(page.rect.width - 612) < .1 and abs(page.rect.height - 792) < .1,
                      media_box=list(page.mediabox), visible_box=list(page.rect))
            for font in page.get_fonts(full=True):
                xref, extension, font_type, name = font[:4]
                if xref in checked_fonts:
                    continue
                checked_fonts.add(xref)
                try:
                    embedded = bool(document.extract_font(xref)[3])
                except (ValueError, RuntimeError):
                    embedded = False
                add_check(report, f"embedded_font_{xref}", embedded, font=name, type=font_type,
                          extension=extension)
                add_check(report, f"no_type3_font_{xref}", font_type.lower() != "type3", font=name)
            for span in span_list(page):
                if normalize_font(span["font"]) in figure_fonts:
                    figure_sizes[round(span["size"], 4)] += 1

            rules = horizontal_rules(page)
            for block in page.get_text("dict")["blocks"]:
                lines = block.get("lines", [])
                if not lines:
                    continue
                first = "".join(span["text"] for span in lines[0]["spans"])
                match = re.match(r"^Table\s+(\d+)\s*([.:])", first)
                if not match:
                    continue
                number = int(match[1])
                found_captions[number] += 1
                spans = [span for line in lines for span in line["spans"] if span["text"].strip()]
                # Superscripts/subscripts inside caption mathematics can be smaller.
                # Measure baseline caption text; keep all sizes in the report.
                base_spans = [span for span in spans if not (span["flags"] & 1)]
                bold = [span["text"] for span in spans if span["flags"] & 16]
                regular_size = all(abs(span["size"] - CAPTION_PT) < .08 for span in base_spans)
                add_check(report, f"table_{number}_caption_label", match[2] == ".",
                          page=page_number, prefix=match[0])
                add_check(report, f"table_{number}_caption_10pt", bool(base_spans) and regular_size,
                          page=page_number,
                          measured_tex_pt=sorted({round(span["size"] / TEX_TO_PDF_PT, 3) for span in spans}))
                add_check(report, f"table_{number}_caption_nonbold", not bold, bold_text=bold)
                measurement = {"table": number, "page": page_number, "caption_bbox": list(block["bbox"])}
                region = inferred_table(page, block, rules)
                if region:
                    measurement["body"] = region
                    # Rule association is heuristic, so body measurements trigger a
                    # review warning, not a conformance assertion or hard failure.
                    if region["smallest_pdf_pt"] < MIN_TABLE_PT - .02:
                        report["warnings"].append(
                            f"Table {number}: inferred body includes text below 7 TeX pt; inspect its page preview.")
                else:
                    report["warnings"].append(
                        f"Table {number}: could not infer its body from booktabs rules; visually inspect placement and size.")
                caption_measurements.append(measurement)
            if render_dir is not None:
                page.get_pixmap(dpi=dpi, alpha=False).save(render_dir / f"page-{page_number:02d}.png")

        add_check(report, "actual_figure_font_floor", bool(figure_sizes) and min(figure_sizes) >= MIN_FIGURE_PT - .01,
                  smallest_pdf_pt=min(figure_sizes) if figure_sizes else None,
                  font_families=sorted(figure_fonts), sizes_pdf_pt=dict(sorted(figure_sizes.items())))
        for number in range(1, source_tables + 1):
            add_check(report, f"table_{number}_caption_found_once", found_captions[number] == 1,
                      occurrences=found_captions[number])
        add_check(report, "table_caption_count", sum(found_captions.values()) == source_tables,
                  pdf_count=sum(found_captions.values()), source_count=source_tables)
        report["table_measurements"] = sorted(caption_measurements, key=lambda item: item["table"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex", type=Path, required=True, help="Main LaTeX source file")
    parser.add_argument("--pdf", type=Path, required=True, help="Compiled paper PDF")
    parser.add_argument("--json", type=Path, help="Optional machine-readable audit report")
    parser.add_argument("--render-dir", type=Path, help="Optional directory for per-page PNG previews")
    parser.add_argument("--dpi", type=int, default=120, help="Preview resolution (default: 120)")
    args = parser.parse_args()
    if not args.tex.is_file() or not args.pdf.is_file():
        parser.error("--tex and --pdf must name existing files")
    if args.dpi < 36 or args.dpi > 600:
        parser.error("--dpi must be between 36 and 600")
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    report = {"tex": str(args.tex.resolve()), "pdf": str(args.pdf.resolve()),
              "checks": [], "warnings": [],
              "limits": ["Table body rectangles are inferred from booktabs rules; inspect the reported regions visually.",
                         "Final figure font sizes are pooled by source font family, not attributed to individual xobjects.",
                         "Readability, grayscale contrast, page balance and semantic accuracy require visual review."]}
    try:
        fonts, tables = source_audit(args.tex, args.tex.read_text(encoding="utf-8-sig"), report)
        pdf_audit(args.pdf, fonts, tables, report, args.render_dir, args.dpi)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    failures = [check for check in report["checks"] if check["status"] == "fail"]
    report["passed"] = not failures
    report["failure_count"] = len(failures)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Pages: {report['pages']}; reliable checks: {len(report['checks'])}; failures: {len(failures)}")
    for check in failures:
        print(f"FAIL {check['check']}")
    for warning in report["warnings"]:
        print(f"REVIEW {warning}")
    for table in report["table_measurements"]:
        body = table.get("body")
        if body:
            minimum = body["smallest_pdf_pt"] / TEX_TO_PDF_PT
            print(f"Table {table['table']}: page {table['page']}, inferred minimum body font {minimum:.2f} TeX pt")
    print("PASS: reliable checks passed; visual review remains required." if not failures else
          "FAIL: resolve the reliable failures and review the measurements.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
