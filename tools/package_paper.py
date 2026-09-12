"""Compile a clean manuscript copy and create an Overleaf-ready source ZIP.

Run from the repository root: python tools/package_paper.py
Requires pdflatex, bibtex, and PyMuPDF for the final layout/page-limit check.
The original manuscript and research results are preserved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
README = """# Overleaf delivery

Upload toxic-steam-overleaf.zip using New Project > Upload Project.
Select main.tex as the main document and pdfLaTeX as the compiler.
Recompile from scratch. All six figure PDFs and the bibliography are included.
Do not change figure widths or scale the tables: the font sizes were checked
at these final dimensions. The verified local PDF has {content_pages} content
pages through Acknowledgments and {total_pages} pages in total. References
start on page {references_page}.

To update an existing project instead, paste this main.tex over the old one
and replace all six PDFs in the project's images folder. main.tex declares
\\graphicspath{{{{images/}}{{main/images/}}}} and guards \\bibliography with
\\IfFileExists, so it compiles whether the sources sit at the project root or
inside main/. Replacing the figures is not optional: the earlier ones have
different canvas heights and would be rescaled by LaTeX, which is exactly
the font-size problem the editorial review raised.

The public research-artifact ZIP is a separate deliverable and is not an
Overleaf project. After checking the Overleaf PDF, upload both the corrected
PDF and the updated LaTeX sources to the conference's paper submission.
"""


def run(command: list[str], directory: Path) -> None:
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stdout[-6000:]}\n{result.stderr[-1000:]}")


def main() -> None:
    candidates = list(ROOT.glob("AIIDE26*/main/main.tex"))
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one AIIDE26 manuscript main/main.tex")
    source_dir = candidates[0].parent
    text = candidates[0].read_text(encoding="utf-8")
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    # \includegraphics carries a bare filename so the same main.tex compiles
    # in the Overleaf project and in this flat package; the directory comes
    # from \graphicspath. Resolve through it and keep the figure at the
    # prefix it was found under, which is what \graphicspath then searches.
    prefixes = re.findall(r"\{([^{}]*)\}", (
        re.search(r"\\graphicspath\s*\{(.+?)\}\s*$", text, re.M) or ["", ""])[1])
    resolved = set()
    for figure in figures:
        for prefix in ("", *prefixes):
            if (source_dir / prefix / figure).is_file():
                resolved.add(f"{prefix}{figure}")
                break
        else:
            raise RuntimeError(f"Figure not found under \\graphicspath: {figure}")
    names = {"main.tex", "references.bib", "aaai24.sty", "aaai24.bst", *resolved}
    for name in names:
        source = source_dir / name
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(source_dir.resolve()):
            raise ValueError(f"Source escapes manuscript directory: {name}")
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper-build-", dir=destination) as temporary:
        work = Path(temporary)
        for name in sorted(names):
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / name, target)
        latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        run(latex, work)
        run(["bibtex", "main"], work)
        run(latex, work)
        run(latex, work)
        log = (work / "main.log").read_text(encoding="utf-8", errors="replace")
        if re.search(r"undefined (?:references|citations)|Overfull \\[hv]box|LaTeX Error", log):
            raise RuntimeError("Final LaTeX log contains unresolved references or layout overflow")
        pdf = destination / "toxic-steam-camera-ready.pdf"
        shutil.copyfile(work / "main.pdf", pdf)
        shutil.copyfile(work / "main.log", destination / "paper-build.log")
        report = destination / "paper-layout-audit.json"
        run([sys.executable, str(ROOT / "tools/check_paper_layout.py"),
             "--tex", str(source_dir / "main.tex"), "--pdf", str(pdf), "--json", str(report)], ROOT)
        audit = json.loads(report.read_text(encoding="utf-8"))
        if audit["content_end_page"] != 9 or audit["references_start_page"] != 10:
            raise RuntimeError("Delivery must end Acknowledgments on page 9 and start references on page 10")
        (work / "README_OVERLEAF.md").write_text(
            README.format(content_pages=audit["content_end_page"], total_pages=audit["pages"],
                          references_page=audit["references_start_page"]), encoding="utf-8")
        archive = destination / "toxic-steam-overleaf.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in sorted(names | {"README_OVERLEAF.md"}):
                bundle.write(work / name, name)
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None or set(bundle.namelist()) != names | {"README_OVERLEAF.md"}:
                raise RuntimeError("Source archive failed integrity validation")
    print(f"PDF: {pdf}")
    print(f"Overleaf ZIP: {archive}")
    print(f"ZIP SHA-256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
    print(f"Layout/page-limit report: {report}")


if __name__ == "__main__":
    main()
