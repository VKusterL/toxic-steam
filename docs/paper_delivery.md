# Corrected paper and Overleaf delivery

The manuscript sources remain in the local `AIIDE26___Toxic_Steam__A_Large_Scale_Characterization_and_Prediction_of_Toxic_Users_in_a_Gaming_Platform/main/` directory, which is excluded from Git. The public research artifact and the manuscript are packaged separately.

## Build

From the repository root, with `pdflatex`, `bibtex`, and Python with PyMuPDF installed:

```bash
python tools/package_paper.py
```

This compiles a clean copy, checks references and layout, and writes:

- `dist/toxic-steam-overleaf.zip`: the complete LaTeX project, with `main.tex` at the ZIP root, style files, bibliography, and the six used figure PDFs.
- `dist/toxic-steam-camera-ready.pdf`: the locally compiled paper.
- `dist/paper-layout-audit.json`: font, caption, page-size and page-limit checks.
- `dist/paper-build.log`: the final compiler log.

No source files, bibliography entries, research outputs, or previously regenerated figures are deleted by packaging. The ZIP excludes unused t-SNE figures, earlier manuscript copies, compiler auxiliaries, and repository data.

## Upload to Overleaf

1. Create a new project using **New Project → Upload Project** and select `toxic-steam-overleaf.zip`.
2. Set the main document to **main.tex**, at the project root, and the compiler to **pdfLaTeX**.
3. Recompile from scratch. The source paths are relative to this root and need no custom environment variables.
4. Confirm that **Acknowledgments ends on page 9** and **References begins on page 10**. The verified v3 build uses the full content allowance: Acknowledgments ends at the bottom of page 9's second column, followed by two reference pages (**11 pages in total**). The packaging script checks these page boundaries before producing the ZIP.
5. Download the resulting PDF and the updated source ZIP for the conference correction. Its request covers both the PDF and the sources.

To update an existing project, first preserve a copy, then replace the corresponding files with the ZIP contents and select its root `main.tex`. Do not upload the whole research repository or the separate artifact ZIP to Overleaf. Preserve the specified figure widths, table typography, margins, and caption settings.

## Editorial changes

- Figure 1 uses column width; the TF-IDF heatmap uses both columns at text width. The three CDF panels retain their authored widths, and the shared legend is included at 45% of text width.
- The six PDFs come from the latest `data/figures-output/` generation. Figures were not redrawn during the manuscript revision.
- All six table captions follow their tables, start with `Table n.`, and use regular 10-point type.
- Table bodies use 10-point type, except the lexical and main prediction tables, which use 9-point type. Wide prediction tables span two columns. No table is scaled with `resizebox`.
- Prediction prose was adjusted to accommodate the larger floats while using all nine content pages. The v3 revision restores methodological detail and interpretation from the earlier manuscript, particularly the modeling population, feature construction, evaluation metrics, LTO control, and LLM narratives. Every table and citation key is preserved; text outside prediction, including limitations, conclusion, and acknowledgments, is unchanged from v2. The initial characterization wording was adjusted to match the figure's term ordering and qualifying-tag comparison.

The PDF checker measures embedded fonts and actual final text sizes, not only plotting settings. Its table-region inference is heuristic; the delivered pages were also inspected visually. These checks address the supplied editorial feedback and do not constitute acceptance by the publisher.

## Research artifact coverage

The layout revision does not fill missing experimental provenance. Before describing the repository as a complete reproduction, recover the original four-confirmatory-test analysis and the fold-level balanced-evaluation export identified in [results.md](results.md). Use the filtered package described in [artifact.md](artifact.md) for public delivery; the original research checkout contains identifiable user-level records.
