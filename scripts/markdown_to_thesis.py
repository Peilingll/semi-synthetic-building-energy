from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(r"D:\ITBE\Thesis\semi-synthetic-building-energy")
SOURCE = ROOT / "doc_processed" / "Thesis" / "draft_V8_08.18_en.md"
TEMPLATE = ROOT / "thesistemplate-main"
CONTENT = TEMPLATE / "content"

FIGURES: dict[str, tuple[str, str]] = {
    "3.1": (
        "fig_3_1_workflow.png",
        "End-to-end study workflow.",
    ),
    "3.2": (
        "fig_3_2_layers.png",
        "Three-layer semi-synthetic dataset structure.",
    ),
    "3.3": (
        "fig_3_3_svi_workflow.png",
        "SVI acquisition and processing workflow.",
    ),
    "3.4": (
        "fig_3_4_examples.png",
        "Representative street-view crops by city and size class.",
    ),
    "3.5": (
        "fig_3_5_vision_models.png",
        "Vision-model configurations and multi-image aggregation.",
    ),
    "3.6": (
        "fig_3_6_tabula_matrix.png",
        "TABULA-NL residential typology matrix.",
    ),
    "4.1": (
        "fig_4_1_distributions.png",
        "Experimental-sample attribute and EPC-label distributions by city.",
    ),
}

REFERENCE_PREFIXES = {
    "Algemene Rekenkamer. (2016).": "algemene2016",
    "Buckley,": "buckley2021",
    "CA EPBD. (n.d.).": "caepbd",
    "CBS (Statistics Netherlands).": "cbs",
    "Chen,": "chen2024",
    "Concerted Action EPBD. (2020).": "caepbd2020",
    "Dabrock,": "dabrock2025",
    "Dalach,": "dalach2025",
    "Dukai,": "dukai2021",
    "European Commission. (2024).": "epbd2024",
    "Hettinga,": "hettinga2023",
    "Ke,": "ke2017",
    "Khayatian,": "khayatian2016",
    "Liang,": "liang2025",
    "Liu,": "liu2025",
    "Loga,": "loga2016",
    "Mayer, K., Haas,": "mayer2023uk",
    "Mayer, K., Heilborn,": "mayer2023dk",
    "Mapillary. (2026).": "mapillary2026",
    "Oquab,": "oquab2023",
    "Pan,": "pan",
    "Peters,": "peters2022",
    "Reinhart,": "reinhart2016",
    "RVO (Rijksdienst": "rvo2011",
    "Sun,": "sun2026",
    "TABULA Project Team. (2012).": "tabula2012",
    "Wurm,": "wurm2021",
    "Zeng,": "zeng2024",
}

CITATION_LINKS = {
    "(European Parliament and Council of the European Union, 2024)": "epbd2024",
    "(Concerted Action EPBD, 2020)": "caepbd2020",
    r"(Reinhart \& Cerezo Davila, 2016)": "reinhart2016",
    "(Loga et al., 2016)": "loga2016",
    "Loga et al. (2016)": "loga2016",
    "Hettinga et al. (2023)": "hettinga2023",
    "Wurm et al. (2021)": "wurm2021",
    "Mayer et al. (2022, 2023)": "mayer2023uk",
    "Mayer et al. (2023)": "mayer2023uk",
    r"(Mayer, Heilborn, \& Fischer, 2023)": "mayer2023dk",
    "(Liang et al., 2025)": "liang2025",
    "Dabrock et al. (2025)": "dabrock2025",
    "Liu et al. (2025)": "liu2025",
    "Sun et al. (2026)": "sun2026",
    "(Oquab et al., 2023)": "oquab2023",
    "Zeng et al. (2024)": "zeng2024",
    "(Dalach et al., 2025)": "dalach2025",
    "(Ke et al., 2017)": "ke2017",
}


def protect_inline(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def token(value: str) -> str:
        key = f"ZZPROTECTED{len(protected)}ZZ"
        protected[key] = value
        return key

    text = re.sub(r"`([^`]+)`", lambda m: token(r"\texttt{" + escape_code(m.group(1)) + "}"), text)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: token(r"\(" + m.group(1) + r"\)"), text)
    text = re.sub(r"https?://[^\s)]+", lambda m: token(r"\url{" + m.group(0) + "}"), text)
    return text, protected


def escape_code(text: str) -> str:
    text = (
        text.replace("≤", "<=")
        .replace("≥", ">=")
        .replace("°", " deg")
        .replace("²", "^2")
    )
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("&", r"\&")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def inline(text: str) -> str:
    text, protected = protect_inline(text)
    # The Markdown draft uses both *...* and _..._ for emphasis. At this point
    # code spans, maths, and URLs are protected, so underscores there are safe.
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"*\1*", text)
    replacements = {
        "–": "--",
        "—": "---",
        "−": "-",
        "×": r"\ensuremath{\times}",
        "±": r"\ensuremath{\pm}",
        "≥": r"\ensuremath{\geq}",
        "≤": r"\ensuremath{\leq}",
        "→": r"\ensuremath{\rightarrow}",
        "κ": r"\ensuremath{\kappa}",
        "²": r"\textsuperscript{2}",
        "°": r"\ensuremath{^\circ}",
        "’": "'",
        "“": "``",
        "”": "''",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("\\", "ZZBACKSLASHZZ")
    for char, escaped in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$")]:
        text = text.replace(char, escaped)
    text = text.replace("_", r"\_")
    text = text.replace("ZZBACKSLASHZZ", "\\")

    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", text)
    text = text.replace("H'tr", r"\(H'_{\mathrm{tr}}\)")

    for key, value in protected.items():
        text = text.replace(key, value)
    text = text.replace(
        r"\texttt{OpenGVLab/InternVL3-2B}",
        r"\texttt{OpenGVLab/}\allowbreak\texttt{InternVL3-2B}",
    )
    text = re.sub(
        r"^Section(?:~|\s)+([0-9]+(?:\.[0-9]+)*)",
        lambda m: rf"\Cref{{sec:{m.group(1).replace('.', '_')}}}",
        text,
    )
    text = re.sub(
        r"(?:§\s*|\bSection(?:~|\s)+)([0-9]+(?:\.[0-9]+)*)",
        lambda m: rf"\cref{{sec:{m.group(1).replace('.', '_')}}}",
        text,
    )
    text = re.sub(
        r"^Table(?:~|\s)+([A-Z0-9]+(?:\.[0-9]+)*)",
        lambda m: rf"\Cref{{tab:{m.group(1).replace('.', '_')}}}",
        text,
    )
    text = re.sub(
        r"\bTable(?:~|\s)+([A-Z0-9]+(?:\.[0-9]+)*)",
        lambda m: rf"\cref{{tab:{m.group(1).replace('.', '_')}}}",
        text,
    )
    text = re.sub(
        r"^(?:Fig\.|Figure)(?:~|\s)+([0-9]+(?:\.[0-9]+)*)",
        lambda m: rf"\Cref{{fig:{m.group(1).replace('.', '_')}}}",
        text,
    )
    text = re.sub(
        r"\b(?:Fig\.|Figure)(?:~|\s)+([0-9]+(?:\.[0-9]+)*)",
        lambda m: rf"\cref{{fig:{m.group(1).replace('.', '_')}}}",
        text,
    )
    for citation, target in sorted(CITATION_LINKS.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(citation, rf"\hyperlink{{ref:{target}}}{{{citation}}}")
    return text


def heading_title(raw: str) -> tuple[str, str | None]:
    raw = raw.strip()
    number = None
    m = re.match(r"(?:CH)?([0-9]+(?:\.[0-9]+)*)[_\s]+(.+)", raw)
    if m:
        number = m.group(1)
        raw = m.group(2)
    raw = raw.replace("_", " ")
    return inline(raw), number


def table_caption(raw: str) -> tuple[str | None, str | None]:
    m = re.match(r"\*\*Table\s+([A-Z0-9.]+)\s*(?:---|—|-|\.)\s*(.+?)\*\*\s*(.*)$", raw.strip())
    if not m:
        return None, None
    caption = m.group(2).rstrip(".")
    if m.group(3).strip():
        caption += " " + m.group(3).strip()
    return m.group(1).rstrip("."), inline(caption)


def parse_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def render_table(rows: list[list[str]], align: list[str], caption: str | None, label: str | None) -> list[str]:
    if label == "4.1":
        rows = [
            ["City", "Dev. buildings", "Dev. images", "Hold-out buildings", "Hold-out images", "Total buildings", "Total images"],
            *rows[1:],
        ]
    cols = len(rows[0])
    specs = []
    for idx in range(cols):
        direction = "r" if idx < len(align) and align[idx] == "r" else "l"
        weight = "1.7" if idx == 0 and cols >= 4 else "1"
        specs.append(f"X[{weight},{direction}]")
    column_load = sum(max(len(row[idx]) if idx < len(row) else 0 for row in rows) for idx in range(cols))
    landscape = cols >= 5 and column_load > 220
    output: list[str] = []
    if landscape:
        output.append(r"\begin{landscape}")
    output.extend([r"\begin{table}[p]" if landscape else r"\begin{table}[H]", r"\centering"])
    if caption:
        output.append(rf"\caption{{{caption}}}")
    if label:
        output.append(rf"\label{{tab:{label.replace('.', '_')}}}")
    output.append(r"\small" if cols <= 4 or landscape else r"\footnotesize")
    output.extend([
        r"\begin{tblr}{",
        r"  width = \linewidth,",
        "  colspec = {" + " ".join(specs) + "},",
        r"  row{1} = {font=\bfseries},",
        r"  rowsep = 3pt,",
        r"  colsep = 4pt",
        r"}",
        r"\toprule",
    ])
    output.append(" & ".join(inline(cell) for cell in rows[0]) + r" \\")
    output.append(r"\midrule")
    for row in rows[1:]:
        row = row + [""] * (cols - len(row))
        output.append(" & ".join(inline(cell) for cell in row[:cols]) + r" \\")
    output.extend([r"\bottomrule", r"\end{tblr}", r"\end{table}"])
    if landscape:
        output.append(r"\end{landscape}")
    output.append("")
    return output


def draft_float(text: str) -> list[str]:
    stripped = re.sub(r"^>\s*", "", text.strip())
    stripped = stripped.strip("_").replace("**", "")
    fig = re.match(r"(?:Fig(?:ure)?\.?\s*)([0-9]+\.[0-9]+)\s*(?:---|—|:|\.)?\s*(.*)", stripped, re.I)
    tab = re.match(r"(?:Table\s*)([0-9]+\.[0-9]+)\s*(?:---|—|:|\.)?\s*(.*)", stripped, re.I)
    match, kind = (fig, "figure") if fig else ((tab, "table") if tab else (None, "quote"))
    if not match:
        return [r"\begin{quote}", inline(stripped), r"\end{quote}", ""]
    number, caption = match.group(1), inline(match.group(2).rstrip(".。"))
    label = number.replace(".", "_")
    title = "Figure placeholder" if kind == "figure" else "Table placeholder"
    if kind == "figure" and number in FIGURES:
        filename, caption = FIGURES[number]
        wide = number in {"3.1", "3.2", "3.3", "3.5"}
        result: list[str] = []
        if wide:
            result.append(r"\begin{landscape}")
        result.extend([
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.98\linewidth,height={'0.78' if wide else '0.70'}\textheight,keepaspectratio]{{figures/fig/{filename}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{fig:{label}}}",
            r"\end{figure}",
            "",
        ])
        if wide:
            result.extend([r"\end{landscape}", ""])
        if number == "3.3":
            result.extend([
                r"\begin{figure}[htbp]",
                r"\centering",
                r"\includegraphics[width=0.98\linewidth,height=0.38\textheight,keepaspectratio]{figures/fig/fig_3_3_detail.png}",
                r"\caption*{Panorama detection and facade-crop reprojection.}",
                r"\end{figure}",
                "",
            ])
        return result
    return [
        rf"\begin{{{kind}}}[htbp]",
        r"\centering",
        r"\fbox{\begin{minipage}{0.88\textwidth}\centering",
        rf"\textbf{{{title}}}\\[0.6em]",
        caption,
        r"\end{minipage}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{'fig' if kind == 'figure' else 'tab'}:{label}}}",
        rf"\end{{{kind}}}",
        "",
    ]


def is_special(line: str) -> bool:
    s = line.strip()
    return (
        not s
        or s.startswith("#")
        or s.startswith("|")
        or s.startswith(">")
        or s.startswith("\\[")
        or s == "---"
        or bool(re.match(r"(?:[-*]|\d+\.)\s+", s))
        or bool(table_caption(s)[0])
        or s.startswith("_Figure")
    )


def convert(lines: list[str], default_table_21: bool = False) -> str:
    out: list[str] = []
    pending_caption: tuple[str | None, str | None] = (None, None)
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s or s == "---":
            out.append("")
            i += 1
            continue
        if s.startswith("\\["):
            block = [s]
            i += 1
            while i < len(lines):
                block.append(lines[i].rstrip())
                if lines[i].strip().endswith("\\]"):
                    i += 1
                    break
                i += 1
            out.extend(block + [""])
            continue
        cap_number, cap_text = table_caption(s)
        if cap_number:
            pending_caption = (cap_number, cap_text)
            i += 1
            continue
        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            raw = s[level:].strip()
            title, number = heading_title(raw)
            if level == 1:
                raw_title = re.sub(r"^CH\d+[_\s]+", "", raw)
                raw_title = re.sub(r"^Appendix\s+[A-Z]\s*(?:---|—|-)?\s*", "", raw_title)
                out.append(rf"\chapter{{{inline(raw_title)}}}")
                if number:
                    out.extend([
                        rf"\label{{ch:{number.replace('.', '_')}}}",
                        rf"\label{{sec:{number.replace('.', '_')}}}",
                    ])
                out.append("")
            else:
                command = {2: "section", 3: "subsection", 4: "subsubsection"}.get(level, "paragraph")
                out.append(rf"\{command}{{{title}}}")
                if number:
                    out.append(rf"\label{{sec:{number.replace('.', '_')}}}")
                out.append("")
            i += 1
            continue
        if s.startswith("_Figure"):
            out.extend(draft_float(s.strip("_")))
            i += 1
            continue
        if s.startswith(">"):
            out.extend(draft_float(s))
            i += 1
            continue
        if s.startswith("|"):
            raw_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw_rows.append(parse_cells(lines[i]))
                i += 1
            align: list[str] = []
            if len(raw_rows) > 1 and is_separator_row(raw_rows[1]):
                for cell in raw_rows[1]:
                    align.append("r" if cell.strip().endswith(":") else "l")
                raw_rows.pop(1)
            number, caption = pending_caption
            pending_caption = (None, None)
            if not number and default_table_21:
                number, caption = "2.1", "Comparison framework for literature-review routes"
                default_table_21 = False
            out.extend(render_table(raw_rows, align, caption, number))
            continue
        if re.match(r"[-*]\s+", s):
            items = []
            while i < len(lines) and re.match(r"[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            out.append(r"\begin{itemize}")
            out.extend(r"\item " + inline(item) for item in items)
            out.extend([r"\end{itemize}", ""])
            continue
        if re.match(r"\d+\.\s+", s):
            items = []
            while i < len(lines) and re.match(r"\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append(r"\begin{enumerate}")
            out.extend(r"\item " + inline(item) for item in items)
            out.extend([r"\end{enumerate}", ""])
            continue
        paragraph = [s]
        i += 1
        while i < len(lines) and not is_special(lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        out.extend([inline(" ".join(paragraph)), ""])
    return "\n".join(out).strip() + "\n"


def split_sections(lines: list[str]) -> dict[str, list[str]]:
    starts: list[tuple[int, str]] = []
    patterns = [
        (r"^# CH1", "01_introduction.tex"),
        (r"^# CH2", "02_literature_review.tex"),
        (r"^# CH3", "03_method.tex"),
        (r"^# CH4", "04_experiments.tex"),
        (r"^# CH5", "05_discussion_conclusion.tex"),
        (r"^# Appendix A", "appendix_a.tex"),
        (r"^# Appendix B", "appendix_b.tex"),
        (r"^# References", "99_references.tex"),
    ]
    for idx, line in enumerate(lines):
        for pattern, name in patterns:
            if re.match(pattern, line):
                starts.append((idx, name))
                break
    result: dict[str, list[str]] = {}
    for pos, (start, name) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        result[name] = lines[start:end]
    return result


def make_references(lines: list[str]) -> str:
    refs = [re.sub(r"^-\s+", "", line.strip()) for line in lines if line.strip().startswith("- ")]
    out = [r"\chapter*{References}", r"\addcontentsline{toc}{chapter}{References}", r"\begingroup\sloppy", ""]
    for ref in refs:
        target = next((key for prefix, key in REFERENCE_PREFIXES.items() if ref.startswith(prefix)), None)
        anchor = rf"\hypertarget{{ref:{target}}}{{}}" if target else ""
        out.extend([r"\noindent " + anchor + inline(ref), r"\par\medskip"])
    out.append(r"\endgroup")
    return "\n".join(out) + "\n"


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    sections = split_sections(lines)
    CONTENT.mkdir(parents=True, exist_ok=True)
    for name, section_lines in sections.items():
        if name == "99_references.tex":
            output = make_references(section_lines)
        else:
            output = convert(section_lines, default_table_21=(name == "02_literature_review.tex"))
        (CONTENT / name).write_text(output, encoding="utf-8", newline="\n")

    abstract = r"""\section*{Abstract}
\addcontentsline{toc}{section}{Abstract}
\textit{Work in progress. The current Draft V8 does not yet contain an abstract.}
"""
    (CONTENT / "00c_abstract.tex").write_text(abstract, encoding="utf-8", newline="\n")

    thesis = r"""\documentclass[
fleqn,
11pt,
oneside,
print=false,
language=english,
citeStyle=none,
thesis=true
]{CMSthesis}

\SetType{Master of Science (M.Sc.)}
\SetFieldOfStudy{[To be completed]}
\SetTopic{\raggedright Quantifying the Performance Cost of\\
Replacing Registry Metadata with\\
Street-View-Derived Building Attributes}
\SetSubtopic{\raggedright in Archetype-Based UBEM Workflows}
\SetAuthor{Pei-Ling Song}
\SetMatrikulationNo{03798081}
\SetEmail{peiling.song@gmail.com}
\SetZipCode{80995}
\SetLocation{München}
\SetStreet{Franz-Kötterl-Straße 19a}
\SetStart{[Date of issue]}
\SetEnd{\today}
\AddSupervisor{Agata Dalach}
\AddSupervisor{Jose Quesada Allerhand}
\AddAffil{\DAchair}

\input{codestyles}
\usepackage{float}
\renewcommand{\ttdefault}{lmtt}
\emergencystretch=2em
\graphicspath{{{./figures/fig/}}}
\captionsetup[table]{position=top,skip=6pt}
\captionsetup[figure]{position=bottom,skip=6pt}
\hypersetup{
  pdftitle={Quantifying the Performance Cost of Replacing Registry Metadata with Street-View-Derived Building Attributes in Archetype-Based UBEM Workflows},
  pdfauthor={Pei-Ling Song},
  bookmarksopen=true
}

\begin{document}
\pagestyle{empty}
\pagenumbering{Alph}
\maketitle
\pagenumbering{Roman}
\cleardoublepage

\include{content/00c_abstract}
\tableofcontents
\cleardoublepage
\listoftables
\cleardoublepage
\listoffigures
\cleardoublepage

\parskip 8pt
\pagenumbering{arabic}
\include{content/01_introduction}
\include{content/02_literature_review}
\include{content/03_method}
\include{content/04_experiments}
\include{content/05_discussion_conclusion}

\begin{appendix}
\include{content/appendix_a}
\include{content/appendix_b}
\end{appendix}

\include{content/99_references}
\Declaration
\cleardoublepage
\end{document}
"""
    (TEMPLATE / "thesis.tex").write_text(thesis, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
