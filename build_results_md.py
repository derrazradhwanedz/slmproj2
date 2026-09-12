"""Build the results section (Markdown and Word) from its text and the per-model tables.

Inputs
    results/analysis/results_text.json   ordered list of {title, paragraph, caption}
    results/analysis/per_model/table.N_<section>_<model>.xlsx   (from analysis_v4.py)
Outputs (generated - edit the JSON, not these files)
    results/analysis/results_section.md
    results/analysis/results_section.docx   (via pandoc, which must be on PATH)

Numbering and layout, driven by each block's title in the JSON:
    ""                       -> plain paragraph under "3. Results"
    "1. Section name"        -> "3.1 Section name" heading (the JSON number sets the order)
    "Model (Table N)"        -> "3.1.1 Model" heading, paragraph, caption, then table N
    any other title          -> "3.1.k Title" heading, paragraph (e.g. Summary)
Tables: the Excel best cell (dark grey) is written in bold, the worst (light
grey) in italics. In the Word file, headings are black and bold, table
header rows are shaded light grey, and a rule is drawn above the
"All datasets" rows and below the last row of each table.

Run from the repository root as: python build_results_md.py
"""

import glob
import io
import json
import os
import re
import subprocess
import zipfile

from openpyxl import load_workbook

ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(ROOT, "results", "analysis")
TEXT_PATH = os.path.join(ANALYSIS_DIR, "results_text.json")
PATTERNS_PATH = os.path.join(ANALYSIS_DIR, "patterns.json")
TABLE_DIR = os.path.join(ANALYSIS_DIR, "per_model")
MD_PATH = os.path.join(ANALYSIS_DIR, "results_section.md")
DOCX_PATH = os.path.join(ANALYSIS_DIR, "results_section.docx")
REFERENCE_DOCX = os.path.join(ANALYSIS_DIR, "reference.docx")

SECTION_NUMBER = 3  # "Results" is section 3 of the paper
DOC_TITLE = "Results"
DARK_GRAY, LIGHT_GRAY = "BFBFBF", "F2F2F2"
HEADER_FILL = "D9D9D9"  # light grey table header row in Word
FONT = "Times New Roman"
BODY_SIZE, TABLE_SIZE = 22, 18  # half-points: 11 pt body and headings, 9 pt tables
DATASET_LABEL = {
    "aqua": "AQUA", "asdiv": "ASDiv", "clutrr": "CLUTRR", "date": "Date", "gsm8k": "GSM8K",
    "MultiArith": "MultiArith", "QASports": "QASports", "saycan": "SayCan", "StrategyQA": "StrategyQA",
    "SVAMP": "SVAMP", "All datasets": "**All datasets**",
}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def find_table(number):
    """Path of the per-model workbook for table `number`."""
    matches = glob.glob(os.path.join(TABLE_DIR, f"table.{number}_*.xlsx"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file for Table {number} in {TABLE_DIR}, found {len(matches)}")
    return matches[0]


def xlsx_to_markdown(path):
    """Markdown table from a per-model workbook; dataset shown once per SCoT/MGCoT pair."""
    ws = load_workbook(path).active
    header = [c.value for c in ws[1]]
    lines = ["| " + " | ".join(["Dataset", "Mechanism"] + header[2:]) + " |",
             "|" + "|".join([":---", ":---"] + ["---:"] * (len(header) - 2)) + "|"]
    for r in range(2, ws.max_row + 1):
        dataset = ws.cell(r, 1).value
        cells = [DATASET_LABEL.get(dataset, dataset) if r % 2 == 0 else "", ws.cell(r, 2).value]
        for c in range(3, ws.max_column + 1):
            value, fill = ws.cell(r, c).value, ws.cell(r, c).fill.fgColor.rgb[-6:]
            cells.append(f"**{value}**" if fill == DARK_GRAY else f"*{value}*" if fill == LIGHT_GRAY else value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def format_p(value):
    """p-value for the text: '< 0.001' below that threshold, three decimals otherwise."""
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def stats_sentence(table_number):
    """Aggregate (all-datasets) BH-corrected p-value of every metric in one table."""
    with open(PATTERNS_PATH, encoding="utf-8") as f:
        patterns = json.load(f)
    facts = next(t for t in patterns if t["table"] == int(table_number))["metrics"]
    tests = "; ".join(f"{f['label']} p {format_p(f['overall']['p_fdr'])}" for f in facts.values())
    return f"Aggregate paired comparisons over all 500 questions (Wilcoxon signed-rank, BH-corrected): {tests}."


def render(blocks):
    """Numbered Markdown for all blocks."""
    parts = [f"# {SECTION_NUMBER}. {DOC_TITLE}"]
    section, sub = 0, 0
    for block in blocks:
        title, paragraph, caption = block["title"], block["paragraph"], block["caption"]
        section_match = re.match(r"^\d+\.\s+(.*)$", title)
        table_ref = re.search(r"\s*\(Table (\d+)\)", title)
        if section_match:
            section, sub = section + 1, 0
            parts.append(f"## {SECTION_NUMBER}.{section} {section_match.group(1)}")
        elif title:
            sub += 1
            name = title[:table_ref.start()] if table_ref else title
            parts.append(f"### {SECTION_NUMBER}.{section}.{sub} {name}")
        if paragraph and table_ref:
            # state the tests behind the "significant" claims, then point to the table
            paragraph = f"{paragraph.rstrip()} {stats_sentence(table_ref.group(1))}"
            ref = f"(Table {table_ref.group(1)})"
            paragraph = f"{paragraph[:-1]} {ref}." if paragraph.endswith(".") else f"{paragraph} {ref}"
        if paragraph:
            parts.append(paragraph)
        if table_ref:
            if caption:
                parts.append(re.sub(r"^(Table \d+\.)", r"**\1**", caption))
            parts.append(xlsx_to_markdown(find_table(table_ref.group(1))))
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Word
# --------------------------------------------------------------------------

def build_reference_docx():
    """Pandoc's default reference.docx restyled for the paper.

    Times New Roman throughout: 11 pt justified body text, 11 pt black bold
    headings, 9 pt table text, and a light grey table header row.
    """
    default = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                             capture_output=True, check=True).stdout
    src = zipfile.ZipFile(io.BytesIO(default))
    styles = src.read("word/styles.xml").decode("utf-8")
    font = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}" />'

    def size(half_points):
        return f'<w:sz w:val="{half_points}" /><w:szCs w:val="{half_points}" />'

    def patch_default(match):
        block = re.sub(r"<w:rFonts [^>]*/>", font, match.group(0))
        return re.sub(r'<w:sz w:val="\d+" />\s*<w:szCs w:val="\d+" />', size(BODY_SIZE), block)

    def patch_heading(match):
        block = re.sub(r"<w:rFonts [^>]*/>", font, match.group(0))
        block = re.sub(r'<w:sz w:val="\d+" />\s*<w:szCs w:val="\d+" />', size(BODY_SIZE), block)
        return re.sub(r"<w:color [^>]*/>", '<w:color w:val="000000" /><w:b /><w:bCs />', block)

    styles = re.sub(r"<w:docDefaults>.*?</w:docDefaults>", patch_default, styles, flags=re.S)
    styles = re.sub(r'<w:style [^>]*w:styleId="Heading[1-3]".*?</w:style>', patch_heading, styles, flags=re.S)
    # justified body text
    styles = re.sub(r'(<w:style [^>]*w:styleId="BodyText".*?<w:spacing [^>]*/>)',
                    r'\1<w:jc w:val="both" />', styles, count=1, flags=re.S)
    # table cells (pandoc's "Compact" style) in a smaller font
    styles = re.sub(r'(<w:style [^>]*w:styleId="Compact".*?</w:pPr>)',
                    rf"\1<w:rPr>{size(TABLE_SIZE)}</w:rPr>", styles, count=1, flags=re.S)
    styles = re.sub(r'(<w:tblStylePr w:type="firstRow">\s*<w:tcPr>)',
                    rf'\1<w:shd w:val="clear" w:color="auto" w:fill="{HEADER_FILL}"/>', styles)

    with zipfile.ZipFile(REFERENCE_DOCX, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = styles.encode("utf-8") if item.filename == "word/styles.xml" else src.read(item.filename)
            out.writestr(item, data)


def _with_border(cell, side):
    """Return a <w:tc> element with a single black `side` (top/bottom) cell border added."""
    border = f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    if "<w:tcBorders>" in cell:
        return cell.replace("<w:tcBorders>", "<w:tcBorders>" + border, 1)
    if "<w:tcPr>" in cell:
        return re.sub(r"(<w:tcPr>(?:<w:tcW [^>]*/>)?)", rf"\1<w:tcBorders>{border}</w:tcBorders>", cell, count=1)
    return cell.replace("<w:tc>", f"<w:tc><w:tcPr><w:tcBorders>{border}</w:tcBorders></w:tcPr>", 1)


def _full_width(table):
    """Stretch one <w:tbl> to the full text width (narrow tables otherwise sit half-empty)."""
    width = '<w:tblW w:w="5000" w:type="pct"/>'
    if "<w:tblW " in table:
        return re.sub(r"<w:tblW [^>]*/>", width, table, count=1)
    return re.sub(r"(<w:tblPr>)", rf"\1{width}", table, count=1)


def add_summary_rules(docx_path):
    """Draw a line above the "All datasets" row pair and below the last row of every table."""
    with zipfile.ZipFile(docx_path) as src:
        items = [(item, src.read(item.filename)) for item in src.infolist()]
    patched = 0
    for i, (item, data) in enumerate(items):
        if item.filename != "word/document.xml":
            continue
        doc = data.decode("utf-8")

        def patch_table(table_match):
            nonlocal patched
            table = _full_width(table_match.group(0))
            rows = list(re.finditer(r"<w:tr[ >].*?</w:tr>", table, re.S))
            if len(rows) < 3 or "All datasets" not in rows[-2].group(0):
                return table
            new_rows = {}
            for row, side in ((rows[-2], "top"), (rows[-1], "bottom")):
                new_rows[row.start()] = (row.end(), re.sub(r"<w:tc>.*?</w:tc>",
                                                           lambda c: _with_border(c.group(0), side),
                                                           row.group(0), flags=re.S))
            for start in sorted(new_rows, reverse=True):
                end, text = new_rows[start]
                table = table[:start] + text + table[end:]
            patched += 1
            return table

        doc = re.sub(r"<w:tbl>.*?</w:tbl>", patch_table, doc, flags=re.S)
        items[i] = (item, doc.encode("utf-8"))
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as out:
        for item, data in items:
            out.writestr(item, data)
    return patched


def main():
    with open(TEXT_PATH, encoding="utf-8") as f:
        blocks = json.load(f)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(render(blocks))

    build_reference_docx()
    subprocess.run(["pandoc", MD_PATH, "-f", "markdown", "-t", "docx",
                    "--reference-doc", REFERENCE_DOCX, "-o", DOCX_PATH], check=True)
    add_summary_rules(DOCX_PATH)
    n_tables = sum(bool(re.search(r"\(Table \d+\)", b["title"])) for b in blocks)
    print(f"Saved {MD_PATH} and {DOCX_PATH}: {len(blocks)} blocks, {n_tables} tables")


if __name__ == "__main__":
    main()
