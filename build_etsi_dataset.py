"""
build_etsi_dataset_v3.py
========================
Builds a HuggingFace Arrow dataset from the ETSI PDF corpus.
Output schema is Tele-Data compatible: id, category, content, metadata.

Run:
    pip install pymupdf pandas tqdm datasets

    # Quick smoke test (one WG, first 50 docs)
    python build_etsi_dataset_v3.py \
        --data-dir  ./data \
        --csv       ./ETSICatalog.csv \
        --output    ./etsi-dataset-v3 \
        --groups    EG \
        --max-docs  50

    # Single WG test
    python build_etsi_dataset_v3.py \
        --data-dir  ./data \
        --csv       ./ETSICatalog.csv \
        --output    ./etsi-dataset-v3 \
        --groups    ETS ETR GTS

    # Full build
    python build_etsi_dataset_v3.py \
        --data-dir  ./data \
        --csv       ./ETSICatalog.csv \
        --output    ./etsi-dataset-v3 \
        --skip-log  ./etsi_skipped_ids_v3.txt

Outputs:
    ./etsi-dataset-v3/train/     HuggingFace Arrow dataset
    ./etsi-dataset-v3/stats.json build statistics
    ./etsi_skipped_ids_v3.txt    doc IDs still skipped (if --skip-log given)
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict
from statistics import mean, median

import fitz  # pymupdf
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

# Suppress MuPDF xref / format error stderr noise
fitz.TOOLS.mupdf_display_errors(False)


# ===========================================================================
# Constants
# ===========================================================================

MIN_YEAR        = 2000
RANDOM_SEED     = 42

# Window fallback parameters
WINDOW_TOKENS   = 512
WINDOW_OVERLAP  = 64


# ===========================================================================
# Heading regex patterns
# ===========================================================================

# Primary: inline "4.1 Title case heading"
#   - Max 2-digit section numbers → rejects "650 Route", "146 MHz", "2.0.0"
#   - Second char lowercase → rejects "MHz", "TSG", "UE" (pure acronyms)
#   - Body 3–60 chars
CLAUSE_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,3})"
    r"\s{1,6}"
    r"([A-Z][a-z][a-zA-Z0-9\s\-,/()']{2,60})$"
)

# Acronym-started titles: "4.1 UE Requirements", "6.2 5G NR Architecture"
CLAUSE_CAPS_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,3})"
    r"\s{1,6}"
    r"([A-Z]{2,8}[\s\-][A-Z][a-zA-Z\s\-,/()']{1,55})$"
)

# All-caps headings: "4.1 GENERAL REQUIREMENTS" (some older EN/ES docs)
ALL_CAPS_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,3})"
    r"\s{1,6}"
    r"([A-Z][A-Z\s\-/]{3,60})$"
)

# Annex headings: "Annex A", "Annex A.1 (normative)", "Annex B: Security"
ANNEX_RE = re.compile(
    r"^(Annex\s+[A-Z](?:\.\d{1,2})?)"
    r"(?:\s+\((?:normative|informative)\))?"
    r"(?:\s*[:(]\s*([A-Z][a-zA-Z\s\-,]{2,60}))?$",
    re.IGNORECASE
)

# Letter-prefixed clauses: "A.1 Scope", "B.2.3 Definitions"
LETTER_RE = re.compile(
    r"^([A-Z]\.\d{1,2}(?:\.\d{1,2}){0,2})"
    r"\s{1,6}"
    r"([A-Z][a-zA-Z\s\-,]{2,60})$"
)

# Old-format: bare clause number on its own line (e.g. "6" or "6.1" or "6.1.2")
OLD_NUM_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}$")

# Old-format: title on the line immediately after the bare number
# Must start with capital, be at least 3 chars, no trailing dots (TOC lines)
OLD_TITLE_RE = re.compile(r"^[A-Z][a-zA-Z\s\-,/()']{2,80}$")

# TOC dotted lines: "Scope .............................7"  — excluded from old-format
TOC_LINE_RE = re.compile(r"\.{5,}")


# ===========================================================================
# Boilerplate stripping
# ===========================================================================

BOILERPLATE_PATTERNS = [
    r"^ETSI$",
    r"^ETSI\s+(EG|EN|ES|TR|TS|ETS|ETR|SR|GS|GR|GTS|TBR|NET|I-ETS|TCRTR)\s+\d",
    r"650 Route des Lucioles",
    r"F-06921 Sophia Antipolis",
    r"Sophia Antipolis Cedex",
    r"Sophia Antipolis - Valbonne",
    r"Tel\.:\s+\+33",
    r"Fax:\s+\+33",
    r"Siret N°",
    r"Association à but non lucratif",
    r"Sous-Préfecture de Grasse",
    r"secretariat@etsi",
    r"editor@etsi",
    r"http://www\.etsi",
    r"http://portal\.etsi",
    r"https://portal\.etsi",
    r"www\.etsi\.org",
    r"Individual copies of",
    r"can be downloaded from",
    r"If you find err",
    r"send your comment",
    r"Important notice",
    r"Copyright Notification",
    r"No part may be reproduced",
    r"The copyright and the foregoing",
    r"European Telecommunications Standards Institute",
    r"ETSI Secretariat",
    r"All rights reserved",
    r"© European Telecommunications",
    r"Postal address",
    r"Office address",
    r"X\.400.*etsi",
    r"ISBN\s+\d",
    r"Dépôt légal",
    r"^\d+$",       # lone page numbers
    r"^-{3,}$",     # horizontal rules
    r"^_{3,}$",     # underscores used as rules
    r"^={3,}$",     # equals rules
    r"^Page\s+\d",  # "Page 3" headers in old docs
    r"Blank page",
]

BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in BOILERPLATE_PATTERNS),
    re.IGNORECASE
)

STRIP_SECTIONS = {
    "history",
    "foreword",
    "intellectual property rights",
    "ipr",
    "references",
    "normative references",
    "document history",
    "modal verbs terminology",
    "abbreviations",       # keep definitions but strip bare abbrev lists
}

# Year/version/deliverable extraction
YEAR_RE    = re.compile(r'\((\d{4})-\d{2}\)')
VERSION_RE = re.compile(r'V(\d+\.\d+\.\d+)')
DELIV_RE   = re.compile(
    r'\b(EG|EN|ES|TR|TS|ETS|ETR|SR|GS|GR|GTS|TBR)\s+(\d[\d\s]+\d)'
)


# ===========================================================================
# Utility functions
# ===========================================================================

def approx_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


def extract_year(text: str):
    m = YEAR_RE.search(text[:2000])
    return int(m.group(1)) if m else None


def extract_version(text: str):
    m = VERSION_RE.search(text[:2000])
    return m.group(1) if m else None


def extract_deliverable(text: str):
    m = DELIV_RE.search(text[:500])
    return (m.group(1), m.group(2).strip()) if m else (None, None)


def strip_boilerplate(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not (line.strip() and BOILERPLATE_RE.search(line.strip()))
    )


def find_content_start(lines: list) -> int:
    """
    Return index of first top-level clause heading.
    Tries inline format first ("1 Scope"), then old format (bare "1" line).
    """
    # Inline: "1 Scope"
    for i, line in enumerate(lines):
        if re.match(r'^1\s{1,6}[A-Z][a-zA-Z]', line.strip()):
            return i
    # Old format: bare "1" on one line, title on next
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "1" and OLD_TITLE_RE.match(lines[i + 1].strip()):
            return i
    return 0


def is_strip_section(title: str) -> bool:
    return title.strip().lower() in STRIP_SECTIONS


def is_heading_inline(line: str):
    """
    Test a single line against all inline heading patterns.
    Returns (True, clause_num, title) or (False, None, None).
    """
    s = line.strip()
    if not s:
        return False, None, None

    for pat in [CLAUSE_RE, CLAUSE_CAPS_RE, ALL_CAPS_RE]:
        m = pat.match(s)
        if m:
            return True, m.group(1), m.group(2).strip()

    m = ANNEX_RE.match(s)
    if m:
        title = m.group(2).strip() if m.group(2) else s
        return True, m.group(1), title

    m = LETTER_RE.match(s)
    if m:
        return True, m.group(1), m.group(2).strip()

    return False, None, None


# ===========================================================================
# Stage 1 — inline clause splitter (modern docs)
# ===========================================================================

def split_by_clauses(text: str) -> list:
    """
    Split on inline clause headings: "4.1 Title case heading"
    Returns list of {"clause_num", "title", "body"} dicts.
    """
    lines = text.splitlines()
    start = find_content_start(lines)
    lines = lines[start:]

    clauses = []
    current = None

    for line in lines:
        found, clause_num, title = is_heading_inline(line)
        if found:
            if current is not None:
                clauses.append(current)
            current = {"clause_num": clause_num, "title": title, "body_lines": []}
        else:
            if current is not None:
                current["body_lines"].append(line)

    if current is not None:
        clauses.append(current)

    return _finalise_clauses(clauses)


# ===========================================================================
# Stage 2 — old-format two-line splitter (pre-2005 ETS/ETR/GTS)
# ===========================================================================

def split_by_clauses_old(text: str) -> list:
    """
    Handles old ETSI PDF format where clause number and title are on
    separate consecutive lines:

        1
        Scope
        2
        Normative references
        6.1
        Protocol stack for Qx interface

    The TOC at the start of the doc has the same format but with trailing
    dots ("Scope .....7"), which are excluded by TOC_LINE_RE.
    """
    lines = [l.strip() for l in text.splitlines()]

    clauses = []
    current = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # Candidate: bare clause number AND next line is a plain title
        if (OLD_NUM_RE.match(line)
                and i + 1 < len(lines)
                and OLD_TITLE_RE.match(lines[i + 1])
                and not TOC_LINE_RE.search(lines[i + 1])):

            if current is not None:
                clauses.append(current)
            current = {
                "clause_num": line,
                "title":      lines[i + 1],
                "body_lines": [],
            }
            i += 2
            continue

        if current is not None:
            current["body_lines"].append(lines[i])
        i += 1

    if current is not None:
        clauses.append(current)

    return _finalise_clauses(clauses)


# ===========================================================================
# Stage 3 — annex-only splitter
# ===========================================================================

def split_by_annexes(text: str) -> list:
    """
    For docs where the only detectable structure is Annex headings.
    Splits on "Annex A (normative)", "Annex B:", etc.
    """
    lines = text.splitlines()
    clauses = []
    current = None

    for line in lines:
        s = line.strip()
        m = ANNEX_RE.match(s)
        if m:
            if current is not None:
                clauses.append(current)
            title = m.group(2).strip() if m.group(2) else s
            current = {
                "clause_num": m.group(1),
                "title":      title,
                "body_lines": [],
            }
        else:
            if current is not None:
                current["body_lines"].append(line)

    if current is not None:
        clauses.append(current)

    return _finalise_clauses(clauses)


# ===========================================================================
# Stage 4 — window fallback (genuinely unstructured)
# ===========================================================================

def split_by_window(text: str,
                    max_tokens: int = WINDOW_TOKENS,
                    overlap: int = WINDOW_OVERLAP) -> list:
    """
    Fixed-size token window with overlap.
    Used only for docs with no detectable structure whatsoever.
    """
    words = text.split()
    if not words:
        return []
    step   = max(1, max_tokens - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + max_tokens]
        chunks.append({
            "clause_num": f"W{len(chunks) + 1}",
            "title":      "text",
            "body":       " ".join(chunk_words),
        })
        i += step
    return chunks


# ===========================================================================
# Shared finaliser
# ===========================================================================

def _finalise_clauses(clauses: list) -> list:
    """Convert body_lines → body string, apply filters."""
    result = []
    for c in clauses:
        if is_strip_section(c["title"]):
            continue
        body = "\n".join(c["body_lines"]).strip()
        if len(body.split()) < 10:
            continue
        result.append({
            "clause_num": c["clause_num"],
            "title":      c["title"],
            "body":       body,
        })
    return result


# ===========================================================================
# Four-stage cascade
# ===========================================================================

def split_document(text: str) -> tuple:
    """
    Run the four-stage splitting cascade.
    Returns (clauses, split_method).
    split_method: "clause" | "clause_old" | "annex" | "window"
    """
    clauses = split_by_clauses(text)
    if clauses:
        return clauses, "clause"

    clauses = split_by_clauses_old(text)
    if clauses:
        return clauses, "clause_old"

    clauses = split_by_annexes(text)
    if clauses:
        return clauses, "annex"

    clauses = split_by_window(text)
    return clauses, "window"


def build_content(clauses: list, meta: dict) -> str:
    """
    Reassemble clause chunks into a single document string matching
    the Tele-Data content format:
      - Document header line
      - Full body as continuous text with \n between sections
    """
    header = (
        f"ETSI {meta.get('deliverable_type', '')} "
        f"{meta.get('deliverable_number', '')} "
        f"V{meta.get('version', '')} ({meta.get('year', '')})"
    ).strip()
    doc_title = meta.get("title", "")
    if doc_title:
        header += f" \n {doc_title}"

    body_parts = []
    for clause in clauses:
        section = f"{clause['clause_num']} {clause['title']} \n {clause['body']}"
        body_parts.append(section)

    return header + " \n " + " \n ".join(body_parts)


# ===========================================================================
# PDF pipeline
# ===========================================================================

def process_pdf(pdf_path: str, csv_meta: dict, txt_meta: dict):
    """
    Full pipeline for one PDF.
    Returns (record_dict, None) on success,
            (None, skip_reason) on skip.

    Emits ONE record per document (full text), matching Tele-Data schema.
    Chunking into training sequences is handled by the tokenizer/packer
    during training, not here.
    """
    try:
        doc      = fitz.open(pdf_path)
        raw_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
    except Exception:
        return None, "extract_error"

    if len(raw_text.strip()) < 200:
        return None, "no_text"

    year = extract_year(raw_text)
    if year is None or year < MIN_YEAR:
        return None, "pre_2000"

    version              = extract_version(raw_text)
    deliv_type, deliv_no = extract_deliverable(raw_text)

    meta = {
        "year":               year,
        "version":            version or "",
        "deliverable_type":   deliv_type or "",
        "deliverable_number": deliv_no or "",
        "title":              csv_meta.get("title",
                                            txt_meta.get("Title", "")),
        "status":             csv_meta.get("Status",
                                            txt_meta.get("Status", "")),
        "technical_body":     csv_meta.get("Technical body",
                                            txt_meta.get("Technical Body", "")),
        "keywords":           csv_meta.get("Keywords",
                                            txt_meta.get("Keywords", "")),
        "scope":              csv_meta.get("Scope",
                                            txt_meta.get("Scope", "")),
        "working_group":      csv_meta.get("working_group", ""),
    }

    clean_text = strip_boilerplate(raw_text)

    # Four-stage cascade — used for boilerplate removal and content detection,
    # but we reassemble into one document rather than splitting into chunks
    clauses, split_method = split_document(clean_text)

    if not clauses:
        return None, "no_chunks"

    # Reassemble full document content as one string (Tele-Data format)
    content = build_content(clauses, meta)

    if approx_tokens(content) < 50:
        return None, "no_chunks"

    return ({
        "content":      content,
        "split_method": split_method,
        "year":         year,
        "tokens":       approx_tokens(content),
        "meta":         meta,
    }, None)


# ===========================================================================
# Metadata loaders
# ===========================================================================

def load_csv(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV not found: {csv_path} — using .txt metadata only")
        return {}
    try:
        df = pd.read_csv(csv_path, delimiter=";",
                         skiprows=1, on_bad_lines="skip")
        lookup = {}
        for _, row in df.iterrows():
            doc_id = str(row.get("id", "")).strip()
            if doc_id:
                lookup[doc_id] = row.to_dict()
        print(f"  CSV loaded: {len(lookup):,} entries")
        return lookup
    except Exception as e:
        print(f"  [WARN] CSV load error: {e}")
        return {}


def parse_metadata_txt(path: str) -> dict:
    meta = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
    except Exception:
        pass
    return meta


# ===========================================================================
# Corpus walker
# ===========================================================================

def find_all_pairs(data_dir: str, allowed_groups=None) -> list:
    pairs = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        pdfs  = {f[:-4]: os.path.join(root, f)
                 for f in files if f.lower().endswith(".pdf")}
        metas = {f.replace("_metadata.txt", ""): os.path.join(root, f)
                 for f in files if f.endswith("_metadata.txt")}

        for doc_id, pdf_path in pdfs.items():
            rel   = os.path.relpath(root, data_dir)
            parts = rel.replace("\\", "/").split("/")
            wg    = parts[0] if parts else "unknown"

            if allowed_groups and wg not in allowed_groups:
                continue

            pairs.append({
                "id":            doc_id,
                "pdf_path":      pdf_path,
                "meta_path":     metas.get(doc_id),
                "working_group": wg,
            })
    return pairs


# ===========================================================================
# Main build
# ===========================================================================

def build(data_dir: str,
          csv_path: str,
          output_dir: str,
          allowed_groups=None,
          max_docs=None,
          skip_log_path=None):

    random.seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  ETSI Dataset Builder v3")
    print(f"{'='*60}")
    print(f"  Data dir       : {data_dir}")
    print(f"  CSV            : {csv_path}")
    print(f"  Output         : {output_dir}")
    print(f"  Year filter    : >= {MIN_YEAR}")
    print(f"  Working groups : {allowed_groups or 'all'}")
    print(f"  Split cascade  : clause → clause_old → annex → window")
    print(f"  Window params  : {WINDOW_TOKENS} tokens / {WINDOW_OVERLAP} overlap")
    print()

    csv_lookup = load_csv(csv_path)
    pairs      = find_all_pairs(data_dir, allowed_groups)
    if max_docs:
        pairs = pairs[:max_docs]
    print(f"  PDFs to process: {len(pairs):,}\n")

    all_records = []
    skipped_ids = []

    stats = {
        "total_pdfs":          len(pairs),
        "skipped_pre2000":     0,
        "skipped_no_text":     0,
        "skipped_extract_err": 0,
        "skipped_no_chunks":   0,
        "processed":           0,
        "split_clause":        0,
        "split_clause_old":    0,
        "split_annex":         0,
        "split_window":        0,
        "by_working_group":    defaultdict(int),
        "by_year":             defaultdict(int),
        "doc_tokens":          [],   # tokens per document (not per chunk)
    }

    for pair in tqdm(pairs, desc="  processing"):
        doc_id = pair["id"]
        wg     = pair["working_group"]

        csv_meta                  = csv_lookup.get(doc_id, {})
        csv_meta["working_group"] = wg
        txt_meta                  = (parse_metadata_txt(pair["meta_path"])
                                     if pair["meta_path"] else {})

        record, skip_reason = process_pdf(pair["pdf_path"], csv_meta, txt_meta)

        if record is None:
            skipped_ids.append(doc_id)
            if skip_reason == "pre_2000":
                stats["skipped_pre2000"] += 1
            elif skip_reason == "no_text":
                stats["skipped_no_text"] += 1
            elif skip_reason == "extract_error":
                stats["skipped_extract_err"] += 1
            else:
                stats["skipped_no_chunks"] += 1
            continue

        content      = record["content"]
        split_method = record["split_method"]
        year         = record["year"]
        meta         = record["meta"]

        all_records.append({
            "id":       f"etsi_{doc_id}",
            "category": "etsi-standard",
            "content":  content,
            "metadata": json.dumps({
                "doc_id":             doc_id,
                "working_group":      wg,
                "year":               year,
                "split_method":       split_method,
                "title":              meta.get("title", ""),
                "technical_body":     meta.get("technical_body", ""),
                "keywords":           meta.get("keywords", ""),
                "deliverable_type":   meta.get("deliverable_type", ""),
                "deliverable_number": meta.get("deliverable_number", ""),
                "version":            meta.get("version", ""),
            }, ensure_ascii=False),
        })

        stats["doc_tokens"].append(record["tokens"])
        stats["by_working_group"][wg]  += 1
        stats["by_year"][str(year)]    += 1
        stats[f"split_{split_method}"] += 1
        stats["processed"]             += 1

    # ── Summary ──────────────────────────────────────────────────────────
    total = stats["total_pdfs"]
    proc  = stats["processed"]
    print(f"\n{'='*60}")
    print(f"  BUILD RESULTS")
    print(f"{'='*60}")
    print(f"  Total PDFs        : {total:,}")
    print(f"  Processed         : {proc:,}  ({100*proc/total:.1f}%)")
    print(f"  Skipped pre-2000  : {stats['skipped_pre2000']:,}")
    print(f"  Skipped no-text   : {stats['skipped_no_text']:,}")
    print(f"  Skipped extract err: {stats['skipped_extract_err']:,}")
    print(f"  Skipped no-chunks : {stats['skipped_no_chunks']:,}")
    print(f"\n  Total documents   : {proc:,}  (1 record per doc)")
    print(f"    clause          : {stats['split_clause']:,}")
    print(f"    clause_old      : {stats['split_clause_old']:,}")
    print(f"    annex           : {stats['split_annex']:,}")
    print(f"    window          : {stats['split_window']:,}")

    if not all_records:
        print("\n  [ERROR] No records produced. Check --data-dir and filters.")
        return

    # ── Save skip log ─────────────────────────────────────────────────────
    if skip_log_path and skipped_ids:
        with open(skip_log_path, "w") as f:
            f.write("\n".join(skipped_ids))
        print(f"\n  Skip log → {skip_log_path}  ({len(skipped_ids):,} IDs)")

    # ── Shuffle and save as single train dataset ──────────────────────────
    # No eval split: the pretraining eval set should be separately curated
    # (e.g. held-out documents, or a task-specific benchmark like Tele-Eval).
    # A random chunk slice from the same corpus is not a meaningful eval.
    random.shuffle(all_records)
    n_train = len(all_records)

    print(f"\n  Total records : {n_train:,}")

    # ── Save dataset ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    Dataset.from_list(all_records).save_to_disk(
        os.path.join(output_dir, "train"))

    # ── Save stats ────────────────────────────────────────────────────────
    tl = stats["doc_tokens"]
    stats_out = {
        **{k: v for k, v in stats.items() if k != "doc_tokens"},
        "by_working_group": dict(stats["by_working_group"]),
        "by_year":          dict(stats["by_year"]),
        "total_examples":   n_train,
        "doc_token_stats": {
            "min":    min(tl) if tl else 0,
            "max":    max(tl) if tl else 0,
            "mean":   round(mean(tl)) if tl else 0,
            "median": median(tl) if tl else 0,
        },
    }
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)

    if tl:
        print(f"\n  Document token stats (approx):")
        print(f"    min={min(tl):,}  max={max(tl):,}  "
              f"mean={round(mean(tl)):,}  median={median(tl):,}")

    print(f"\n  Dataset → {os.path.join(output_dir, 'train')}")
    print(f"  Stats   → {stats_path}")
    print(f"  Note    : No eval split. Use a curated benchmark (e.g. Tele-Eval)")
    print(f"            for downstream evaluation of the fine-tuned model.")
    print(f"\n{'='*60}\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build HuggingFace NLP dataset from ETSI PDF corpus (v3)"
    )
    parser.add_argument("--data-dir",  required=True,
                        help="Root folder containing working group subfolders")
    parser.add_argument("--csv",       required=True,
                        help="Path to ETSICatalog.csv")
    parser.add_argument("--output",    required=True,
                        help="Output directory for HuggingFace dataset")
    parser.add_argument("--groups",    nargs="+", default=None,
                        help="Limit to specific WGs: --groups EG TR TS")
    parser.add_argument("--max-docs",  type=int, default=None,
                        help="Cap total PDFs (for smoke testing)")
    parser.add_argument("--skip-log",  default=None,
                        help="Save still-skipped doc IDs for inspection")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        exit(1)

    build(
        data_dir=args.data_dir,
        csv_path=args.csv,
        output_dir=args.output,
        allowed_groups=args.groups,
        max_docs=args.max_docs,
        skip_log_path=args.skip_log,
    )