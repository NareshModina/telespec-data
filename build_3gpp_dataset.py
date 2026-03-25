"""
build_3gpp_dataset.py
=====================
Converts the TSpec-LLM markdown corpus (rasoul-nikbakht/TSpec-LLM)
into a HuggingFace Arrow dataset compatible with the TeleSpec-Data schema.

Directory structure expected:
    TSpec-LLM/
      3GPP-clean/
        Rel-8/
          {series}_series/
            {number}-{version}.md
        Rel-9/ ... Rel-19/

Output schema (TeleSpec-Data compatible):
    {
        "id":       "3gpp_43064_Rel-14_e60",
        "category": "3gpp-standard",
        "content":  "3GPP TS 43.064 Release 14 \n Foreword \n ...",
        "metadata": '{"release": "14", "series": "43", "doc_number": "43064",
                      "version": "e60", "filename": "43064-e60.md"}'
    }

Run:
    pip install datasets tqdm

    python build_3gpp_dataset.py \
        --data-dir  ./TSpec-LLM/3GPP-clean \
        --output    ./3gpp-dataset \
        --min-tokens 50

    # Limit to specific releases
    python build_3gpp_dataset.py \
        --data-dir  ./TSpec-LLM/3GPP-clean \
        --output    ./3gpp-dataset \
        --releases  Rel-15 Rel-16 Rel-17 Rel-18 Rel-19
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict
from statistics import mean, median

from datasets import Dataset
from tqdm import tqdm


# ===========================================================================
# Constants
# ===========================================================================

RANDOM_SEED = 42

# Markdown artifacts from docx→md conversion to clean up
MD_ARTIFACT_RE = re.compile(
    r'\^([^\^]+)\^'          # superscripts: 3^rd^ → 3rd
    r'|\\\\'                 # double backslash
    r'|\\([\'\"(){}\[\]])'  # escaped punctuation
)

# TOC anchor links: [12](#section-name) — strip entire lines that are only these
TOC_LINE_RE = re.compile(r'^\[[\d]+\]\(#[^\)]+\)\s*$')

# Boilerplate patterns to strip
BOILERPLATE_RE = re.compile(
    r'3GPP TSG|CHANGE\s+REQUEST|CR\s+page|draft\s+CR|'
    r'^\s*\[\s*\]\s*$|'           # empty checkbox lines
    r'^\s*\\\s*$',                # lone backslash lines
    re.IGNORECASE
)

# Sections to drop entirely
STRIP_SECTIONS = {
    "foreword", "references", "abbreviations",
    "document history", "change history", "change request history",
    "intellectual property rights", "ipr",
}


# ===========================================================================
# Helpers
# ===========================================================================

def approx_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


def parse_filename(filename: str) -> dict:
    """
    Parse metadata from filename like '43064-e60.md'.
    Returns {doc_number, version, series}.
    """
    stem = filename.replace(".md", "")
    parts = stem.split("-")
    doc_number = parts[0] if parts else stem
    version    = parts[1] if len(parts) > 1 else ""
    series     = doc_number[:2] if len(doc_number) >= 2 else ""
    return {
        "doc_number": doc_number,
        "version":    version,
        "series":     series,
        "filename":   filename,
    }


def parse_release(rel_dir: str) -> str:
    """'Rel-14' → '14'"""
    return rel_dir.replace("Rel-", "").strip()


def clean_markdown(text: str) -> str:
    """Remove markdown artifacts from docx conversion."""
    # Remove superscript markers: 3^rd^ → 3rd
    text = re.sub(r'\^([^\^]*)\^', r'\1', text)
    # Remove escaped punctuation backslashes
    text = re.sub(r'\\([\'\"(){}\[\]\-\*\_\#\`\~\|\>])', r'\1', text)
    # Remove lone backslashes at end of line
    text = re.sub(r'\\\s*$', '', text, flags=re.MULTILINE)
    return text


def find_content_start(lines: list) -> int:
    """
    Find the line index where real content starts.
    Skips the TOC (anchor link lines) and finds 'Foreword' or first heading.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Foreword heading in two forms: "Foreword\n=======" or "# Foreword"
        if re.match(r'^Foreword\s*$', stripped, re.IGNORECASE):
            return i
        if re.match(r'^#+\s+Foreword', stripped, re.IGNORECASE):
            return i
        # First numbered section heading as fallback: "# 1 Scope" or "1 Scope\n====="
        if re.match(r'^#+\s+\d+[\s\.]', stripped):
            return i
        if re.match(r'^1\s+\w', stripped) and i > 50:
            return i
    return 0


def is_strip_section(heading: str) -> bool:
    h = heading.strip().lower().rstrip(":")
    if h in STRIP_SECTIONS:
        return True
    return any(kw in h for kw in STRIP_SECTIONS)


def extract_content(text: str, meta: dict) -> str:
    """
    Extract clean content from a markdown file.
    - Strips TOC
    - Strips boilerplate sections
    - Cleans markdown artifacts
    - Reassembles as single string with ' \n ' separators (TeleSpec-Data format)
    """
    lines = text.splitlines()

    # Find where real content starts
    start = find_content_start(lines)
    lines = lines[start:]

    # Remove pure TOC anchor lines
    lines = [l for l in lines if not TOC_LINE_RE.match(l)]

    # Remove boilerplate lines
    lines = [l for l in lines if not BOILERPLATE_RE.search(l)]

    # Split into sections and drop unwanted ones
    sections = []
    current_heading = None
    current_lines   = []
    skip_section    = False

    for line in lines:
        stripped = line.strip()

        # Detect heading
        m = re.match(r'^(#{1,4})\s+(.+)$', stripped)
        # Also detect setext-style headings (underline with === or ---)
        is_setext = False
        if not m and current_lines:
            last = current_lines[-1].strip() if current_lines else ""
            if re.match(r'^[=]{2,}$|^[-]{2,}$', stripped) and last:
                m_text = last
                current_lines.pop()
                m = re.match(r'^(.+)$', m_text)
                is_setext = True

        if m:
            # Save previous section
            if current_heading is not None and not skip_section:
                body = "\n".join(current_lines).strip()
                if len(body.split()) >= 5:
                    sections.append(f"{current_heading} \n {body}")

            heading_text = m.group(2) if not is_setext else m.group(1)
            current_heading = heading_text.strip()
            current_lines   = []
            skip_section    = is_strip_section(current_heading)
        else:
            current_lines.append(line)

    # Save last section
    if current_heading is not None and not skip_section:
        body = "\n".join(current_lines).strip()
        if len(body.split()) >= 5:
            sections.append(f"{current_heading} \n {body}")

    if not sections:
        return ""

    # Build document header
    series     = meta.get("series", "")
    doc_number = meta.get("doc_number", "")
    version    = meta.get("version", "")
    release    = meta.get("release", "")
    header     = f"3GPP TS {series}.{doc_number[len(series):]} Release {release} V{version}"

    full_text = header + " \n " + " \n ".join(sections)

    # Normalise double newlines to match TeleSpec-Data separator format
    full_text = full_text.replace("\n\n", " \n ")

    # Normalise non-breaking spaces
    full_text = full_text.replace("\xa0", " ")

    # Clean markdown artifacts
    full_text = clean_markdown(full_text)

    return full_text.strip()


# ===========================================================================
# Corpus walker
# ===========================================================================

def find_all_md_files(data_dir: str, releases: list = None) -> list:
    """
    Walk 3GPP-clean/ and return list of dicts with path and parsed metadata.
    """
    records = []
    for rel_dir in sorted(os.listdir(data_dir)):
        if not rel_dir.startswith("Rel-"):
            continue
        if releases and rel_dir not in releases:
            continue

        release    = parse_release(rel_dir)
        rel_path   = os.path.join(data_dir, rel_dir)

        for series_dir in sorted(os.listdir(rel_path)):
            series_path = os.path.join(rel_path, series_dir)
            if not os.path.isdir(series_path):
                continue

            for filename in sorted(os.listdir(series_path)):
                if not filename.endswith(".md"):
                    continue

                meta = parse_filename(filename)
                meta["release"]    = release
                meta["series_dir"] = series_dir
                meta["filepath"]   = os.path.join(series_path, filename)
                records.append(meta)

    return records


# ===========================================================================
# Main build
# ===========================================================================

def build(data_dir: str,
          output_dir: str,
          releases: list = None,
          min_tokens: int = 50,
          max_docs: int = None):

    random.seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  3GPP Dataset Builder (TSpec-LLM source)")
    print(f"{'='*60}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Output    : {output_dir}")
    print(f"  Releases  : {releases or 'all'}")
    print(f"  Min tokens: {min_tokens}")
    print()

    # Find all files
    all_files = find_all_md_files(data_dir, releases)
    if max_docs:
        all_files = all_files[:max_docs]
    print(f"  Files found: {len(all_files):,}")

    # Process
    records = []
    stats = {
        "total_files":    len(all_files),
        "skipped_empty":  0,
        "skipped_short":  0,
        "processed":      0,
        "by_release":     defaultdict(int),
        "by_series":      defaultdict(int),
        "doc_tokens":     [],
    }

    for meta in tqdm(all_files, desc="  processing"):
        filepath = meta["filepath"]
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except Exception:
            stats["skipped_empty"] += 1
            continue

        if len(raw.strip()) < 100:
            stats["skipped_empty"] += 1
            continue

        content = extract_content(raw, meta)

        if not content:
            stats["skipped_empty"] += 1
            continue

        tokens = approx_tokens(content)
        if tokens < min_tokens:
            stats["skipped_short"] += 1
            continue

        doc_id = (f"3gpp_{meta['doc_number']}_"
                  f"Rel-{meta['release']}_"
                  f"{meta['version']}")

        records.append({
            "id":       doc_id,
            "category": "3gpp-standard",
            "content":  content,
            "metadata": json.dumps({
                "doc_number":  meta["doc_number"],
                "series":      meta["series"],
                "release":     meta["release"],
                "version":     meta["version"],
                "filename":    meta["filename"],
                "series_dir":  meta["series_dir"],
            }, ensure_ascii=False),
        })

        stats["processed"]                       += 1
        stats["by_release"][meta["release"]]     += 1
        stats["by_series"][meta["series_dir"]]   += 1
        stats["doc_tokens"].append(tokens)

    # Summary
    tl = stats["doc_tokens"]
    print(f"\n{'='*60}")
    print(f"  BUILD RESULTS")
    print(f"{'='*60}")
    print(f"  Total files   : {stats['total_files']:,}")
    print(f"  Processed     : {stats['processed']:,}")
    print(f"  Skipped empty : {stats['skipped_empty']:,}")
    print(f"  Skipped short : {stats['skipped_short']:,}")
    if tl:
        print(f"\n  Doc token stats:")
        print(f"    min={min(tl):,}  max={max(tl):,}  "
              f"mean={round(mean(tl)):,}  median={median(tl):,}")
    print(f"\n  By release:")
    for rel, cnt in sorted(stats["by_release"].items(), key=lambda x: int(x[0])):
        print(f"    Rel-{rel:<4} : {cnt:,}")

    if not records:
        print("\n  [ERROR] No records produced.")
        return

    # Shuffle and save
    random.shuffle(records)
    os.makedirs(output_dir, exist_ok=True)
    Dataset.from_list(records).save_to_disk(os.path.join(output_dir, "train"))

    # Stats
    stats_out = {
        **{k: v for k, v in stats.items() if k != "doc_tokens"},
        "by_release":  dict(stats["by_release"]),
        "by_series":   dict(stats["by_series"]),
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

    print(f"\n  Dataset → {os.path.join(output_dir, 'train')}")
    print(f"  Stats   → {stats_path}")
    print(f"\n{'='*60}\n")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 3GPP dataset from TSpec-LLM markdown corpus"
    )
    parser.add_argument("--data-dir",   required=True,
                        help="Path to TSpec-LLM/3GPP-clean directory")
    parser.add_argument("--output",     required=True,
                        help="Output directory for HuggingFace Arrow dataset")
    parser.add_argument("--releases",   nargs="+", default=None,
                        help="Limit to specific releases: --releases Rel-17 Rel-18 Rel-19")
    parser.add_argument("--min-tokens", type=int, default=50,
                        help="Minimum tokens per document (default: 50)")
    parser.add_argument("--max-docs",   type=int, default=None,
                        help="Cap total docs for smoke testing")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        exit(1)

    build(
        data_dir=args.data_dir,
        output_dir=args.output,
        releases=args.releases,
        min_tokens=args.min_tokens,
        max_docs=args.max_docs,
    )