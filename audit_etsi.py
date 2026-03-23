"""
ETSI Corpus Audit Script
=========================
Audits the ETSI document corpus before building the NLP dataset.
Reads ETSICatalog.csv as the master metadata source, walks all 15
working group subfolders, and reports everything needed to design
the extraction pipeline.

Run:
    pip install pymupdf pandas tqdm

    python audit_etsi.py \
        --data-dir  ./data \
        --csv       ./ETSICatalog.csv \
        --output    ./etsi_audit_report.json \
        --samples   3

Outputs:
    - Console summary
    - etsi_audit_report.json   full machine-readable report
    - etsi_samples.txt         sample extracted text per working group
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean, median
from tqdm import tqdm

import pandas as pd
import fitz  # pymupdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


def extract_year_from_text(text: str) -> int | None:
    """
    Extract publication year from ETSI PDF header.
    Looks for patterns like (1997-06) or (2003-12) in the first 2000 chars.
    """
    snippet = text[:2000]
    m = re.search(r'\((\d{4})-\d{2}\)', snippet)
    if m:
        return int(m.group(1))
    return None


def extract_version_from_text(text: str) -> str | None:
    """Extract version string like V1.1.1 from first 2000 chars."""
    snippet = text[:2000]
    m = re.search(r'V(\d+\.\d+\.\d+)', snippet)
    if m:
        return m.group(1)
    return None


def extract_deliverable_type(text: str) -> str | None:
    """
    Extract ETSI deliverable type from header.
    e.g. 'EG', 'EN', 'ES', 'TR', 'TS', 'ETS'
    """
    snippet = text[:500]
    m = re.search(r'\b(EG|EN|ES|TR|TS|ETS|ETR|SR|GS|GR)\s+\d+', snippet)
    if m:
        return m.group(1)
    return None


def is_text_extractable(text: str) -> bool:
    """
    Returns True if meaningful text was extracted.
    Scanned PDFs return empty or near-empty strings.
    """
    return len(text.strip()) > 200


def extract_pdf_text(pdf_path: str) -> tuple[str, bool]:
    """
    Extract all text from PDF using pymupdf.
    Returns (text, success).
    Tables and images are skipped — text blocks only.
    """
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page in doc:
            # Extract text only — no images, no tables
            # "text" flag extracts plain text blocks only
            text = page.get_text("text")
            pages_text.append(text)
        doc.close()
        full_text = "\n".join(pages_text)
        return full_text, is_text_extractable(full_text)
    except Exception as e:
        return "", False


CLAUSE_PATTERNS = [
    (r"^\d+\.\d+\.\d+\.\d+\s+\S", "4-level (X.X.X.X)"),
    (r"^\d+\.\d+\.\d+\s+\S",       "3-level (X.X.X)"),
    (r"^\d+\.\d+\s+\S",            "2-level (X.X)"),
    (r"^\d+\s+\S",                 "1-level (X)"),
]

def count_clause_headings(text: str) -> dict:
    counts = defaultdict(int)
    for line in text.splitlines():
        line = line.strip()
        for pattern, label in CLAUSE_PATTERNS:
            if re.match(pattern, line):
                counts[label] += 1
                break
    return dict(counts)

def extract_clause_headings(text: str, n: int = 8) -> list:
    found = []
    for line in text.splitlines():
        line = line.strip()
        for pattern, _ in CLAUSE_PATTERNS:
            if re.match(pattern, line):
                found.append(line[:100])
                break
        if len(found) >= n:
            break
    return found


def parse_metadata_txt(path: str) -> dict:
    """Parse the {id}_metadata.txt sidecar file."""
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


# ---------------------------------------------------------------------------
# Walk corpus
# ---------------------------------------------------------------------------

def find_all_pairs(data_dir: str) -> list[dict]:
    """
    Walk all subfolders under data_dir and find PDF + metadata pairs.
    Returns list of dicts with keys: id, pdf_path, meta_path, working_group
    """
    pairs = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        pdfs = {f[:-4]: os.path.join(root, f)
                for f in files if f.lower().endswith(".pdf")}
        metas = {f.replace("_metadata.txt", ""): os.path.join(root, f)
                 for f in files if f.endswith("_metadata.txt")}

        for doc_id, pdf_path in pdfs.items():
            # Determine working group from folder path
            rel = os.path.relpath(root, data_dir)
            parts = rel.replace("\\", "/").split("/")
            working_group = parts[0] if parts else "unknown"

            pairs.append({
                "id":            doc_id,
                "pdf_path":      pdf_path,
                "meta_path":     metas.get(doc_id),
                "working_group": working_group,
                "has_metadata":  doc_id in metas,
            })
    return pairs


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit(data_dir: str, csv_path: str, output_path: str,
          n_samples: int = 3, max_docs: int = None):

    print(f"\n{'='*60}")
    print(f"  ETSI Corpus Audit")
    print(f"{'='*60}")
    print(f"  Data dir : {data_dir}")
    print(f"  CSV      : {csv_path}")
    print()

    # ── Load CSV ──────────────────────────────────────────────────────────
    print("  Loading ETSICatalog.csv...")
    try:
        df_csv = pd.read_csv(csv_path, delimiter=";",
                             skiprows=1, on_bad_lines="skip")
        print(f"  CSV rows          : {len(df_csv):,}")
        print(f"  CSV columns       : {list(df_csv.columns)}")
    except Exception as e:
        print(f"  [WARN] Could not load CSV: {e}")
        df_csv = None

    # ── Find all PDF/metadata pairs ───────────────────────────────────────
    print(f"\n  Scanning {data_dir} for PDF pairs...")
    pairs = find_all_pairs(data_dir)
    if max_docs:
        pairs = pairs[:max_docs]

    print(f"  Total PDFs found  : {len(pairs):,}")
    wg_counts = Counter(p["working_group"] for p in pairs)
    print(f"  Working groups    : {len(wg_counts)}")
    for wg, cnt in sorted(wg_counts.items(), key=lambda x: -x[1]):
        print(f"    {wg:<20} {cnt:,}")

    no_meta = sum(1 for p in pairs if not p["has_metadata"])
    print(f"\n  PDFs with metadata    : {len(pairs)-no_meta:,}")
    print(f"  PDFs missing metadata : {no_meta:,}")

    # ── Sample-based text extraction audit ────────────────────────────────
    # Sample up to 500 docs evenly across working groups for speed
    sample_size   = min(500, len(pairs))
    step          = max(1, len(pairs) // sample_size)
    sample_pairs  = pairs[::step][:sample_size]

    print(f"\n  Sampling {len(sample_pairs)} documents for text audit...")

    year_counts        = Counter()
    extractable_count  = 0
    not_extractable    = 0
    token_lens         = []
    clause_agg         = defaultdict(int)
    wg_samples         = defaultdict(list)    # working_group -> sample texts
    year_extractable   = defaultdict(int)
    year_total         = defaultdict(int)

    for pair in tqdm(sample_pairs, desc="  auditing"):
        text, ok = extract_pdf_text(pair["pdf_path"])

        year = extract_year_from_text(text) if ok else None
        if year:
            year_counts[year] += 1
            year_total[year]  += 1
            if ok:
                year_extractable[year] += 1
        else:
            year_total["unknown"] += 1

        if ok:
            extractable_count += 1
            tokens = approx_tokens(text)
            token_lens.append(tokens)

            clause_counts = count_clause_headings(text)
            for label, cnt in clause_counts.items():
                clause_agg[label] += cnt

            wg = pair["working_group"]
            if len(wg_samples[wg]) < n_samples:
                wg_samples[wg].append({
                    "id":       pair["id"],
                    "year":     year,
                    "version":  extract_version_from_text(text),
                    "type":     extract_deliverable_type(text),
                    "tokens":   tokens,
                    "headings": extract_clause_headings(text),
                    "text_snippet": text[:1500],
                })
        else:
            not_extractable += 1

    # ── Year distribution ─────────────────────────────────────────────────
    pre2000  = sum(v for k, v in year_counts.items()
                   if isinstance(k, int) and k < 2000)
    post2000 = sum(v for k, v in year_counts.items()
                   if isinstance(k, int) and k >= 2000)
    unknown_yr = year_total.get("unknown", 0)

    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AUDIT RESULTS  (sampled {len(sample_pairs)} docs)")
    print(f"{'='*60}")

    print(f"\n  TEXT EXTRACTABILITY")
    print(f"    Extractable   : {extractable_count:,} ({100*extractable_count/len(sample_pairs):.1f}%)")
    print(f"    Not extractable: {not_extractable:,} ({100*not_extractable/len(sample_pairs):.1f}%)")

    print(f"\n  YEAR DISTRIBUTION  (from PDF header)")
    print(f"    Pre-2000      : {pre2000:,}")
    print(f"    Post-2000     : {post2000:,}")
    print(f"    Unknown year  : {unknown_yr:,}")
    print(f"    Year range    : {min((k for k in year_counts if isinstance(k,int)), default='?')} "
          f"– {max((k for k in year_counts if isinstance(k,int)), default='?')}")

    print(f"\n  YEAR BREAKDOWN (post-2000):")
    for yr in sorted(k for k in year_counts if isinstance(k, int) and k >= 2000):
        bar = "█" * min(30, year_counts[yr])
        print(f"    {yr}  {bar}  {year_counts[yr]}")

    if token_lens:
        print(f"\n  TOKEN LENGTH  (extractable docs)")
        print(f"    min    : {min(token_lens):>10,}")
        print(f"    max    : {max(token_lens):>10,}")
        print(f"    mean   : {mean(token_lens):>10,.0f}")
        print(f"    median : {median(token_lens):>10,.0f}")

        buckets = {"<2k": 0, "2k-8k": 0, "8k-32k": 0, "32k-100k": 0, ">100k": 0}
        for t in token_lens:
            if   t < 2_000:   buckets["<2k"]      += 1
            elif t < 8_000:   buckets["2k-8k"]    += 1
            elif t < 32_000:  buckets["8k-32k"]   += 1
            elif t < 100_000: buckets["32k-100k"] += 1
            else:             buckets[">100k"]    += 1
        print(f"\n  TOKEN BUCKET DISTRIBUTION:")
        n = len(token_lens)
        for bucket, count in buckets.items():
            bar = "█" * int(30 * count / n)
            print(f"    {bucket:>10}  {bar}  {count} ({100*count/n:.1f}%)")

    print(f"\n  CLAUSE HEADINGS  (across extractable sample)")
    total_clauses = sum(clause_agg.values())
    if total_clauses == 0:
        print(f"    [WARN] No clause headings detected — check heading format")
    else:
        for label in ["4-level (X.X.X.X)", "3-level (X.X.X)",
                      "2-level (X.X)", "1-level (X)"]:
            cnt = clause_agg.get(label, 0)
            if cnt:
                print(f"    {label}  →  {cnt:,}")

    # ── Sample clause headings per working group ───────────────────────────
    print(f"\n  SAMPLE CLAUSE HEADINGS  (one doc per working group):")
    for wg, samples in sorted(wg_samples.items()):
        if samples:
            s = samples[0]
            print(f"\n  [{wg}]  id={s['id']}  year={s['year']}  "
                  f"version={s['version']}  tokens={s['tokens']:,}")
            for h in s["headings"][:6]:
                print(f"    {h}")

    # ── CSV metadata completeness ─────────────────────────────────────────
    if df_csv is not None:
        print(f"\n  CSV METADATA COMPLETENESS  ({len(df_csv):,} entries)")
        for col in df_csv.columns:
            filled = df_csv[col].notna().sum()
            pct    = 100 * filled / len(df_csv)
            print(f"    {col:<25} {filled:>7,}  ({pct:.1f}%)")

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "data_dir":           data_dir,
        "csv_path":           csv_path,
        "total_pdfs_found":   len(pairs),
        "working_groups":     dict(wg_counts),
        "pdfs_missing_meta":  no_meta,
        "sample_size":        len(sample_pairs),
        "extractable":        extractable_count,
        "not_extractable":    not_extractable,
        "year_counts":        {str(k): v for k, v in year_counts.items()},
        "pre_2000_count":     pre2000,
        "post_2000_count":    post2000,
        "unknown_year_count": unknown_yr,
        "token_lens": {
            "min":    min(token_lens) if token_lens else 0,
            "max":    max(token_lens) if token_lens else 0,
            "mean":   round(mean(token_lens)) if token_lens else 0,
            "median": median(token_lens) if token_lens else 0,
        },
        "token_buckets":     buckets if token_lens else {},
        "clause_headings":   dict(clause_agg),
        "csv_columns":       list(df_csv.columns) if df_csv is not None else [],
        "csv_rows":          len(df_csv) if df_csv is not None else 0,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {output_path}")

    # ── Save samples ──────────────────────────────────────────────────────
    samples_path = output_path.replace(".json", "_samples.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        for wg, samples in sorted(wg_samples.items()):
            for s in samples:
                f.write(f"\n{'='*60}\n")
                f.write(f"  Working Group : {wg}\n")
                f.write(f"  ID            : {s['id']}\n")
                f.write(f"  Year          : {s['year']}\n")
                f.write(f"  Version       : {s['version']}\n")
                f.write(f"  Tokens        : {s['tokens']:,}\n")
                f.write(f"{'='*60}\n")
                f.write(s["text_snippet"])
                f.write("\n")
    print(f"  Samples saved → {samples_path}")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit ETSI PDF corpus before building NLP dataset"
    )
    parser.add_argument("--data-dir", default="./data",
                        help="Root folder containing working group subfolders")
    parser.add_argument("--csv",      default="./ETSICatalog.csv",
                        help="Path to ETSICatalog.csv")
    parser.add_argument("--output",   default="./etsi_audit_report.json",
                        help="Output path for JSON report")
    parser.add_argument("--samples",  type=int, default=3,
                        help="Number of sample docs to save per working group")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap total docs for a quick test run")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        exit(1)

    audit(
        data_dir=args.data_dir,
        csv_path=args.csv,
        output_path=args.output,
        n_samples=args.samples,
        max_docs=args.max_docs,
    )