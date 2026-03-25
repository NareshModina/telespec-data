"""
build_telespec_dataset.py
=========================
Combines two telecommunications standards sources into a single
HuggingFace dataset: TeleSpec-Data.

Sources:
  - 3GPP standards  : AliMaatouk/Tele-Data (category="standard") → "3gpp-standard"
  - ETSI standards  : local Arrow dataset built by build_etsi_dataset_v3.py → "etsi-standard"

Output structure on HuggingFace:
  data/
    3gpp-standard/train-*.parquet
    etsi-standard/train-*.parquet

Subset loading:
  load_dataset("NareshModina/TeleSpec-Data")                        # all
  load_dataset("NareshModina/TeleSpec-Data", name="3gpp-standard")  # 3GPP only
  load_dataset("NareshModina/TeleSpec-Data", name="etsi-standard")  # ETSI only

Run:
    pip install datasets huggingface_hub

    python build_telespec_dataset.py \
        --etsi-dir   ./etsi-dataset/train \
        --output-dir ./telespec-dataset \
        --repo-id    NareshModina/TeleSpec-Data \
        --push

    # Dry run (no push, just build locally and print samples)
    python build_telespec_dataset.py \
        --etsi-dir   ./etsi-dataset/train \
        --output-dir ./telespec-dataset
"""

import argparse
import json
import os
import random

from datasets import (
    Dataset,
    concatenate_datasets,
    load_from_disk,
)
from huggingface_hub import HfApi


# ===========================================================================
# Constants
# ===========================================================================

RANDOM_SEED = 42

# Canonical columns — must match exactly in both subsets
COLUMNS = ["id", "category", "content", "metadata"]


# ===========================================================================
# Helpers
# ===========================================================================

def normalise_columns(ds: Dataset, required: list) -> Dataset:
    """
    Keep only the required columns, in order.
    Raises clearly if a required column is missing.
    """
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. "
                         f"Available: {ds.column_names}")
    extra = [c for c in ds.column_names if c not in required]
    if extra:
        print(f"  Dropping extra columns: {extra}")
        ds = ds.remove_columns(extra)
    # Reorder to canonical order
    ds = ds.select_columns(required)
    return ds


def print_sample(ds: Dataset, category: str, n: int = 2):
    """Print n random examples from the dataset for a given category."""
    subset = ds.filter(lambda x: x["category"] == category)
    indices = random.sample(range(len(subset)), min(n, len(subset)))
    for i in indices:
        ex = subset[i]
        meta = json.loads(ex["metadata"]) if ex["metadata"] else {}
        print(f"\n  {'─'*66}")
        print(f"  ID       : {ex['id']}")
        print(f"  Category : {ex['category']}")
        print(f"  Metadata : {json.dumps(meta, indent=4)}")
        print(f"  Content  :\n{repr(ex['content'][:400])}")
    print(f"  {'─'*66}\n")


# ===========================================================================
# Step 1 — Load and prepare 3GPP subset from Tele-Data
# ===========================================================================

def load_3gpp(gpp_dir: str, sample: int = None) -> Dataset:
    print(f"\n{'='*60}")
    print(f"  Loading 3GPP dataset from {gpp_dir} ...")
    print(f"{'='*60}")

    ds = load_from_disk(gpp_dir)
    print(f"  Loaded {len(ds):,} records")
    print(f"  Columns: {ds.column_names}")

    if sample:
        ds = ds.select(range(min(sample, len(ds))))
        print(f"  Sampled {len(ds):,} records for testing")

    # Verify category
    categories = set(ds.unique("category"))
    print(f"  Categories found: {categories}")
    if categories != {"3gpp-standard"}:
        print(f"  Remapping all to '3gpp-standard' ...")
        ds = ds.map(lambda x: {"category": "3gpp-standard"},
                    desc="  remapping category")

    # Normalise \n\n → " \n " for consistency with ETSI content
    print("  Normalising content separators...")
    ds = ds.map(
        lambda x: {"content": x["content"].replace("\n\n", " \n ")},
        desc="  normalising content"
    )

    ds = normalise_columns(ds, COLUMNS)

    print(f"  3GPP records ready : {len(ds):,}")
    print(f"  Columns            : {ds.column_names}")
    return ds


# ===========================================================================
# Step 2 — Load ETSI dataset from disk
# ===========================================================================

def load_etsi(etsi_dir: str, sample: int = None) -> Dataset:
    print(f"\n{'='*60}")
    print(f"  Loading ETSI dataset from {etsi_dir} ...")
    print(f"{'='*60}")

    ds = load_from_disk(etsi_dir)
    print(f"  Loaded {len(ds):,} records")
    print(f"  Columns: {ds.column_names}")

    if sample:
        ds = ds.select(range(min(sample, len(ds))))
        print(f"  Sampled {len(ds):,} records for testing")

    # Verify category is already correct
    categories = set(ds.unique("category"))
    print(f"  Categories found: {categories}")
    if categories != {"etsi-standard"}:
        print(f"  [WARN] Unexpected categories: {categories}")
        print(f"  Remapping all to 'etsi-standard' ...")
        ds = ds.map(lambda x: {"category": "etsi-standard"},
                    desc="  remapping category")

    ds = normalise_columns(ds, COLUMNS)

    print(f"  ETSI records ready : {len(ds):,}")
    print(f"  Columns            : {ds.column_names}")
    return ds


# ===========================================================================
# Step 3 — Combine and validate
# ===========================================================================

def combine(gpp_ds: Dataset, etsi_ds: Dataset) -> Dataset:
    print(f"\n{'='*60}")
    print(f"  Combining datasets ...")
    print(f"{'='*60}")

    combined = concatenate_datasets([gpp_ds, etsi_ds])

    # Shuffle so 3GPP and ETSI records are interleaved
    combined = combined.shuffle(seed=RANDOM_SEED)

    print(f"  Total records : {len(combined):,}")
    cats = {}
    for cat in combined.unique("category"):
        cats[cat] = sum(1 for c in combined["category"] if c == cat)
    for cat, count in cats.items():
        print(f"    {cat:<20} : {count:,}")

    return combined


# ===========================================================================
# Step 4 — Save as parquet with subset structure
# ===========================================================================

def save_parquet(gpp_ds: Dataset, etsi_ds: Dataset, output_dir: str):
    print(f"\n{'='*60}")
    print(f"  Saving parquet files to {output_dir} ...")
    print(f"{'='*60}")

    gpp_path  = os.path.join(output_dir, "data", "3gpp-standard")
    etsi_path = os.path.join(output_dir, "data", "etsi-standard")
    os.makedirs(gpp_path,  exist_ok=True)
    os.makedirs(etsi_path, exist_ok=True)

    gpp_ds.to_parquet(os.path.join(gpp_path, "train-00000.parquet"))
    etsi_ds.to_parquet(os.path.join(etsi_path, "train-00000.parquet"))

    print(f"  3GPP parquet  → {gpp_path}/train-00000.parquet")
    print(f"  ETSI parquet  → {etsi_path}/train-00000.parquet")


# ===========================================================================
# Step 5 — Push to HuggingFace
# ===========================================================================

def push_to_hub(output_dir: str, repo_id: str):
    print(f"\n{'='*60}")
    print(f"  Pushing to HuggingFace: {repo_id} ...")
    print(f"{'='*60}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"  Repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"  [WARN] Repo creation: {e}")

    # Upload parquet files
    data_dir = os.path.join(output_dir, "data")
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for fname in sorted(files):
            if not fname.endswith(".parquet"):
                continue
            local_path = os.path.join(root, fname)
            # Preserve the data/subset/filename structure in the repo
            path_in_repo = os.path.relpath(local_path, output_dir).replace("\\", "/")
            print(f"  Uploading {path_in_repo} ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )

    # Upload README if present
    readme_path = os.path.join(output_dir, "README.md")
    if os.path.exists(readme_path):
        print(f"  Uploading README.md ...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"\n  Done → https://huggingface.co/datasets/{repo_id}")


# ===========================================================================
# Verification — print samples
# ===========================================================================

def verify(combined: Dataset):
    print(f"\n{'='*60}")
    print(f"  VERIFICATION — sample records")
    print(f"{'='*60}")

    print(f"\n  [3gpp-standard] — 2 random samples:")
    print_sample(combined, "3gpp-standard", n=2)

    print(f"  [etsi-standard] — 2 random samples:")
    print_sample(combined, "etsi-standard", n=2)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build TeleSpec-Data by combining Tele-Data 3GPP + ETSI dataset"
    )
    parser.add_argument("--gpp-dir",    required=True,
                        help="Path to local 3GPP Arrow dataset (./3gpp-dataset/train)")
    parser.add_argument("--etsi-dir",   required=True,
                        help="Path to local ETSI Arrow dataset (./etsi-dataset/train)")
    parser.add_argument("--output-dir", default="./telespec-dataset",
                        help="Local output directory for parquet files")
    parser.add_argument("--repo-id",    default="NareshModina/TeleSpec-Data",
                        help="HuggingFace repo id to push to")
    parser.add_argument("--push",       action="store_true",
                        help="Push to HuggingFace after building")
    parser.add_argument("--readme",     default=None,
                        help="Path to README.md to include in the repo")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Limit each source to N records (for testing)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  TeleSpec-Data Dataset Builder")
    print(f"{'='*60}")
    print(f"  3GPP dir   : {args.gpp_dir}")
    print(f"  ETSI dir   : {args.etsi_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Repo ID    : {args.repo_id}")
    print(f"  Push       : {args.push}")
    print(f"  Sample     : {args.sample or 'all'}")

    # ── Load ─────────────────────────────────────────────────────────────
    gpp_ds  = load_3gpp(args.gpp_dir, sample=args.sample)
    etsi_ds = load_etsi(args.etsi_dir, sample=args.sample)

    # ── Combine for verification ──────────────────────────────────────────
    combined = combine(gpp_ds, etsi_ds)

    # ── Verify ───────────────────────────────────────────────────────────
    verify(combined)

    # ── Save parquet ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy README if provided
    if args.readme and os.path.exists(args.readme):
        import shutil
        shutil.copy(args.readme, os.path.join(args.output_dir, "README.md"))
        print(f"\n  README copied → {args.output_dir}/README.md")

    save_parquet(gpp_ds, etsi_ds, args.output_dir)

    # ── Final stats ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL STATS")
    print(f"{'='*60}")
    print(f"  3GPP records  : {len(gpp_ds):,}")
    print(f"  ETSI records  : {len(etsi_ds):,}")
    print(f"  Total         : {len(combined):,}")
    print(f"  Output dir    : {args.output_dir}")

    # ── Push ─────────────────────────────────────────────────────────────
    if args.push:
        push_to_hub(args.output_dir, args.repo_id)
    else:
        print(f"\n  [DRY RUN] Not pushed. Run with --push to upload to HuggingFace.")
        print(f"  Preview locally:")
        print(f"    from datasets import load_dataset")
        print(f"    ds = load_dataset('parquet',")
        print(f"         data_files='{args.output_dir}/data/*/*.parquet')")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()