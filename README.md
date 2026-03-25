# TeleSpec-Data

Pipeline for building [TeleSpec-Data](https://huggingface.co/datasets/NareshModina/TeleSpec-Data) — a telecommunications standards dataset combining ETSI and 3GPP documents for LLM continual pretraining.

## Overview

| Script | Purpose |
|---|---|
| `audit_etsi.py` | Corpus audit — year distribution, extractability, clause heading detection |
| `build_etsi_dataset_v3.py` | Extract and structure ETSI PDFs into HuggingFace Arrow format |
| `build_3gpp_dataset.py` | Convert TSpec-LLM markdown files into HuggingFace Arrow format |
| `build_telespec_dataset.py` | Combine ETSI + 3GPP subsets and push to HuggingFace |

## Requirements

```bash
pip install pymupdf pandas datasets tqdm huggingface_hub
```

## Usage

### 1. Audit the ETSI corpus
```bash
python audit_etsi.py \
    --data-dir ./data \
    --csv      ./ETSICatalog.csv \
    --output   ./etsi_audit_report.json
```

### 2. Build the ETSI dataset
```bash
python build_etsi_dataset_v3.py \
    --data-dir ./data \
    --csv      ./ETSICatalog.csv \
    --output   ./etsi-dataset \
    --skip-log ./etsi_skipped.txt
```

### 3. Inspect skipped documents (optional)
```bash
python inspect_skipped_etsi.py \
    --data-dir ./data \
    --skip-log ./etsi_skipped.txt \
    --output   ./etsi_skipped_inspection.txt
```

### 4. Build and push TeleSpec-Data
```bash
# Dry run
python build_telespec_dataset.py \
    --gpp-dir    ./3gpp-dataset/train \
    --etsi-dir   ./etsi-dataset/train \
    --output-dir ./telespec-dataset \
    --readme     ./README_HF.md

# Build 3GPP dataset from TSpec-LLM
python build_3gpp_dataset.py \
    --data-dir ./TSpec-LLM/3GPP-clean \
    --output   ./3gpp-dataset

# Full build + push
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
python build_telespec_dataset.py \
    --gpp-dir    ./3gpp-dataset/train \
    --etsi-dir   ./etsi-dataset/train \
    --output-dir ./telespec-dataset \
    --readme     ./README_HF.md \
    --repo-id    YourUsername/TeleSpec-Data \
    --push
```

## Data Sources

- **3GPP standards**: Markdown corpus from [rasoul-nikbakht/TSpec-LLM](https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM) (Rel-8 to Rel-19, updated April 2025)
- **ETSI documents**: PDF corpus from [rasoul-nikbakht/NetSpec-LLM](https://huggingface.co/datasets/rasoul-nikbakht/NetSpec-LLM)

## Dataset

The final dataset is available at [NareshModina/TeleSpec-Data](https://huggingface.co/datasets/NareshModina/TeleSpec-Data).

```python
from datasets import load_dataset

ds = load_dataset("NareshModina/TeleSpec-Data")                        # full
ds = load_dataset("NareshModina/TeleSpec-Data", name="etsi-standard")  # ETSI only
ds = load_dataset("NareshModina/TeleSpec-Data", name="3gpp-standard")  # 3GPP only
```

## License

CC BY-NC 4.0 — see [LICENSE](LICENSE) for details.