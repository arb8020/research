# Data Directory

Contains scripts for downloading and preprocessing the FineWeb-10B dataset.

## Quick Start (Recommended)

Download pre-tokenized data (fast, ~5-10 minutes):

```bash
# Download 9 chunks (~1.8 GB, ~900M tokens) - enough for speedrun
python cached_fineweb10B.py 9

# Or download full dataset (103 chunks, ~20 GB)
python cached_fineweb10B.py
```

This creates `fineweb10B/` directory with `.bin` files.

## Alternative: Tokenize From Scratch

If you want to customize tokenization (~1 hour):

```bash
# Download and tokenize FineWeb-10B
python fineweb.py --version 10B

# Or FineWeb-100B (much larger)
python fineweb.py --version 100B
```

## Files

- **`cached_fineweb10B.py`** - Download pre-tokenized .bin files (fast, recommended)
- **`fineweb.py`** - Download raw text and tokenize yourself (slow, for customization)

## Output

Both scripts produce the same format:

```
data/
├── cached_fineweb10B.py
├── fineweb.py
└── fineweb10B/              # Created after running either script
    ├── fineweb_val_000000.bin
    ├── fineweb_train_000001.bin
    ├── fineweb_train_000002.bin
    └── ... (up to 103 training chunks)
```

## Binary File Format

Each `.bin` file contains:
- Header: 256 × 4 bytes (magic number, version, token count, reserved)
- Tokens: N × 2 bytes (uint16, GPT-2 tokenized)

This format is read by `DistributedDataLoader` in the training scripts.
