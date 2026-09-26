# Business Entity Resolution

Two-stage matching pipeline (blocking + RandomForest) using only the challenge TSV files.

Stack is ordinary Python: `pandas`, `numpy`, `scikit-learn`, plus the standard library (`re`, `difflib`, `unicodedata`). No LightGBM, no web APIs, no extra business databases.

## Data location

Place the official files here (tab-separated):

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

If those files are missing, `src/main.py` writes a **small local stand-in** so the code can run. Replace it with the official dataset before any leaderboard upload.

## Setup

```bash
pip install -r requirements.txt
```

## Run

From this folder (`code/business_entity_resolution`):

```bash
python src/main.py
```

Or from the workspace root:

```bash
python code/business_entity_resolution/src/main.py
```

The pipeline:

1. Loads and inspects train/test files
2. Normalizes names and addresses
3. Builds a high-recall candidate set (union of blocking keys)
4. Builds pairwise features and trains a RandomForest
5. Picks a decision threshold (and optional score margin) on a Source-1 holdout using macro F0.5
6. Scores test candidates
7. Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`
8. Runs `utils/validate_submission.py`

## Validate only

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Run that from the workspace root (the folder that contains `dataset/`, `output/`, and `utils/`).
