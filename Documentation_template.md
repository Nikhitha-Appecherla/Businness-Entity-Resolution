# Methodology

## Problem

Match each Source 1 business record to zero or more Source 2 / Source 3 records. Evaluation is **macro F0.5** (precision-weighted). Singletons (empty match lists) are scored and must not be forced to a match.

Country is treated as an open string. No country list is hard-coded in the model. Normalization includes common legal suffixes (including ones that appear in French company names) without filtering records by country.

## Data used

Only the supplied `dataset/train` and `dataset/test` TSV files. No geocoding, no company registries, no web search, no external ER APIs.

## Candidate generation (Stage 1)

Candidates are the **union** of these blocking keys:

1. Exact normalized name
2. Name prefix (4 and 5 characters)
3. First name token
4. Normalized name + country
5. Name prefix + country
6. First name token + country
7. Sorted significant name tokens
8. Longest address token + country
9. Address numbers + country
10. Exact normalized address + country

Very large buckets are skipped so one generic key cannot explode into a full cartesian product. The candidate file is this union — the same pairs scored by the matcher.

## Features (Stage 2)

Name and address: exact match, SequenceMatcher ratio, Jaccard, token overlap, Levenshtein ratio, TF-IDF cosine (char n-grams for names, word n-grams for addresses), length ratio.

Other: shared address numbers, country match / mismatch / both missing, missing-value flags, same first token, average of name and address sequence scores.

## Model

`sklearn.ensemble.RandomForestClassifier` (balanced class weights). Positives come from `train_ground_truth.tsv`. Negatives are **unlabeled candidate pairs** (hard negatives), downsampled. Source 1 entities are split for validation so threshold tuning is not done on test labels.

Thresholds tried: 0.50–0.95. A score-margin rule keeps matches within `margin` of the best score for that Source 1 entity, which drops weak extra hits. Both hyperparameters are chosen by validation macro F0.5.

If no candidate is above the threshold, the match list is empty.

## Reproducibility

```bash
python code/business_entity_resolution/src/main.py
```

Random seed is 42.
