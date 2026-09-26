"""Train a RandomForest matcher and choose a precision-heavy threshold."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from blocking import blocking_recall, candidates_for_sources, flatten_candidates
from evaluate import gt_to_map, macro_f05, scores_to_pred_map
from features import FEATURE_NAMES, TfidfSpace, build_feature_matrix, fit_tfidf
from utils import SEED

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MARGINS = [0.08, 0.12, 0.20, 1.0]
NEG_PER_POS = 5


@dataclass
class TrainedModel:
    model: RandomForestClassifier
    tfidf: TfidfSpace
    threshold: float
    margin: float
    val_metrics: dict
    blocking_stats: dict


def combine_records(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)


def _positive_set(gt) -> set[tuple[str, str]]:
    pos = set()
    for s1_id, matches in zip(gt["source1_entity_id"], gt["matched_list"]):
        for other_id in matches:
            pos.add((s1_id, other_id))
    return pos


def _sample_training_pairs(
    candidate_pairs: list[tuple[str, str]],
    positives: set[tuple[str, str]],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    pos_pairs = [p for p in candidate_pairs if p in positives]
    # Recover GT pairs even if blocking missed them, so the model sees true matches.
    extra_pos = [p for p in positives if p not in set(candidate_pairs)]
    pos_pairs = pos_pairs + extra_pos
    neg_pairs = [p for p in candidate_pairs if p not in positives]
    n_keep = min(len(neg_pairs), max(len(pos_pairs) * NEG_PER_POS, 50))
    if len(neg_pairs) > n_keep:
        neg_pairs = rng.sample(neg_pairs, n_keep)
    pairs = pos_pairs + neg_pairs
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int32)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    pairs = [pairs[i] for i in order]
    labels = labels[order]
    return pairs, labels


def split_s1_ids(s1_ids: list[str], gt, seed: int = SEED, val_frac: float = 0.25):
    rng = random.Random(seed)
    true_map = gt_to_map(gt)
    has_match = [i for i in s1_ids if true_map.get(i)]
    no_match = [i for i in s1_ids if not true_map.get(i)]
    rng.shuffle(has_match)
    rng.shuffle(no_match)

    def split(ids):
        n_val = max(1, int(len(ids) * val_frac)) if ids else 0
        return ids[n_val:], ids[:n_val]

    tr_h, va_h = split(has_match)
    tr_n, va_n = split(no_match)
    return tr_h + tr_n, va_h + va_n


def _fit_forest(x: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def tune_decision_rule(
    pair_rows: list[tuple[str, str]],
    scores: np.ndarray,
    s1_ids: list[str],
    true_map: dict[str, list[str]],
) -> tuple[float, float, dict]:
    best = None
    for thresh in THRESHOLDS:
        for margin in MARGINS:
            pred = scores_to_pred_map(pair_rows, scores, thresh, margin)
            metrics = macro_f05(pred, true_map, s1_ids)
            key = (metrics["f05"], metrics["precision"], -thresh)
            if best is None or key > best[0]:
                best = (key, thresh, margin, metrics)
    _, thresh, margin, metrics = best
    return thresh, margin, metrics


def train_pipeline(train_frames: dict, gt, seed: int = SEED) -> TrainedModel:
    rng = random.Random(seed)
    df1, df2, df3 = train_frames["source1"], train_frames["source2"], train_frames["source3"]
    records = combine_records(df1, df2, df3)
    positives = _positive_set(gt)
    true_map = gt_to_map(gt)
    s1_ids = list(df1["entity_id"])
    train_ids, val_ids = split_s1_ids(s1_ids, gt, seed=seed)

    df1_tr = df1[df1["entity_id"].isin(train_ids)].copy()
    df1_va = df1[df1["entity_id"].isin(val_ids)].copy()

    cand_tr = candidates_for_sources(df1_tr, df2, df3)
    cand_va = candidates_for_sources(df1_va, df2, df3)
    cand_all = candidates_for_sources(df1, df2, df3)

    block_recall_all = blocking_recall(cand_all, gt)
    block_recall_val = blocking_recall(cand_va, gt.loc[gt["source1_entity_id"].isin(val_ids)])

    n_s1 = len(df1)
    n_other = len(df2) + len(df3)
    n_cand = sum(len(v) for v in cand_all.values())
    brute = n_s1 * n_other
    reduction = 1.0 - (n_cand / brute) if brute else 1.0

    tfidf = fit_tfidf([df1, df2, df3])
    tr_pairs, tr_y = _sample_training_pairs(flatten_candidates(cand_tr), positives, rng)
    x_tr = build_feature_matrix(tr_pairs, records, tfidf)
    model = _fit_forest(x_tr, tr_y)

    va_pairs = flatten_candidates(cand_va)
    if va_pairs:
        x_va = build_feature_matrix(va_pairs, records, tfidf)
        va_scores = model.predict_proba(x_va)[:, 1]
    else:
        va_scores = np.array([])
    thresh, margin, val_metrics = tune_decision_rule(va_pairs, va_scores, val_ids, true_map)

    # Refit on all training entities (still using candidate hard negatives).
    all_pairs, all_y = _sample_training_pairs(flatten_candidates(cand_all), positives, rng)
    x_all = build_feature_matrix(all_pairs, records, tfidf)
    model = _fit_forest(x_all, all_y)

    blocking_stats = {
        "n_source1": n_s1,
        "n_other": n_other,
        "brute_force_pairs": brute,
        "candidate_pairs": n_cand,
        "reduction_ratio": reduction,
        "blocking_recall_all": block_recall_all,
        "blocking_recall_val": block_recall_val,
        "n_train_pairs": len(tr_pairs),
        "n_train_positives": int(tr_y.sum()) if len(tr_y) else 0,
        "feature_names": FEATURE_NAMES,
    }
    val_metrics = dict(val_metrics)
    val_metrics["threshold"] = thresh
    val_metrics["margin"] = margin
    val_metrics["blocking_recall_val"] = block_recall_val
    return TrainedModel(
        model=model,
        tfidf=tfidf,
        threshold=thresh,
        margin=margin,
        val_metrics=val_metrics,
        blocking_stats=blocking_stats,
    )
