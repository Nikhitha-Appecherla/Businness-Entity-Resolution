"""Macro F0.5 and related validation metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from utils import parse_id_list


def f05_from_pr(precision: float, recall: float) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_f05(pred_ids: set[str], true_ids: set[str]) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return f05_from_pr(precision, recall)


def macro_f05(pred_map: dict[str, list[str]], true_map: dict[str, list[str]], s1_ids: list[str]) -> dict:
    f_scores = []
    precs = []
    recs = []
    n_pred_links = 0
    n_singletons = 0
    for s1_id in s1_ids:
        pred = set(pred_map.get(s1_id, []))
        true = set(true_map.get(s1_id, []))
        n_pred_links += len(pred)
        if not pred:
            n_singletons += 1
        f_scores.append(entity_f05(pred, true))
        if not true and not pred:
            precs.append(1.0)
            recs.append(1.0)
        elif not true and pred:
            precs.append(0.0)
            recs.append(0.0)
        elif true and not pred:
            precs.append(0.0)
            recs.append(0.0)
        else:
            tp = len(pred & true)
            precs.append(tp / len(pred))
            recs.append(tp / len(true))
    return {
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "f05": float(np.mean(f_scores)) if f_scores else 0.0,
        "n_predicted_matches": int(n_pred_links),
        "n_predicted_singletons": int(n_singletons),
        "n_entities": len(s1_ids),
    }


def gt_to_map(gt) -> dict[str, list[str]]:
    return dict(zip(gt["source1_entity_id"], gt["matched_list"]))


def scores_to_pred_map(
    pair_rows: list[tuple[str, str]],
    scores: np.ndarray,
    threshold: float,
    margin: float,
) -> dict[str, list[str]]:
    by_s1 = defaultdict(list)
    for (s1_id, other_id), score in zip(pair_rows, scores):
        by_s1[s1_id].append((other_id, float(score)))
    pred = {}
    for s1_id, items in by_s1.items():
        items.sort(key=lambda x: x[1], reverse=True)
        best = items[0][1] if items else 0.0
        kept = []
        for other_id, score in items:
            if score >= threshold and (best - score) <= margin:
                kept.append(other_id)
        pred[s1_id] = kept
    return pred


def parse_submission_map(path, id_col: str, list_col: str) -> dict[str, list[str]]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row[id_col]: parse_id_list(row[list_col]) for _, row in df.iterrows()}
