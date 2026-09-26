"""Score test candidates and write submission TSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from blocking import candidates_for_sources, flatten_candidates
from evaluate import scores_to_pred_map
from features import build_feature_matrix
from train import TrainedModel, combine_records
from utils import format_id_list


def predict_split(
    frames: dict,
    trained: TrainedModel,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    df1, df2, df3 = frames["source1"], frames["source2"], frames["source3"]
    records = combine_records(df1, df2, df3)
    candidate_map = candidates_for_sources(df1, df2, df3)
    pairs = flatten_candidates(candidate_map)
    if pairs:
        x = build_feature_matrix(pairs, records, trained.tfidf)
        scores = trained.model.predict_proba(x)[:, 1]
        pred_map = scores_to_pred_map(pairs, scores, trained.threshold, trained.margin)
    else:
        pred_map = {}
    for s1_id in df1["entity_id"]:
        candidate_map.setdefault(s1_id, [])
        pred_map.setdefault(s1_id, [])
        allowed = set(candidate_map[s1_id])
        pred_map[s1_id] = [x for x in pred_map[s1_id] if x in allowed]
    return candidate_map, pred_map


def write_pair_file(path: Path, s1_ids: list[str], mapping: dict[str, list[str]], id_col: str, list_col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in s1_ids:
        rows.append({id_col: s1_id, list_col: format_id_list(mapping.get(s1_id, []))})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def write_outputs(
    output_dir: Path,
    s1_ids: list[str],
    candidate_map: dict[str, list[str]],
    pred_map: dict[str, list[str]],
) -> tuple[Path, Path]:
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    write_pair_file(matching_path, s1_ids, pred_map, "source1_entity_id", "matched_entity_ids")
    write_pair_file(candidate_path, s1_ids, candidate_map, "source1_entity_id", "candidate_entity_ids")
    return matching_path, candidate_path
