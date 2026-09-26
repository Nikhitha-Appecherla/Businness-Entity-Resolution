"""High-recall candidate generation by union of several blocking keys."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from preprocessing import extract_numbers, first_token, longest_token, name_prefix, significant_tokens

MAX_BUCKET_PAIRS = 2500


def _bucket(df: pd.DataFrame, keys: pd.Series) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for entity_id, key in zip(df["entity_id"], keys):
        if not key:
            continue
        buckets[str(key)].append(entity_id)
    return buckets


def _emit_pairs(left: dict[str, list[str]], right: dict[str, list[str]], out: set[tuple[str, str]]) -> None:
    for key, left_ids in left.items():
        right_ids = right.get(key)
        if not right_ids:
            continue
        if len(left_ids) * len(right_ids) > MAX_BUCKET_PAIRS:
            continue
        for a in left_ids:
            for b in right_ids:
                out.add((a, b))


def generate_candidates(df_s1: pd.DataFrame, df_other: pd.DataFrame) -> set[tuple[str, str]]:
    """Return (s1_id, other_id) pairs using the union of blocking strategies."""
    pairs: set[tuple[str, str]] = set()

    s1_name = df_s1["name_norm"]
    ot_name = df_other["name_norm"]
    s1_addr = df_s1["addr_norm"]
    ot_addr = df_other["addr_norm"]
    s1_cty = df_s1["country_norm"]
    ot_cty = df_other["country_norm"]

    strategies = {
        "exact_name": (s1_name, ot_name),
        "name_prefix": (s1_name.map(lambda x: name_prefix(x, 5)), ot_name.map(lambda x: name_prefix(x, 5))),
        "name_prefix4": (s1_name.map(lambda x: name_prefix(x, 4)), ot_name.map(lambda x: name_prefix(x, 4))),
        "first_token": (s1_name.map(first_token), ot_name.map(first_token)),
        "name_country": (
            (s1_name + "|" + s1_cty).where(s1_name.ne("") & s1_cty.ne(""), ""),
            (ot_name + "|" + ot_cty).where(ot_name.ne("") & ot_cty.ne(""), ""),
        ),
        "prefix_country": (
            (s1_name.map(lambda x: name_prefix(x, 5)) + "|" + s1_cty).where(s1_cty.ne(""), ""),
            (ot_name.map(lambda x: name_prefix(x, 5)) + "|" + ot_cty).where(ot_cty.ne(""), ""),
        ),
        "first_token_country": (
            (s1_name.map(first_token) + "|" + s1_cty).where(s1_cty.ne(""), ""),
            (ot_name.map(first_token) + "|" + ot_cty).where(ot_cty.ne(""), ""),
        ),
        "sorted_name_tokens": (
            s1_name.map(lambda x: " ".join(sorted(significant_tokens(x))) if len(significant_tokens(x)) >= 2 else ""),
            ot_name.map(lambda x: " ".join(sorted(significant_tokens(x))) if len(significant_tokens(x)) >= 2 else ""),
        ),
        "longest_addr_country": (
            (s1_addr.map(longest_token) + "|" + s1_cty).where(s1_cty.ne("") & s1_addr.ne(""), ""),
            (ot_addr.map(longest_token) + "|" + ot_cty).where(ot_cty.ne("") & ot_addr.ne(""), ""),
        ),
        "addr_number_country": (
            s1_addr.map(lambda x: "|".join(extract_numbers(x)[:2])) + "|" + s1_cty,
            ot_addr.map(lambda x: "|".join(extract_numbers(x)[:2])) + "|" + ot_cty,
        ),
        "exact_addr_country": (
            (s1_addr + "|" + s1_cty).where(s1_addr.ne("") & s1_cty.ne(""), ""),
            (ot_addr + "|" + ot_cty).where(ot_addr.ne("") & ot_cty.ne(""), ""),
        ),
    }

    # Drop empty / weak numeric-only keys from number blocking.
    s1_num_keys = []
    ot_num_keys = []
    for key in strategies["addr_number_country"][0]:
        s1_num_keys.append(key if key and not str(key).startswith("|") else "")
    for key in strategies["addr_number_country"][1]:
        ot_num_keys.append(key if key and not str(key).startswith("|") else "")
    strategies["addr_number_country"] = (pd.Series(s1_num_keys), pd.Series(ot_num_keys))

    for _name, (left_keys, right_keys) in strategies.items():
        _emit_pairs(_bucket(df_s1, left_keys), _bucket(df_other, right_keys), pairs)

    return pairs


def candidates_for_sources(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame) -> dict[str, list[str]]:
    pairs = generate_candidates(df_s1, df_s2) | generate_candidates(df_s1, df_s3)
    mapping: dict[str, list[str]] = {eid: [] for eid in df_s1["entity_id"]}
    seen = {eid: set() for eid in df_s1["entity_id"]}
    for s1_id, other_id in pairs:
        if other_id not in seen.get(s1_id, set()):
            seen.setdefault(s1_id, set()).add(other_id)
            mapping.setdefault(s1_id, []).append(other_id)
    return mapping


def flatten_candidates(candidate_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    rows = []
    for s1_id, others in candidate_map.items():
        for other_id in others:
            rows.append((s1_id, other_id))
    return rows


def blocking_recall(candidate_map: dict[str, list[str]], gt: pd.DataFrame) -> float:
    recovered = 0
    total = 0
    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_list"]))
    for s1_id, true_ids in gt_map.items():
        total += len(true_ids)
        cand = set(candidate_map.get(s1_id, []))
        recovered += sum(1 for t in true_ids if t in cand)
    if total == 0:
        return 1.0
    return recovered / total
