#!/usr/bin/env python3
"""Validate matching_results.tsv and candidate_pairs.tsv (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_ids(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def load_tsv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def load_source_ids(test_dir: Path, filename: str, prefix: str) -> set[str]:
    path = test_dir / filename
    rows = load_tsv(path)
    ids = set()
    for row in rows:
        eid = (row.get("entity_id") or "").strip()
        if eid:
            ids.add(eid)
            if not eid.startswith(prefix):
                pass
    return ids


def check_file(path: Path, list_col: str, s1_ids: set[str], allowed: set[str], label: str) -> list[str]:
    issues = []
    if not path.exists():
        return [f"{label}: file not found: {path}"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required = ["source1_entity_id", list_col]
        for col in required:
            if col not in fieldnames:
                issues.append(f"{label}: missing column '{col}'. Found: {fieldnames}")
                return issues
        rows = list(reader)

    seen_s1 = []
    seen_set = set()
    for i, row in enumerate(rows, start=2):
        s1 = (row.get("source1_entity_id") or "").strip()
        if not s1:
            issues.append(f"{label}: line {i} has empty source1_entity_id")
            continue
        if s1 in seen_set:
            issues.append(f"{label}: duplicate source1_entity_id {s1}")
        seen_set.add(s1)
        seen_s1.append(s1)
        if not s1.startswith("S1-"):
            issues.append(f"{label}: {s1} is not a Source 1 id")
        ids = parse_ids(row.get(list_col) or "")
        if len(ids) != len(set(ids)):
            issues.append(f"{label}: duplicate ids in list for {s1}")
        for eid in ids:
            if not (eid.startswith("S2-") or eid.startswith("S3-")):
                issues.append(f"{label}: {s1} contains non S2/S3 id {eid}")
            elif eid not in allowed:
                issues.append(f"{label}: {s1} references unknown test id {eid}")

    missing = sorted(s1_ids - seen_set)
    extra = sorted(seen_set - s1_ids)
    if missing:
        issues.append(f"{label}: missing {len(missing)} Source 1 ids, e.g. {missing[:5]}")
    if extra:
        issues.append(f"{label}: extra {len(extra)} Source 1 ids, e.g. {extra[:5]}")
    if len(rows) != len(s1_ids) and not missing and not extra:
        issues.append(f"{label}: expected {len(s1_ids)} rows, found {len(rows)}")
    return issues


def check_subset(matching_path: Path, candidate_path: Path) -> list[str]:
    issues = []
    match_rows = load_tsv(matching_path)
    cand_rows = load_tsv(candidate_path)
    cand_map = {
        (row.get("source1_entity_id") or "").strip(): set(parse_ids(row.get("candidate_entity_ids") or ""))
        for row in cand_rows
    }
    for row in match_rows:
        s1 = (row.get("source1_entity_id") or "").strip()
        matched = parse_ids(row.get("matched_entity_ids") or "")
        cands = cand_map.get(s1, set())
        for eid in matched:
            if eid not in cands:
                issues.append(
                    f"matched id {eid} for {s1} is not in candidate_pairs.tsv"
                )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()

    matching = Path(args.matching)
    candidate = Path(args.candidate)
    test_dir = Path(args.test_dir)

    issues = []
    s1 = load_source_ids(test_dir, "test_source1.tsv", "S1-")
    s2 = load_source_ids(test_dir, "test_source2.tsv", "S2-")
    s3 = load_source_ids(test_dir, "test_source3.tsv", "S3-")
    allowed = s2 | s3
    if not s1:
        issues.append("No Source 1 ids found in test_source1.tsv")

    issues.extend(check_file(matching, "matched_entity_ids", s1, allowed, "matching_results.tsv"))
    issues.extend(check_file(candidate, "candidate_entity_ids", s1, allowed, "candidate_pairs.tsv"))
    if matching.exists() and candidate.exists():
        issues.extend(check_subset(matching, candidate))

    if issues:
        print("FAIL")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
