"""Pairwise similarity features for name, address, and country."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import extract_numbers
from utils import tokenize

FEATURE_NAMES = [
    "name_exact",
    "name_seq_ratio",
    "name_jaccard",
    "name_token_overlap",
    "name_lev_ratio",
    "name_tfidf",
    "name_len_ratio",
    "addr_exact",
    "addr_seq_ratio",
    "addr_jaccard",
    "addr_token_overlap",
    "addr_lev_ratio",
    "addr_tfidf",
    "addr_number_jaccard",
    "country_match",
    "country_mismatch",
    "country_both_missing",
    "name_and_addr_avg",
    "name_missing_left",
    "name_missing_right",
    "addr_missing_left",
    "addr_missing_right",
    "same_first_token",
]


def _seq_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _token_overlap(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a = a[:80]
    b = b[:80]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return 1.0 - prev[-1] / max(n, m)


def _len_ratio(a: str, b: str) -> float:
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    return min(n, m) / max(n, m) if max(n, m) else 0.0


def _number_jaccard(a: str, b: str) -> float:
    sa, sb = set(extract_numbers(a)), set(extract_numbers(b))
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class TfidfSpace:
    name_vec: TfidfVectorizer
    addr_vec: TfidfVectorizer


def fit_tfidf(frames: list[pd.DataFrame]) -> TfidfSpace:
    names, addrs = [], []
    for df in frames:
        names.extend([(x or " ") for x in df["name_norm"].tolist()])
        addrs.extend([(x or " ") for x in df["addr_norm"].tolist()])
    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    addr_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    name_vec.fit(names)
    addr_vec.fit(addrs)
    return TfidfSpace(name_vec=name_vec, addr_vec=addr_vec)


def transform_tfidf(tfidf: TfidfSpace, records: pd.DataFrame) -> dict:
    ids = records["entity_id"].tolist()
    names = [(x or " ") for x in records["name_norm"].tolist()]
    addrs = [(x or " ") for x in records["addr_norm"].tolist()]
    return {
        "name": {"mat": tfidf.name_vec.transform(names), "ix": {eid: i for i, eid in enumerate(ids)}},
        "addr": {"mat": tfidf.addr_vec.transform(addrs), "ix": {eid: i for i, eid in enumerate(ids)}},
    }


def _tfidf_cos(space_dict: dict, id_a: str, id_b: str) -> float:
    ix = space_dict["ix"]
    if id_a not in ix or id_b not in ix:
        return 0.0
    va = space_dict["mat"][ix[id_a]]
    vb = space_dict["mat"][ix[id_b]]
    sim = cosine_similarity(va, vb)
    return float(sim[0, 0])


def pair_feature_row(left: pd.Series, right: pd.Series, tfidf_mats: dict | None) -> list[float]:
    n1, n2 = left["name_norm"], right["name_norm"]
    a1, a2 = left["addr_norm"], right["addr_norm"]
    c1, c2 = left["country_norm"], right["country_norm"]

    name_exact = float(n1 != "" and n1 == n2)
    name_seq = _seq_ratio(n1, n2)
    name_jac = _jaccard(n1, n2)
    addr_exact = float(a1 != "" and a1 == a2)
    addr_seq = _seq_ratio(a1, a2)
    addr_jac = _jaccard(a1, a2)
    country_match = float(c1 != "" and c1 == c2)
    country_mismatch = float(c1 != "" and c2 != "" and c1 != c2)

    name_tfidf = _tfidf_cos(tfidf_mats["name"], left["entity_id"], right["entity_id"]) if tfidf_mats else 0.0
    addr_tfidf = _tfidf_cos(tfidf_mats["addr"], left["entity_id"], right["entity_id"]) if tfidf_mats else 0.0

    return [
        name_exact,
        name_seq,
        name_jac,
        _token_overlap(n1, n2),
        _levenshtein_ratio(n1, n2),
        name_tfidf,
        _len_ratio(n1, n2),
        addr_exact,
        addr_seq,
        addr_jac,
        _token_overlap(a1, a2),
        _levenshtein_ratio(a1, a2),
        addr_tfidf,
        _number_jaccard(a1, a2),
        country_match,
        country_mismatch,
        float(c1 == "" and c2 == ""),
        (name_seq + addr_seq) / 2.0,
        float(n1 == ""),
        float(n2 == ""),
        float(a1 == ""),
        float(a2 == ""),
        float(bool(n1) and bool(n2) and n1.split()[0] == n2.split()[0]),
    ]


def build_feature_matrix(
    pairs: list[tuple[str, str]],
    records: pd.DataFrame,
    tfidf: TfidfSpace | None,
) -> np.ndarray:
    lookup = records.set_index("entity_id", drop=False)
    tfidf_mats = transform_tfidf(tfidf, records) if tfidf is not None else None
    rows = []
    for s1_id, other_id in pairs:
        left = lookup.loc[s1_id]
        right = lookup.loc[other_id]
        rows.append(pair_feature_row(left, right, tfidf_mats))
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
