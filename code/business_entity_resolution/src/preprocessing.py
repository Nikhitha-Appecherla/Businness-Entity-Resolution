"""Normalize business names and addresses. Keep originals elsewhere."""

from __future__ import annotations

import re
import unicodedata

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "llp",
    "lp",
    "plc",
    "pvt",
    "private",
    "pvtltd",
    "gmbh",
    "sarl",
    "sas",
    "sa",
    "spa",
    "bv",
    "nv",
    "pty",
    "group",
    "holdings",
    "holding",
    "enterprises",
    "enterprise",
    "services",
    "service",
    "solutions",
    "industries",
    "industry",
}

# Applied as whole-token replacements after lowercasing.
NAME_ABBREV = {
    "intl": "international",
    "int": "international",
    "intl.": "international",
    "tech": "technology",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "mgmt": "management",
    "assoc": "associates",
    "assn": "association",
    "bros": "brothers",
    "dept": "department",
}

ADDRESS_ABBREV = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "hwy": "highway",
    "rte": "route",
    "pkwy": "parkway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "po": "pobox",
    "pob": "pobox",
    "pin": "pin",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def basic_clean(text) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return ""
    raw = _strip_accents(raw).lower()
    raw = raw.replace("&", " and ")
    raw = raw.replace("+", " and ")
    raw = _PUNCT_RE.sub(" ", raw)
    raw = _MULTI_SPACE_RE.sub(" ", raw).strip()
    return raw


def normalize_name(text) -> str:
    cleaned = basic_clean(text)
    if not cleaned:
        return ""
    tokens = []
    for tok in cleaned.split():
        tok = NAME_ABBREV.get(tok, tok)
        if tok in LEGAL_SUFFIXES:
            continue
        tokens.append(tok)
    # If everything was a suffix, keep the cleaned string.
    if not tokens:
        return cleaned
    return " ".join(tokens)


def normalize_address(text) -> str:
    cleaned = basic_clean(text)
    if not cleaned:
        return ""
    tokens = []
    skip_next_box = False
    for tok in cleaned.split():
        if skip_next_box and tok in {"box", "boxes"}:
            skip_next_box = False
            tokens.append("pobox")
            continue
        skip_next_box = False
        mapped = ADDRESS_ABBREV.get(tok, tok)
        if mapped == "pobox":
            skip_next_box = False
        tokens.append(mapped)
    return " ".join(tokens)


def name_prefix(name_norm: str, n: int = 5) -> str:
    compact = name_norm.replace(" ", "")
    return compact[:n]


def first_token(text_norm: str) -> str:
    parts = text_norm.split()
    return parts[0] if parts else ""


def significant_tokens(text_norm: str, min_len: int = 3) -> list[str]:
    return [tok for tok in text_norm.split() if len(tok) >= min_len]


def extract_numbers(text_norm: str) -> list[str]:
    return _DIGIT_RE.findall(text_norm or "")


def longest_token(text_norm: str) -> str:
    parts = text_norm.split()
    if not parts:
        return ""
    return max(parts, key=len)
