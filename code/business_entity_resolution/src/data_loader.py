"""Load TSV sources, profile the data, and create a local stand-in if files are missing."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from preprocessing import normalize_address, normalize_name
from utils import parse_id_list, set_seed

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_source_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Found: {list(df.columns)}")
    df = df[SOURCE_COLUMNS].copy()
    for col in SOURCE_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df.loc[df[col].str.lower().isin(["nan", "none", "null"]), col] = ""
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["addr_norm"] = df["business_address"].map(normalize_address)
    df["country_norm"] = df["country"].str.strip().str.lower()
    return df


def read_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Found: {list(df.columns)}")
    df = df[GT_COLUMNS].copy()
    df["source1_entity_id"] = df["source1_entity_id"].astype(str).str.strip()
    df["matched_list"] = df["matched_entity_ids"].map(parse_id_list)
    df["n_matches"] = df["matched_list"].map(len)
    return df


def add_preprocessed_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["addr_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].fillna("").astype(str).str.strip().str.lower()
    return out


def _blank_rate(series: pd.Series) -> float:
    return float((series.fillna("").astype(str).str.strip() == "").mean())


def profile_sources(name: str, frames: dict[str, pd.DataFrame], gt: pd.DataFrame | None = None) -> dict:
    """Collect the inspection stats requested before modeling."""
    stats = {"split": name, "sources": {}}
    for src, df in frames.items():
        stats["sources"][src] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing": {
                "business_name": _blank_rate(df["business_name"]),
                "business_address": _blank_rate(df["business_address"]),
                "country": _blank_rate(df["country"]),
            },
            "n_countries": int(df["country"].replace("", pd.NA).nunique(dropna=True)),
            "country_counts": df["country"].value_counts(dropna=False).head(10).to_dict(),
            "head": df[SOURCE_COLUMNS].head(5).to_dict(orient="records"),
        }
    if gt is not None:
        n_s1 = len(frames.get("source1", []))
        n_singleton = int((gt["n_matches"] == 0).sum())
        match_dist = gt["n_matches"].value_counts().sort_index().to_dict()
        n_links = int(gt["n_matches"].sum())
        stats["ground_truth"] = {
            "rows": int(len(gt)),
            "total_links": n_links,
            "singletons": n_singleton,
            "singleton_rate": n_singleton / max(n_s1, 1),
            "match_count_distribution": {str(k): int(v) for k, v in match_dist.items()},
        }
    return stats


def print_profile(stats: dict, noisy_name_rows: list, noisy_addr_rows: list) -> None:
    print("=" * 72)
    print(f"DATA INSPECTION ({stats['split']})")
    print("=" * 72)
    for src, info in stats["sources"].items():
        print(f"\n[{src}] rows={info['rows']}  columns={info['columns']}")
        print(f"  missing name={info['missing']['business_name']:.3f}  "
              f"address={info['missing']['business_address']:.3f}  "
              f"country={info['missing']['country']:.3f}")
        print(f"  country counts: {info['country_counts']}")
        print("  first rows:")
        for row in info["head"]:
            print(f"    {row}")
    if "ground_truth" in stats:
        gt = stats["ground_truth"]
        print("\n[ground_truth]")
        print(f"  rows={gt['rows']}  total_links={gt['total_links']}  "
              f"singletons={gt['singletons']} ({gt['singleton_rate']:.3f})")
        print(f"  match-count distribution: {gt['match_count_distribution']}")
    print("\nNoisy name examples:")
    for row in noisy_name_rows:
        print(f"  {row}")
    print("\nNoisy address examples:")
    for row in noisy_addr_rows:
        print(f"  {row}")
    print("=" * 72)


def pick_noisy_examples(frames: dict[str, pd.DataFrame], n: int = 8) -> tuple[list, list]:
    names, addrs = [], []
    for df in frames.values():
        for _, row in df.iterrows():
            name = row["business_name"]
            addr = row["business_address"]
            if any(ch in name for ch in "&.,/") or name != name.title():
                names.append({"id": row["entity_id"], "business_name": name})
            if any(tok in addr.lower() for tok in ["rd", "st.", "near", "pvt", "#"]):
                addrs.append({"id": row["entity_id"], "business_address": addr})
    return names[:n], addrs[:n]


def load_split(train_dir: Path | None = None, test_dir: Path | None = None) -> dict:
    from utils import workspace_paths

    paths = workspace_paths()
    train_dir = Path(train_dir) if train_dir else paths["train_dir"]
    test_dir = Path(test_dir) if test_dir else paths["test_dir"]

    if not (train_dir / "train_source1.tsv").exists():
        print("WARNING: official training files were not found.")
        print(f"Expected: {train_dir / 'train_source1.tsv'}")
        print("Creating a small local stand-in dataset so the pipeline can run.")
        print("Replace dataset/ with the official challenge files and re-run.")
        create_standin_dataset(paths["root"] / "dataset")

    train = {
        "source1": read_source_tsv(train_dir / "train_source1.tsv"),
        "source2": read_source_tsv(train_dir / "train_source2.tsv"),
        "source3": read_source_tsv(train_dir / "train_source3.tsv"),
    }
    gt = read_ground_truth(train_dir / "train_ground_truth.tsv")
    test = {
        "source1": read_source_tsv(test_dir / "test_source1.tsv"),
        "source2": read_source_tsv(test_dir / "test_source2.tsv"),
        "source3": read_source_tsv(test_dir / "test_source3.tsv"),
    }
    return {"train": train, "gt": gt, "test": test, "paths": paths}


def create_standin_dataset(dataset_root: Path, seed: int = 42) -> None:
    """Build a tiny noisy dataset that follows the PDF schema. Not official data."""
    set_seed(seed)
    rng = random.Random(seed)
    dataset_root = Path(dataset_root)
    (dataset_root / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "test").mkdir(parents=True, exist_ok=True)

    us_cities = [
        ("12 Main St", "Austin", "TX", "78701"),
        ("400 Market Street", "San Francisco", "CA", "94105"),
        ("88 Oak Rd", "Chicago", "IL", "60601"),
        ("15 Broadway", "New York", "NY", "10006"),
        ("230 Pine Ave", "Seattle", "WA", "98101"),
    ]
    in_cities = [
        ("14 MG Road", "Bengaluru", "KA", "560001"),
        ("7 Linking Road", "Mumbai", "MH", "400050"),
        ("22 Anna Salai", "Chennai", "TN", "600002"),
        ("5 Park Street", "Kolkata", "WB", "700016"),
        ("11 Connaught Place", "New Delhi", "DL", "110001"),
    ]
    fr_cities = [
        ("10 Rue de Rivoli", "Paris", "", "75001"),
        ("4 Avenue Jean Jaures", "Lyon", "", "69007"),
        ("18 Boulevard de la Liberte", "Lille", "", "59000"),
        ("2 Place Bellecour", "Lyon", "", "69002"),
        ("9 Rue Sainte Catherine", "Bordeaux", "", "33000"),
    ]
    stems = [
        "Acme Technology",
        "Sunrise Foods",
        "Blue River Logistics",
        "Pacific Hardware",
        "Nimbus Software",
        "Green Leaf Pharma",
        "Harbor Retail",
        "Summit Construction",
        "Lotus Textiles",
        "Vertex Consulting",
        "Silver Oak Finance",
        "Metro Auto Parts",
        "Cedar Health Clinic",
        "Bright Spark Education",
        "Orion Media",
        "Maple Grocery",
        "Atlas Engineering",
        "Phoenix Apparel",
        "Canyon Coffee",
        "Delta Packaging",
        "Kaveri Mills",
        "Ganga Traders",
        "Sahara Electronics",
        "Deccan Motors",
        "Himalaya Spices",
    ]

    def noise_name(name: str) -> str:
        variants = [
            name,
            name.replace("Technology", "Tech"),
            name + " Pvt Ltd",
            name + " Inc.",
            name + " Corporation",
            name.replace("and", "&") if "and" in name.lower() else name + " LLC",
            name.upper(),
            name.replace(" ", "  "),
            name[: max(3, len(name) - 1)] + (name[-1] if name else ""),
        ]
        if rng.random() < 0.15:
            # small typo
            chars = list(name)
            i = rng.randrange(len(chars))
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            return "".join(chars)
        return rng.choice(variants)

    def noise_addr(street, city, region, pin, country_code) -> str:
        street_v = rng.choice(
            [
                street,
                street.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave"),
                street.replace("St", "Street") if "St" in street else street,
            ]
        )
        extra = ""
        if rng.random() < 0.25:
            extra = " Near SBI ATM" if country_code == "IN" else " Ste 200"
        if rng.random() < 0.2:
            return f"{street_v}, {city}{extra}"
        if country_code == "IN":
            return f"{street_v}, {city}, {region} {pin}{extra}"
        if country_code == "FR":
            return f"{street_v}, {pin} {city}{extra}"
        return f"{street_v}, {city}, {region} {pin}{extra}"

    def make_entity(i, prefix, stem, loc, country_label, country_code):
        street, city, region, pin = loc
        return {
            "entity_id": f"{prefix}-{i:05d}",
            "business_name": noise_name(stem),
            "business_address": noise_addr(street, city, region, pin, country_code),
            "country": country_label,
        }

    def write_tsv(path: Path, rows: list[dict]) -> None:
        pd.DataFrame(rows)[SOURCE_COLUMNS].to_csv(path, sep="\t", index=False)

    def build_split(n_s1: int, countries, include_france: bool, id_base: int):
        s1, s2, s3, gt_rows = [], [], [], []
        s2_i = id_base
        s3_i = id_base
        for i in range(n_s1):
            stem = stems[i % len(stems)] + (" " + str(i // len(stems)) if i >= len(stems) else "")
            if include_france and i % 5 == 0:
                country_label, code, loc = "France", "FR", rng.choice(fr_cities)
            elif i % 2 == 0:
                country_label, code, loc = rng.choice([("United States", "US"), ("US", "US")]), "US", rng.choice(us_cities)
                if isinstance(country_label, tuple):
                    country_label, code = country_label
            else:
                country_label, code, loc = rng.choice([("India", "IN"), ("IN", "IN")]), "IN", rng.choice(in_cities)
                if isinstance(country_label, tuple):
                    country_label, code = country_label

            rec1 = make_entity(id_base + i, "S1", stem, loc, country_label, code)
            s1.append(rec1)

            n_match = rng.choices([0, 1, 2, 3], weights=[0.28, 0.42, 0.22, 0.08])[0]
            matched = []
            for k in range(n_match):
                if k % 2 == 0:
                    s2_i += 1
                    rec = make_entity(s2_i, "S2", stem, loc, country_label, code)
                    s2.append(rec)
                    matched.append(rec["entity_id"])
                else:
                    s3_i += 1
                    rec = make_entity(s3_i, "S3", stem, loc, country_label, code)
                    s3.append(rec)
                    matched.append(rec["entity_id"])
            gt_rows.append(
                {
                    "source1_entity_id": rec1["entity_id"],
                    "matched_entity_ids": ",".join(matched),
                }
            )

        # extra unmatched S2/S3 distractors
        for j in range(max(40, n_s1 // 3)):
            stem = "Distractor " + rng.choice(stems)
            loc = rng.choice(us_cities + in_cities + (fr_cities if include_france else []))
            country_label = "France" if include_france and j % 7 == 0 else ("US" if j % 2 == 0 else "India")
            code = "FR" if country_label == "France" else ("US" if country_label == "US" else "IN")
            s2_i += 1
            s2.append(make_entity(s2_i, "S2", stem, loc, country_label, code))
            if j % 2 == 0:
                s3_i += 1
                s3.append(make_entity(s3_i, "S3", stem, loc, country_label, code))
        return s1, s2, s3, gt_rows

    tr_s1, tr_s2, tr_s3, tr_gt = build_split(220, None, include_france=False, id_base=1)
    te_s1, te_s2, te_s3, _ = build_split(120, None, include_france=True, id_base=5000)

    write_tsv(dataset_root / "train" / "train_source1.tsv", tr_s1)
    write_tsv(dataset_root / "train" / "train_source2.tsv", tr_s2)
    write_tsv(dataset_root / "train" / "train_source3.tsv", tr_s3)
    pd.DataFrame(tr_gt)[GT_COLUMNS].to_csv(
        dataset_root / "train" / "train_ground_truth.tsv", sep="\t", index=False
    )
    write_tsv(dataset_root / "test" / "test_source1.tsv", te_s1)
    write_tsv(dataset_root / "test" / "test_source2.tsv", te_s2)
    write_tsv(dataset_root / "test" / "test_source3.tsv", te_s3)
