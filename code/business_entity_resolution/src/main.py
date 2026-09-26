"""End-to-end business entity resolution pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking import candidates_for_sources
from data_loader import load_split, pick_noisy_examples, print_profile, profile_sources
from predict import predict_split, write_outputs
from train import train_pipeline
from utils import set_seed, workspace_paths


def main() -> None:
    set_seed(42)
    data = load_split()
    train_frames, gt, test_frames = data["train"], data["gt"], data["test"]
    paths = data["paths"] if "paths" in data else workspace_paths()

    train_stats = profile_sources("train", train_frames, gt)
    noisy_names, noisy_addrs = pick_noisy_examples(train_frames)
    print_profile(train_stats, noisy_names, noisy_addrs)

    test_stats = profile_sources("test", test_frames)
    t_names, t_addrs = pick_noisy_examples(test_frames)
    print_profile(test_stats, t_names, t_addrs)

    print("\nTraining pairwise matcher...")
    trained = train_pipeline(train_frames, gt)
    print("\nVALIDATION")
    for k, v in trained.val_metrics.items():
        print(f"  {k}: {v}")
    print("\nBLOCKING")
    for k, v in trained.blocking_stats.items():
        if k == "feature_names":
            print(f"  features ({len(v)}): {v}")
        else:
            print(f"  {k}: {v}")

    print("\nScoring test candidates...")
    candidate_map, pred_map = predict_split(test_frames, trained)
    s1_ids = list(test_frames["source1"]["entity_id"])
    matching_path, candidate_path = write_outputs(paths["output_dir"], s1_ids, candidate_map, pred_map)

    n_matches = sum(len(v) for v in pred_map.values())
    n_singletons = sum(1 for v in pred_map.values() if not v)
    n_cands = sum(len(v) for v in candidate_map.values())
    print(f"\nWrote {matching_path}")
    print(f"Wrote {candidate_path}")
    print(f"Test predicted links: {n_matches}")
    print(f"Test predicted singletons: {n_singletons} / {len(s1_ids)}")
    print(f"Test candidate pairs: {n_cands}")

    validator = paths["root"] / "utils" / "validate_submission.py"
    if not validator.exists():
        # Fall back to the copy bundled with this package.
        validator = SRC_DIR.parent.parent.parent / "utils" / "validate_submission.py"
    if validator.exists():
        import subprocess

        cmd = [
            sys.executable,
            str(validator),
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
            "--test-dir",
            str(paths["test_dir"]),
        ]
        print("\nRunning validator:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    else:
        print("Validator script not found; skipped.")

    # Training-set sanity: how the chosen rule behaves on a fresh blocking pass.
    train_cands = candidates_for_sources(
        train_frames["source1"], train_frames["source2"], train_frames["source3"]
    )
    print(f"\nTrain candidate pairs (full): {sum(len(v) for v in train_cands.values())}")
    print("Done.")


if __name__ == "__main__":
    main()
