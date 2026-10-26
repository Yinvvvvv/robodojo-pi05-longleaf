"""Accept a completed data manifest only after recorded inspection of all 40 samples."""
import argparse
import json
from pathlib import Path
import os

from download_long8 import sha256
from task_sampling import TASKS, dataset_sources


def check_review(report, reviews):
    expected = set(report["sampled_source_files"])
    if {entry["source_file"] for entry in reviews} != expected or len(reviews) != len(expected):
        raise ValueError("Visual review must cover exactly the fixed sampled episodes")
    episodes = {entry["source_file"]: entry for entry in report["episodes"]}
    for review in reviews:
        if review["status"] not in {"pass", "pass_with_documented_limitations"}:
            raise ValueError("An unresolved visual issue prevents acceptance")
        if len(review.get("notes", "")) < 20:
            raise ValueError("Record concrete visual observations, not only a pass flag")
        path = Path(episodes[review["source_file"]]["visual_contact_sheet"])
        if sha256(path) != review["contact_sheet_sha256"]:
            raise ValueError("Reviewed contact sheet hash mismatch")


def prepare_training_storage(root, reports):
    import subprocess
    import pyarrow.parquet as pq

    def size(path):
        return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])

    kept = []
    cache_bytes = 0
    for task in TASKS:
        report = reports[task]
        repo = ("RoboDojo-organize_table_only-arx_x5-joint" if task == "organize_table"
                else f"RoboDojo-{task}-arx_x5-joint")
        path = root / "data/lerobot" / repo
        if path.is_symlink() or Path(report["dataset_root"]).resolve() != path.resolve():
            raise ValueError("Unexpected native multi-dataset source path")
        info = json.loads((path / "meta/info.json").read_text())
        if info["total_episodes"] != 100 or info["total_frames"] != report["total_frames"]:
            raise ValueError("All eight completed native source datasets must be retained")
        files = sorted((path / "data").rglob("*.parquet"))
        if not files:
            raise ValueError("Completed dataset has no Parquet files")
        for file in files:
            metadata = pq.read_metadata(file)
            cache_bytes += sum(metadata.row_group(i).column(j).total_uncompressed_size
                               for i in range(metadata.num_row_groups)
                               for j in range(metadata.num_columns))
        kept.append(path)
    # ST reads the same organize source/cache that native MultiLeRobotDataset reuses.
    cache_bytes = int(cache_bytes * 1.05)
    used = size(root)
    reserve = 250_000_000_000
    projected = used + cache_bytes + reserve
    if projected > 1_900_000_000_000:
        raise ValueError(f"Training cache/checkpoint budget exceeds 1.9TB: {projected}")
    cache = root / "data/hf_datasets_cache"
    cache.mkdir(parents=True, exist_ok=True)
    plan = {"project_bytes_before_training": used, "estimated_unique_source_arrow_cache_bytes": cache_bytes,
            "checkpoint_and_work_reserve_bytes": reserve, "projected_bytes": projected,
            "limit_bytes": 1_900_000_000_000, "cache_root": str(cache),
            "retained_datasets": [str(path) for path in kept], "raw_sources_retained": True,
            "deleted_datasets": [], "storage_format": "lerobot_native_multi_dataset",
            "reason": "All8 native source datasets are required; ST shares organize source with MT. No image merge or source deletion."}
    (root / "data/long8/training_storage_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def storage_self_check():
    import tempfile
    import subprocess
    from unittest.mock import patch
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data/long8").mkdir(parents=True)
        raw = root / "data/raw/source.hdf5"
        raw.parent.mkdir(); raw.write_bytes(b"raw-must-survive")
        reports = {}
        for task in TASKS:
            repo = ("RoboDojo-organize_table_only-arx_x5-joint" if task == "organize_table"
                    else f"RoboDojo-{task}-arx_x5-joint")
            path = root / "data/lerobot" / repo
            (path / "data").mkdir(parents=True)
            (path / "meta").mkdir()
            reports[task] = {"dataset_root": str(path), "total_frames": 100}
            (path / "meta/info.json").write_text(json.dumps({"total_episodes": 100, "total_frames": 100}))
            pq.write_table(pa.table({"image": [b"png"] * 100}), path / "data/file.parquet")
        original = subprocess.check_output
        def excessive(cmd, **kwargs):
            return "2000000000000 fixture" if cmd[-1] == str(root) else original(cmd, **kwargs)
        with patch("subprocess.check_output", side_effect=excessive):
            try:
                prepare_training_storage(root, reports)
            except ValueError as error:
                assert "exceeds 1.9TB" in str(error)
            else:
                raise AssertionError("Capacity gate did not reject oversized training cache")
        bad = {k: dict(v) for k, v in reports.items()}
        bad["fill_pen_holder"]["dataset_root"] = str(raw.parent)
        try:
            prepare_training_storage(root, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected dataset source path accepted")
        plan = prepare_training_storage(root, reports)
        assert raw.read_bytes() == b"raw-must-survive"
        assert len(plan["retained_datasets"]) == 8 and plan["deleted_datasets"] == []
        assert all(Path(v["dataset_root"]).exists() for v in reports.values())
    print("TRAINING_STORAGE_GUARD_AND_ALL8_RETENTION_CHECK_PASSED")


def main():
    root = Path(os.environ["WM_ROOT"])
    folder = root / "data/long8"
    reviews_path = folder / "visual_reviews.json"
    reviews = json.loads(reviews_path.read_text())
    reports = {}
    for task in TASKS:
        reports[task] = json.loads((folder / f"qc/{task}.json").read_text())
        if len(reports[task]["sampled_source_files"]) != 5:
            raise ValueError("Need the five frozen random samples per task")
        check_review(reports[task], reviews[task])
    manifest = json.loads((root / "data/baseline_manifest.pending.json").read_text())
    composite = manifest["datasets"]["RoboDojo-long_horizon8-arx_x5-joint"]
    if (composite.get("storage_format") != "lerobot_native_multi_dataset"
            or composite.get("source_datasets") != dataset_sources(reports)
            or len(composite["episodes"]) != 800):
        raise ValueError("All800 native multi-dataset source mappings must match verified QC")
    for repo, dataset in manifest["datasets"].items():
        norm = root / "assets/pi05_robodojo_longleaf_full" / repo / "norm_stats.json"
        if sha256(norm) != dataset["norm_stats_sha256"]:
            raise ValueError("Normalization changed after checks")
        group_splits = {}
        for episode in dataset["episodes"].values():
            group = episode["split_group"]
            if group in group_splits and group_splits[group] != episode["split"]:
                raise ValueError("Scene/episode group leakage across splits")
            group_splits[group] = episode["split"]
        if set(group_splits.values()) != {"train", "val", "test"}:
            raise ValueError("A fixed split is missing")
    manifest["training_storage_plan"] = prepare_training_storage(root, reports)
    manifest["qc_status"] = "accepted"
    manifest["visual_review_sha256"] = sha256(reviews_path)
    manifest["visual_review_samples"] = 40
    manifest["qc_acceptance_scope"] = "All automated checks and 5 visual samples per task; timestamp, layout and terminal-label limitations remain explicitly documented."
    target = root / "data/baseline_manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    for task, report in reports.items():
        reviewed = set(report["sampled_source_files"])
        for episode in report["episodes"]:
            episode["visual_review"] = ("reviewed_see_visual_reviews_json"
                if episode["source_file"] in reviewed else "not_selected_for_visual_review")
        report["status"] = "automated_and_sampled_visual_checks_complete_with_documented_limitations"
        (folder / f"qc/{task}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LONG8_DATA_REVIEW_ACCEPTED", target, flush=True)


def self_check():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/"contact_sheet"
        path.write_bytes(b"test-fixture-only")
        report = {"sampled_source_files": ["episode"], "episodes": [
            {"source_file": "episode", "visual_contact_sheet": str(path)}]}
        reviews = [{"source_file": "episode", "status": "pass_with_documented_limitations",
                    "notes": "Test of review binding and required coverage only.",
                    "contact_sheet_sha256": sha256(path)}]
        check_review(report, reviews)
        try:
            check_review(report, [])
        except ValueError:
            pass
        else:
            raise AssertionError("Missing reviews were accepted")
        reviews[0]["contact_sheet_sha256"] = "wrong"
        try:
            check_review(report, reviews)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed contact sheet was accepted")
    print("VISUAL_REVIEW_BINDING_CHECK_PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--storage-self-check", action="store_true")
    args = parser.parse_args()
    if args.storage_self_check:
        storage_self_check()
    else:
        self_check() if args.self_check else main()
