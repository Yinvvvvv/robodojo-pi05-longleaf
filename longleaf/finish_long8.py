"""Merge eight tasks, freeze episode splits and compute numeric-only train normalization."""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from openpi import transforms
from openpi.shared import normalize
from openpi.training import config
from prepare_long8 import ROOT, WORK, ST_REPO, MT_REPO, task_repo, write_json, merge
from task_sampling import TASKS, frame_weights, dataset_sources
from download_long8 import sha256


def splits(reports):
    assignments = {}
    for task, report in reports.items():
        groups = sorted({e["split_group"] for e in report["episodes"]},
            key=lambda key: hashlib.sha256(("long8-split-20261021/" + key).encode()).hexdigest())
        if len(groups) < 10:
            raise ValueError(f"Too few independent episode/layout groups for {task}: {len(groups)}")
        heldout = max(1, round(len(groups) * .1))
        labels = {key: ("test" if i < heldout else "val" if i < heldout * 2 else "train")
                  for i, key in enumerate(groups)}
        for episode in report["episodes"]:
            assignments[episode["source_file"]] = labels[episode["split_group"]]
    return assignments


def transform():
    base = config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
    # Keep exactly the pinned model/data transforms; only image loading is avoided.
    factory = dataclasses.replace(base.data, assets=config.AssetsConfig(asset_id="long8-numeric-norm"))
    data_config = factory.create(ROOT / "assets", base.model)
    return transforms.compose(data_config.data_transforms.inputs), base.model.action_horizon


def windows(actions, start, stop, horizon):
    indices = np.minimum(np.arange(start, stop)[:, None] + np.arange(horizon), len(actions) - 1)
    return actions[indices]


def transformed(fn, state, actions):
    batch = len(state)
    return fn({"state": state, "actions": actions,
        "images": {key: np.zeros((batch, 3, 2, 2), np.uint8)
                   for key in ["cam_high", "cam_left_wrist", "cam_right_wrist"]}})


def numeric_table(root):
    import pyarrow.compute as pc
    roots = root if isinstance(root, list) else [root]
    tables = []
    offset = 0
    for source in roots:
        for path in sorted((source / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
            position = table.schema.get_field_index("episode_index")
            table = table.set_column(position, "episode_index", pc.add(table["episode_index"], offset))
            tables.append(table)
        offset += json.loads((source / "meta/info.json").read_text())["total_episodes"]
    if not tables:
        raise ValueError("No numeric parquet data")
    import pyarrow as pa
    table = pa.concat_tables(tables)
    return (np.array(table["episode_index"].to_pylist(), dtype=np.int64),
            np.array(table["frame_index"].to_pylist(), dtype=np.int64),
            np.array(table["observation.state"].to_pylist(), dtype=np.float32),
            np.array(table["action"].to_pylist(), dtype=np.float32))


def norm(repo_id, episodes, dataset_root):
    ep_ids, frame_ids, state, action = numeric_table(dataset_root)
    fn, horizon = transform()
    running = {key: normalize.RunningStats() for key in ["state", "actions"]}
    train_frames = action_vectors = 0
    for episode in sorted(np.unique(ep_ids)):
        entry = episodes[str(int(episode))]
        if entry["split"] != "train":
            continue
        rows = np.flatnonzero(ep_ids == episode)
        rows = rows[np.argsort(frame_ids[rows])]
        np.testing.assert_array_equal(frame_ids[rows], np.arange(len(rows)))
        states, actions = state[rows], action[rows]
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            data = transformed(fn, states[start:stop], windows(actions, start, stop, horizon))
            for key in running:
                running[key].update(np.asarray(data[key]))
            train_frames += stop - start
            action_vectors += (stop - start) * horizon
    result = {key: stats.get_statistics() for key, stats in running.items()}
    output = ROOT / "assets/pi05_robodojo_longleaf_full" / repo_id
    normalize.save(output, result)
    print("TRAIN_ONLY_NORM_COMPLETE", repo_id, train_frames, action_vectors, flush=True)
    return {"norm_source_split": "train", "norm_stats_sha256": sha256(output / "norm_stats.json"),
        "norm_train_frames": train_frames, "norm_action_vectors_including_horizon_padding": action_vectors,
        "norm_method": "all train frames; pinned data transforms and RunningStats; clamped within-episode action windows; images omitted only because numeric transforms are independent"}


def main():
    reports = {task: json.loads((WORK / f"qc/{task}.json").read_text()) for task in TASKS}
    if any(r["total_episodes"] != 100 for r in reports.values()):
        raise ValueError("Every task must contain 100 converted episodes")
    assignments = splits(reports)
    sources = dataset_sources(reports)
    infos = [json.loads((Path(source["root"]) / "meta/info.json").read_text()) for source in sources]
    for source, info in zip(sources, infos, strict=True):
        if (info["total_episodes"] != 100 or info["total_frames"] != source["frames"]
                or info["fps"] != infos[0]["fps"] or info["features"] != infos[0]["features"]):
            raise ValueError("Native multi-dataset source metadata mismatch")
    print("NATIVE_MULTI_DATASET_READY", len(sources), sum(s["frames"] for s in sources), flush=True)
    datasets = {ST_REPO: {"episodes": {}}, MT_REPO: {"episodes": {}}}
    datasets[MT_REPO].update({"storage_format": "lerobot_native_multi_dataset",
                             "source_datasets": sources, "fps": infos[0]["fps"]})
    global_index = 0
    for task in TASKS:
        for local_index, episode in enumerate(reports[task]["episodes"]):
            entry = {key: episode[key] for key in ["task", "source_file", "source_sha256", "frames",
                "fps", "split_group", "split_group_basis", "layout_metadata"]}
            entry["split"] = assignments[episode["source_file"]]
            datasets[MT_REPO]["episodes"][str(global_index)] = entry
            if task == "organize_table":
                datasets[ST_REPO]["episodes"][str(local_index)] = entry
            global_index += 1
    for repo_id, entry in datasets.items():
        root = ([Path(source["root"]) for source in sources] if repo_id == MT_REPO
                else Path(os.environ["HF_LEROBOT_HOME"]) / repo_id)
        # Validate fixed task mass and ensure held-out frames can never be sampled.
        frame_ids = [int(i) for i, episode in entry["episodes"].items() for _ in range(episode["frames"])]
        for setting in (["ST-Organize"] if repo_id == ST_REPO else ["MT-Uniform", "MT-Organize50"]):
            weights = frame_weights(frame_ids, entry["episodes"], setting)
            if not np.isclose(sum(weights), 1.):
                raise ValueError("Task sampling mass mismatch")
        entry.update(norm(repo_id, entry["episodes"], root))
    inventory = json.loads((WORK / "inventory.json").read_text())
    manifest = {"schema_version": 1, "qc_status": "pending_visual_review",
        "source_repo": inventory["source_repo"], "source_revision": inventory["source_revision"],
        "inventory_sha256": sha256(WORK / "inventory.json"),
        "split_seed": 20261021, "datasets": datasets,
        "limitations": ["Missing layout metadata falls back to episode groups, additionally grouping identical initial head RGB.",
            "No source timestamps: frame counts, source alignment-fix marker, and sample motion are evidence, not independent timestamp verification.",
            "No verified explicit terminal flag; sampled endings and prompts require visual inspection.",
            "Source-recorded FPS is used; legacy adapter default 50Hz is not copied.",
            "Action targets follow pinned next-state adapter; native raw actions are retained and compared in QC."]}
    write_json(ROOT / "data/baseline_manifest.pending.json", manifest)
    write_json(WORK / "automated_summary.json", {
        "episodes": 800, "frames": sum(r["total_frames"] for r in reports.values()),
        "status": "conversion_and_numeric_checks_complete_visual_review_pending",
        "manifest": str(ROOT / "data/baseline_manifest.pending.json")})
    print("LONG8_AUTOMATED_PREPARATION_COMPLETE_VISUAL_REVIEW_PENDING", flush=True)


def self_check(smoke_root):
    example = {task: {'episodes': [{'source_file': f'{task}/{i}', 'split_group': f'{task}/{i//2}'}
                 for i in range(40)]} for task in ['a', 'b']}
    assigned = splits(example)
    assert set(assigned.values()) == {'train', 'val', 'test'}
    assert assigned == splits(example)
    for task in example:
        for i in range(0, 40, 2):
            assert assigned[f'{task}/{i}'] == assigned[f'{task}/{i+1}']
    fn, horizon = transform()
    actions = np.arange(60 * 14, dtype=np.float32).reshape(60, 14) / 100
    chunk = windows(actions, 58, 60, horizon)
    np.testing.assert_array_equal(chunk[1], np.repeat(actions[-1:], horizon, axis=0))
    # Compare numeric-only path against real LeRobot action-horizon lookup and real RGB transforms.
    meta = LeRobotDatasetMetadata(ST_REPO, root=smoke_root)
    loaded = LeRobotDataset(ST_REPO, root=smoke_root,
        delta_timestamps={"action": [i / meta.fps for i in range(horizon)]})
    ids, frames, states, targets = numeric_table(smoke_root)
    for index in [0, len(states)//2, len(states)-1]:
        item = loaded[index]
        import h5py
        from XPolicyLab.utils.process_data import decode_image_bit
        source = ROOT / 'data/raw/data/RoboDojo/organize_table/arx_x5/data/episode_0000000.hdf5'
        with h5py.File(source, 'r') as raw:
            for src, dest in {'cam_head': 'cam_high', 'cam_left_wrist': 'cam_left_wrist',
                              'cam_right_wrist': 'cam_right_wrist'}.items():
                expected = decode_image_bit(raw[f'vision/{src}/colors'][index])
                actual = np.asarray(item[f'observation.images.{dest}'])
                if actual.shape[0] == 3:
                    actual = np.moveaxis(actual, 0, -1)
                if actual.dtype.kind == 'f':
                    actual = np.rint(actual * 255).astype(np.uint8)
                np.testing.assert_array_equal(actual, expected)
        real = fn({"state": np.asarray(item["observation.state"]),
            "actions": np.asarray(item["action"]),
            "images": {name: np.asarray(item[f"observation.images.{name}"])
                       for name in ["cam_high", "cam_left_wrist", "cam_right_wrist"]}})
        numeric = transformed(fn, states[index:index+1], windows(targets, index, index+1, horizon))
        for key in ["state", "actions"]:
            np.testing.assert_allclose(numeric[key][0], real[key], rtol=1e-6, atol=1e-6)
    print("NUMERIC_ONLY_NORM_VS_REAL_LEROBOT_CHECK_PASSED", flush=True)
    print("RGB_PIXEL_ROUNDTRIP_AND_GROUP_SPLIT_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(WORK / "work/smoke-aggregate")
    else:
        main()
