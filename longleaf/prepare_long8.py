"""Audit and losslessly convert frozen RoboDojo episodes with native LeRobot APIs."""
import argparse
from collections import Counter
import hashlib
import datasets
import json
import os
from pathlib import Path
import random
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.aggregate import aggregate_datasets
from XPolicyLab.utils.process_data import decode_image_bit
from download_long8 import sha256
from task_sampling import TASKS

ROOT = Path(os.environ["WM_ROOT"])
WORK = ROOT / "data/long8"
CAMERAS = {"cam_head": "cam_high", "cam_left_wrist": "cam_left_wrist",
           "cam_right_wrist": "cam_right_wrist"}
ST_REPO = "RoboDojo-organize_table_only-arx_x5-joint"
MT_REPO = "RoboDojo-long_horizon8-arx_x5-joint"


def merge(**kwargs):
    # HF from_parquet writes a builder cache even with keep_in_memory=True.
    # Direct Arrow reads preserve the file's HF feature metadata without that cache.
    original = datasets.Dataset.from_parquet
    def read_parquet(path, filters=None, features=None):
        return datasets.Dataset(pq.read_table(path, filters=filters),
            info=datasets.DatasetInfo(features=features))
    datasets.Dataset.from_parquet = read_parquet
    try:
        aggregate_datasets(**kwargs)
    finally:
        datasets.Dataset.from_parquet = original


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def joints(group):
    return np.concatenate([np.asarray(group["left_arm_joint_states"]),
        np.asarray(group["left_ee_joint_states"]).reshape(-1, 1),
        np.asarray(group["right_arm_joint_states"]),
        np.asarray(group["right_ee_joint_states"]).reshape(-1, 1)], axis=1).astype(np.float32)


def language(f):
    for name in ["instruction", "instructions"]:
        if name not in f:
            continue
        value = f[name][()]
        values = np.asarray(value).reshape(-1)
        texts = [(item.decode("utf-8") if isinstance(item, bytes) else str(item)).strip()
                 for item in values]
        texts = [text for text in texts if text]
        if texts:
            return texts[0], name, texts
    raise ValueError("Source instruction missing or empty; no generated fallback allowed")


def metadata(f, task, source):
    state = joints(f["state"])
    raw_action = joints(f["action"])
    if state.ndim != 2 or state.shape[1] != 14 or len(state) < 50:
        raise ValueError(f"Invalid state shape: {state.shape}")
    if raw_action.shape != state.shape or not np.isfinite(state).all() or not np.isfinite(raw_action).all():
        raise ValueError("Invalid action shape or non-finite robot values")
    if not np.any(np.ptp(state, axis=0) > 1e-6):
        raise ValueError("Episode has no detectable robot motion")
    marker = str(f.attrs.get('fix_obs_shift_rgb', ''))
    if marker != 'shift_drop_first_drop_last+rgb_jpeg':
        raise ValueError(f'Unrecognized observation/RGB correction marker: {marker!r}')
    prompt, prompt_key, instructions = language(f)
    fps = int(np.asarray(f["additional_info/frequency"])[()])
    if fps <= 0:
        raise ValueError("Invalid recorded frequency")
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    for camera in CAMERAS:
        if len(f[f"vision/{camera}/colors"]) != len(state):
            raise ValueError("Camera/state frame count mismatch")
    first = decode_image_bit(f["vision/cam_head/colors"][0])
    first_hash = hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()
    layout = None
    for key in ["layout_id", "layout_seed", "seed",
                "additional_info/layout_id", "additional_info/layout_seed", "additional_info/seed"]:
        if key in f and isinstance(f[key], h5py.Dataset) and f[key].size == 1:
            layout = f"{key}:{str(f[key][()])}"
            break
        if key in f.attrs:
            layout = f"{key}:{str(f.attrs[key])}"
            break
    timestamps = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and ("timestamp" in name.lower() or name.lower().endswith("/time")):
            values = np.asarray(obj).reshape(-1)
            timestamps[name] = {"count": len(values), "finite": bool(np.isfinite(values).all()),
                "strictly_increasing": bool(np.all(np.diff(values) > 0))}
    f.visititems(visit)
    if any(not v["finite"] or not v["strictly_increasing"] for v in timestamps.values()):
        raise ValueError("Invalid source timestamps")
    motion = np.linalg.norm(np.diff(state, axis=0), axis=1)
    report = {"task": task, "source_file": source["path"], "source_sha256": source["sha256"],
        "frames": len(state), "state_dim": 14, "action_dim": 14, "fps": fps,
        "prompt": prompt, "prompt_field": prompt_key, "source_instructions": instructions,
        "state_min": state.min(axis=0).tolist(), "state_max": state.max(axis=0).tolist(),
        "action_convention": "Pinned XPolicyLab next-state targets; final state repeated",
        "raw_action_vs_state_mae": float(np.mean(np.abs(raw_action - state))),
        "raw_action_vs_next_state_mae": float(np.mean(np.abs(raw_action - action))),
        "raw_action_vs_next_state_max_abs": float(np.max(np.abs(raw_action - action))),
        "motion_median": float(np.median(motion)), "tail_motion_mean": float(motion[-10:].mean()),
        "layout_metadata": layout, "initial_head_rgb_sha256": first_hash,
        "split_group": f"{task}/{layout or first_hash}",
        "split_group_basis": "source_layout_or_seed" if layout else "initial_head_exact_rgb_hash",
        "timestamps": timestamps, "temporal_check": "source_timestamps_checked" if timestamps
            else "no_source_timestamps; indexed frames and correction marker checked",
        "source_attrs": {key: str(value) for key, value in f.attrs.items()},
        "completion_check": "no_explicit_terminal_flag_verified; inspect sampled final frames",
        "visual_review": "pending", "decoded_frames": 0}
    return state, action, report


def convert_episode(task, source, sampled):
    stem = Path(source["path"]).stem
    output = WORK / "work/episodes" / task / stem
    receipt = output / "conversion.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["source_sha256"] != source["sha256"]:
            raise ValueError("Existing conversion has a different source hash")
        return old, output
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        # Only our uncommitted derived directory is discarded; raw source is retained.
        shutil.rmtree(partial)
    path = ROOT / "data/raw" / source["path"]
    with h5py.File(path, "r") as f:
        state, action, report = metadata(f, task, source)
        features = {key: {"dtype": "float32", "shape": (14,), "names": None}
                    for key in ["observation.state", "action"]}
        for camera in CAMERAS.values():
            features[f"observation.images.{camera}"] = {
                "dtype": "image", "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]}
        dataset = LeRobotDataset.create(repo_id=f"long8-{task}-{stem}", root=partial,
            fps=report["fps"], robot_type="arx_x5", features=features,
            use_videos=False, image_writer_threads=4)
        chosen = sorted({0, len(state) // 2, len(state) - 1})
        images_for_review = {}
        dark_frames = Counter()
        try:
            for i in range(len(state)):
                frame = {"observation.state": state[i], "action": action[i], "task": report["prompt"]}
                for src, dest in CAMERAS.items():
                    image = decode_image_bit(f[f"vision/{src}/colors"][i])
                    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                        raise ValueError(f"Invalid RGB frame {src}:{i} {image.shape} {image.dtype}")
                    frame[f"observation.images.{dest}"] = image
                    if float(image[::16, ::16].std()) < 1:
                        dark_frames[src] += 1
                    if sampled and i in chosen:
                        images_for_review[i, src] = image.copy()
                dataset.add_frame(frame)
                report["decoded_frames"] += 1
            dataset.save_episode()
        finally:
            dataset.finalize()
            dataset.stop_image_writer()
        info = json.loads((partial / "meta/info.json").read_text())
        if info["total_episodes"] != 1 or info["total_frames"] != len(state):
            raise ValueError("LeRobot written count mismatch")
        report["low_texture_frame_counts"] = dict(dark_frames)
        if sampled:
            canvas = Image.new("RGB", (960, 3 * 264), "white")
            draw = ImageDraw.Draw(canvas)
            for row, index in enumerate(chosen):
                for col, camera in enumerate(CAMERAS):
                    tile = Image.fromarray(images_for_review[index, camera]).resize((320, 240))
                    canvas.paste(tile, (col * 320, row * 264 + 24))
                    draw.text((col * 320 + 4, row * 264 + 4), f"{camera} frame {index}", fill="black")
            target = WORK / "qc/visual" / task / f"{stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target)
            report["visual_contact_sheet"] = str(target)
        write_json(partial / "conversion.json", report)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(output)
    print("EPISODE_CONVERTED", task, stem, len(state), flush=True)
    return report, output


def task_repo(task):
    return ST_REPO if task == "organize_table" else f"RoboDojo-{task}-arx_x5-joint"


def prepare_task(task, smoke=False):
    inventory = json.loads((WORK / "inventory.json").read_text())
    if not smoke and not (WORK / f"download-{task}.json").exists():
        raise ValueError("Verified task download is incomplete")
    sources = sorted([f for f in inventory["tasks"][task]["files"] if f["path"].endswith(".hdf5")],
                     key=lambda f: f["path"])
    if not smoke and (WORK / f"qc/{task}.json").exists():
        old = json.loads((WORK / f"qc/{task}.json").read_text())
        meta = LeRobotDatasetMetadata(task_repo(task), root=Path(old["dataset_root"]))
        if (old["source_revision"] == inventory["source_revision"] and old["total_episodes"] == 100
                and meta.total_episodes == 100 and meta.total_frames == old["total_frames"]):
            print("TASK_ALREADY_CONVERTED", task, flush=True)
            return
        raise ValueError("Existing task output is inconsistent; inspect before restarting")
    sampled = set(random.Random(20261021).sample([f["path"] for f in sources], 5))
    if smoke:
        sources = [next(f for f in sources if f["path"].endswith("episode_0000000.hdf5"))]
        if sha256(ROOT / "data/raw" / sources[0]["path"]) != sources[0]["sha256"]:
            raise ValueError("Smoke source checksum mismatch")
    reports, roots = [], []
    for source in sources:
        report, path = convert_episode(task, source, smoke or source["path"] in sampled)
        reports.append(report)
        roots.append(path)
    if len({entry["fps"] for entry in reports}) != 1:
        raise ValueError("Mixed recorded frequencies within task")
    destination = (WORK / "work/smoke-aggregate" if smoke
                   else Path(os.environ["HF_LEROBOT_HOME"]) / task_repo(task))
    partial = destination.with_name(destination.name + ".partial")
    if not destination.exists():
        if partial.exists():
            shutil.rmtree(partial)
        merge(repo_ids=[f"long8-{task}-{Path(f['source_file']).stem}" for f in reports],
            roots=roots, aggr_repo_id=task_repo(task), aggr_root=partial)
        meta = LeRobotDatasetMetadata(task_repo(task), root=partial)
        if meta.total_episodes != len(reports) or meta.total_frames != sum(r["frames"] for r in reports):
            raise ValueError("Aggregated episode or frame count mismatch")
        partial.replace(destination)
    summary = {"task": task, "repo_id": task_repo(task), "episodes": reports,
        "total_episodes": len(reports), "total_frames": sum(r["frames"] for r in reports),
        "source_revision": inventory["source_revision"], "status": "automated_checks_passed_visual_pending",
        "sampled_source_files": sorted(sampled), "dataset_root": str(destination)}
    write_json(WORK / ("smoke.json" if smoke else f"qc/{task}.json"), summary)
    print("TASK_CONVERSION_COMPLETE", task, len(reports), summary["total_frames"], flush=True)
    if not smoke:
        # Native aggregate preserves the data; verified sources + QC records remain.
        for path in roots:
            shutil.rmtree(path)


def self_check():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "episode.hdf5"
        values = np.arange(60 * 14, dtype=np.float32).reshape(60, 14) / 100
        with h5py.File(path, "w") as f:
            for group in ["state", "action"]:
                f[f"{group}/left_arm_joint_states"] = values[:, :6]
                f[f"{group}/left_ee_joint_states"] = values[:, 6:7]
                f[f"{group}/right_arm_joint_states"] = values[:, 7:13]
                f[f"{group}/right_ee_joint_states"] = values[:, 13:14]
            np.testing.assert_array_equal(joints(f["state"]), values)
            f["instruction"] = np.bytes_("Source instruction, not a generated prompt.")
            assert language(f)[0] == "Source instruction, not a generated prompt."
            del f["instruction"]
            try:
                language(f)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing source language was accepted")
    print("LONG8_NUMERIC_AND_LANGUAGE_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--merge-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.merge_check:
        source = WORK / 'work/smoke-aggregate'
        target = WORK / 'work/cache-merge-check'
        if target.exists():
            shutil.rmtree(target)
        merge(repo_ids=['long8-organize_table-episode_0000000'], roots=[source],
              aggr_repo_id='long8-cache-check', aggr_root=target)
        meta = json.loads((target / 'meta/info.json').read_text())
        assert meta['total_episodes'] == 1 and meta['total_frames'] == 838
        source_table = pa.concat_tables([pq.read_table(p) for p in sorted((source / 'data').rglob('*.parquet'))])
        target_table = pa.concat_tables([pq.read_table(p) for p in sorted((target / 'data').rglob('*.parquet'))])
        assert source_table.equals(target_table, check_metadata=False), 'Merged frame/image bytes changed'
        source_features = json.loads(source_table.schema.metadata[b'huggingface'])['info']['features']
        target_features = json.loads(target_table.schema.metadata[b'huggingface'])['info']['features']
        assert source_features == target_features, 'HF feature metadata changed'
        cache_files = list(Path(os.environ['HF_DATASETS_CACHE']).rglob('*.arrow'))
        assert not cache_files, f'Unexpected HF Arrow builder cache: {cache_files}'
        write_json(WORK / 'cache_merge_check.json', {'episodes': 1, 'frames': 838, 'status': 'passed',
            'frame_and_image_bytes_identical': True, 'hf_feature_metadata_identical': True,
            'arrow_cache_files': len(cache_files), 'job_id': os.environ.get('SLURM_JOB_ID')})
        shutil.rmtree(target)
        print('NATIVE_IN_MEMORY_AGGREGATE_CHECK_PASSED', flush=True)
    else:
        prepare_task(args.task or ("organize_table" if args.smoke
                     else TASKS[int(os.environ["SLURM_ARRAY_TASK_ID"])]), smoke=args.smoke)
