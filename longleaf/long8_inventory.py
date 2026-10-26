"""Freeze the eight-task Hub file list and inspect the existing raw schema."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import h5py
from huggingface_hub import HfApi
from task_sampling import TASKS

root = Path(os.environ["WM_ROOT"])
out = root / "data/long8"
out.mkdir(parents=True, exist_ok=True)
api = HfApi()
source = "RoboDojo-Benchmark/RoboDojo"
target = out / "inventory.json"
if target.exists():
    print("INVENTORY_EXISTS", target, flush=True)
else:
    revision = api.dataset_info(source).sha
    tasks = {}
    for task in TASKS:
        entries = []
        prefix = f"data/RoboDojo/{task}/arx_x5"
        for item in api.list_repo_tree(source, path_in_repo=prefix, repo_type="dataset",
                                       revision=revision, recursive=True):
            if not hasattr(item, "size"):
                continue
            lfs = getattr(item, "lfs", None)
            checksum = (lfs.get("sha256") if isinstance(lfs, dict)
                        else getattr(lfs, "sha256", None))
            entries.append({"path": item.path, "bytes": item.size, "sha256": checksum})
        if not entries:
            raise ValueError(f"No source files for {task}")
        episodes = [entry for entry in entries if entry["path"].endswith(".hdf5")]
        tasks[task] = {"files": entries, "episodes": len(episodes),
                       "bytes": sum(entry["bytes"] for entry in entries),
                       "episode_bytes": sum(entry["bytes"] for entry in episodes)}
        print("TASK_INVENTORY", task, tasks[task]["episodes"], tasks[task]["bytes"], flush=True)
    record = {"schema_version": 1, "source_repo": source, "source_revision": revision,
              "tasks": tasks, "bytes": sum(task["bytes"] for task in tasks.values())}
    target.write_text(json.dumps(record, indent=2) + "\n")
    print("INVENTORY_READY", json.dumps({k: v for k, v in record.items() if k != "tasks"}), flush=True)

pilot = root / "data/raw/data/RoboDojo/organize_table/arx_x5/data/episode_0000000.hdf5"
with h5py.File(pilot, "r") as f:
    structure = {"root_attrs": {k: str(v)[:2000] for k, v in f.attrs.items()}, "datasets": {}}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            structure["datasets"][name] = {"shape": obj.shape, "dtype": str(obj.dtype),
                "attrs": {k: str(v)[:500] for k, v in obj.attrs.items()}}
            if obj.size < 20 or name == "instructions":
                try:
                    structure["datasets"][name]["sample"] = str(obj[()])[:2000]
                except Exception as exc:
                    structure["datasets"][name]["sample_error"] = str(exc)
    f.visititems(visit)
(out / "pilot_schema.json").write_text(json.dumps(structure, indent=2) + "\n")
print("PILOT_SCHEMA_READY", flush=True)
