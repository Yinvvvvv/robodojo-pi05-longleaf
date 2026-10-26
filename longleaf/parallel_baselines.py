"""Keep each parallel baseline owned by one Slurm job."""
import hashlib
import json
from pathlib import Path
import time

SETTINGS = ("ST-Organize", "MT-Uniform", "MT-Organize50")


def training_protocol(root):
    path = root / "longleaf/training-protocol.json"
    if not path.exists():
        return {"protocol_id": "legacy-full-60000", "optimizer_updates": 60000,
                "effective_batch_size": 256, "warmup_steps": 1000,
                "decay_steps": 30000, "peak_lr": 2.5e-5, "decay_lr": 2.5e-6,
                "validation_checkpoint_labels": []}
    protocol = json.loads(path.read_text())
    if (protocol.get("schema_version") != 1 or protocol.get("mode") != "full"
            or protocol.get("effective_batch_size") != 256
            or type(protocol.get("optimizer_updates")) is not int
            or type(protocol.get("warmup_steps")) is not int
            or not 0 < protocol["warmup_steps"] < protocol["optimizer_updates"]
            or not isinstance(protocol.get("protocol_id"), str)
            or not protocol["protocol_id"]
            or protocol.get("validation_samples_per_episode", 0) < 2
            or protocol.get("validation_checkpoint_labels") != [2000, protocol["optimizer_updates"] - 1]
            or protocol["decay_steps"] != protocol["optimizer_updates"]
            or protocol["manifest_sha256"] != hashlib.sha256(
                (root / "data/baseline_manifest.json").read_bytes()).hexdigest()):
        raise ValueError("Invalid or unbound user training protocol")
    return protocol


def owner_for(root, setting, manifest, job_id):
    path = root / "longleaf/parallel-baselines.json"
    if not path.exists():
        return str(job_id)
    assignment = json.loads(path.read_text())
    owners = assignment["owners"]
    if (set(owners) != set(SETTINGS) or len(set(owners.values())) != 3
            or any(not isinstance(v, str) or not v.isdigit() for v in owners.values())
            or assignment["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest()):
        raise ValueError("Invalid baseline ownership or changed accepted manifest")
    return owners[setting]


def owner_complete(root, owner, setting, manifest):
    path = root / f"logs/train-result-{owner}-{setting}-seed0.json"
    expected = root / "checkpoints/pi05_robodojo_longleaf_full" / f"{setting}-seed0"
    try:
        protocol = training_protocol(root)
        updates = protocol["optimizer_updates"]
        final_label = str(updates - 1)
        result = json.loads(path.read_text())
        if protocol['protocol_id'] != 'legacy-full-60000':
            checkpoint = expected / final_label
            metadata = json.loads((checkpoint / '_CHECKPOINT_METADATA').read_text())
            if not metadata.get('commit_timestamp_nsecs') or not all(
                    (checkpoint / name).exists() for name in ['train_state/_METADATA', 'params/_METADATA', 'assets']):
                return False
            from checkpoint_validation import receipt_path
            final_validation = json.loads(receipt_path(root, setting, int(final_label), owner).read_text())
            if final_validation.get('checkpoint_commit_timestamp_nsecs') != metadata['commit_timestamp_nsecs']:
                return False
        return (result["mode"] == "full" and result["setting"] == setting and result["seed"] == 0
                and result["optimizer_updates"] == updates and result["effective_batch_size"] == 256
                and (protocol["protocol_id"] == "legacy-full-60000"
                     or result.get("protocol_id") == protocol["protocol_id"])
                and result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
                and Path(result["checkpoint_dir"]).resolve() == expected.resolve()
                and final_label in result["saved_checkpoint_labels"] and (expected / final_label).is_dir()
                and all(validation_complete(root, owner, setting, label, protocol)
                        for label in protocol["validation_checkpoint_labels"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validation_complete(root, owner, setting, label, protocol):
    from checkpoint_validation import receipt_path
    try:
        result = json.loads(receipt_path(root, setting, label, owner).read_text())
        return (result['status'] == 'complete' and result['protocol_id'] == protocol['protocol_id']
                and result['manifest_sha256'] == protocol['manifest_sha256']
                and result['checkpoint_label'] == label and result['optimizer_state_step'] == label + 1
                and result['split'] == 'val' and result['weights'] == 'EMA'
                and result['samples'] > 0)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def wait_for_owner(root, setting, manifest, job_id, *, sleeper=time.sleep):
    owner = owner_for(root, setting, manifest, job_id)
    if owner == str(job_id):
        return True
    print("BASELINE_ASSIGNED_TO_OTHER_JOB", setting, owner,
          "waiting for its committed full result; no duplicate training", flush=True)
    while not owner_complete(root, owner, setting, manifest):
        if owner_for(root, setting, manifest, job_id) != owner:
            raise ValueError("Baseline owner changed while waiting")
        sleeper(60)
    print("BASELINE_COMPLETED_BY_OWNER", setting, owner, flush=True)
    return False
