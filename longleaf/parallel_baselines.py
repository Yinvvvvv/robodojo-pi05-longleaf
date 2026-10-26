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


def self_check():
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from train_session import run_session
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("longleaf", "logs", "data"):
            (root / name).mkdir()
        manifest = root / "data/baseline_manifest.json"
        manifest.write_text(json.dumps({"qc_status": "accepted"}))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        assignment = {"manifest_sha256": digest,
                      "owners": dict(zip(SETTINGS, ["101", "102", "103"]))}
        assignment_path = root / "longleaf/parallel-baselines.json"
        assignment_path.write_text(json.dumps(assignment))
        assert wait_for_owner(root, "MT-Uniform", manifest, "102")
        checkpoint = root / "checkpoints/pi05_robodojo_longleaf_full/MT-Uniform-seed0"
        (checkpoint / "59999").mkdir(parents=True)
        result = {"mode": "full", "setting": "MT-Uniform", "seed": 0, "optimizer_updates": 60000,
                  "effective_batch_size": 256, "manifest_sha256": "wrong",
                  "checkpoint_dir": str(checkpoint), "saved_checkpoint_labels": ["59999"]}
        report = root / "logs/train-result-102-MT-Uniform-seed0.json"
        report.write_text(json.dumps(result))
        assert not owner_complete(root, "102", "MT-Uniform", manifest)
        slept = []
        def finish_owner(seconds):
            slept.append(seconds)
            result["manifest_sha256"] = digest
            report.write_text(json.dumps(result))
        assert wait_for_owner(root, "MT-Uniform", manifest, "101", sleeper=finish_owner) is False
        assert slept == [60]
        assignment["manifest_sha256"] = "wrong"
        assignment_path.write_text(json.dumps(assignment))
        try:
            wait_for_owner(root, "ST-Organize", manifest, "101")
        except ValueError:
            pass
        else:
            raise AssertionError("Changed accepted manifest was allowed")
        job = "102"
        (root / "longleaf/training-start.json").write_text(json.dumps(
            {"job_id": "101", "start_training": True, "scope": "three_baselines_seed0_full"}))
        (root / f"longleaf/training-start-{job}.json").write_text(json.dumps(
            {"job_id": job, "start_training": True, "scope": "three_baselines_seed0_full"}))
        control_path = root / f"longleaf/session-control-{job}.json"
        control = {"job_id": job, "retry_revision": 0, "release_approved": False}
        control_path.write_text(json.dumps(control))
        (root / f"logs/gpu-check-{job}.json").write_text(json.dumps(
            {"job_id": job, "sharded_reduction": "passed"}))
        calls = []
        def runner(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)
        def release_after_completion(seconds):
            status = json.loads((root / f"logs/session-{job}.json").read_text())
            assert status["phase"] == "completed_awaiting_user_release_approval"
            control.update(release_approved=True, approved_by="user", approved_at="test-only")
            control_path.write_text(json.dumps(control))
        run_session(root, job, Path("python"), runner=runner, sleeper=release_after_completion,
                    poll_seconds=0, setting="MT-Uniform")
        assert calls == [["bash", str(root / "longleaf/baselines.sbatch"), "MT-Uniform"]]
    print("PARALLEL_BASELINE_OWNERSHIP_AND_SCOPED_SESSION_CHECK_PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.parse_args()
    self_check()
