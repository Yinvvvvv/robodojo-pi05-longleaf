"""Retain one allocation through training failures until user-approved release."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def readiness(manifest, control, job_id):
    ready = read_json(manifest).get("qc_status") == "accepted"
    approved = read_json(control)
    start = (ready and approved.get("start_training") is True
             and approved.get("job_id") == str(job_id)
             and approved.get("scope") == "three_baselines_seed0_full")
    return ready, start


def release_allowed(control, job_id):
    return (control.get("job_id") == str(job_id)
            and control.get("release_approved") is True
            and control.get("approved_by") == "user"
            and bool(control.get("approved_at")))


def retry_revision(control, job_id):
    value = control.get("retry_revision", 0)
    if (control.get("job_id") != str(job_id) or type(value) is not int or value < 0):
        return None
    return value


def write_status(root, job_id, phase, **details):
    record = {"job_id": str(job_id), "phase": phase,
              "updated_utc": datetime.now(timezone.utc).isoformat(),
              "release_requires_user_approval": True, **details}
    target = root / f"logs/session-{job_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(target)
    print("SESSION_PHASE", json.dumps(record), flush=True)


def run_session(root, job_id, python, *, runner=subprocess.run, sleeper=time.sleep, poll_seconds=60, setting=None):
    manifest = root / "data/baseline_manifest.json"
    authorization = root / f"longleaf/training-start-{job_id}.json"
    if not authorization.exists():
        authorization = root / "longleaf/training-start.json"
    control_path = root / f"longleaf/session-control-{job_id}.json"
    attempted_revision = None
    prior_gpu_check = read_json(root / f"logs/gpu-check-{job_id}.json")
    hardware_passed = (prior_gpu_check.get("job_id") == str(job_id)
                       and prior_gpu_check.get("sharded_reduction") == "passed")
    completed = False
    last_phase = None
    while True:
        control = read_json(control_path)
        if release_allowed(control, job_id):
            write_status(root, job_id, "user_approved_release")
            return
        ready, approved = readiness(manifest, authorization, job_id)
        revision = retry_revision(control, job_id)
        if completed:
            phase = "completed_awaiting_user_release_approval"
        elif not ready:
            phase = "data_preparation"
        elif not approved or revision is None:
            phase = "awaiting_job_scoped_authorization"
        elif attempted_revision is not None and revision <= attempted_revision:
            phase = "training_failed_retaining_allocation"
        else:
            attempted_revision = revision
            try:
                if not hardware_passed:
                    write_status(root, job_id, "hardware_validation")
                    result = runner([str(python), str(root / "longleaf/gpu_check.py")], check=False)
                    if result.returncode:
                        raise RuntimeError(f"GPU validation returned {result.returncode}")
                    hardware_passed = True
                write_status(root, job_id, "training", retry_revision=revision)
                print("SESSION_STARTING_CONFIRMED_BASELINES", flush=True)
                result = runner(["bash", str(root / "longleaf/baselines.sbatch"),
                                 *([setting] if setting else [])], check=False)
                if result.returncode:
                    raise RuntimeError(f"Baselines returned {result.returncode}")
                completed = True
                phase = "completed_awaiting_user_release_approval"
            except Exception as error:
                # A repair increments retry_revision; never hot-loop a failing model.
                phase = "training_failed_retaining_allocation"
                write_status(root, job_id, phase, retry_revision=revision,
                             error=f"{type(error).__name__}: {error}")
                last_phase = phase
        if phase != last_phase:
            write_status(root, job_id, phase, retry_revision=attempted_revision,
                         data_ready=ready, training_authorized=approved)
            last_phase = phase
        # An honest wait for repair or human release approval; no GPU busy-work.
        sleeper(poll_seconds)


def main(setting=None):
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Run inside the submitted Slurm allocation")
    root = Path(os.environ["WM_ROOT"])
    python = Path(os.environ["OPENPI_ROOT"]) / ".venv/bin/python"
    run_session(root, job_id, python, setting=setting)


def self_check():
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    for raise_error in (False, True):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("longleaf", "logs", "data"):
                (root / name).mkdir()
            job = "example"
            manifest = root / "data/baseline_manifest.json"
            auth = root / "longleaf/training-start.json"
            control_path = root / f"longleaf/session-control-{job}.json"
            control = {"job_id": job, "retry_revision": 0, "release_approved": False}
            manifest.write_text(json.dumps({"qc_status": "accepted"}))
            auth.write_text(json.dumps({"start_training": True, "job_id": job,
                                       "scope": "three_baselines_seed0_full"}))
            control_path.write_text(json.dumps(control))
            if raise_error:
                (root / f"logs/gpu-check-{job}.json").write_text(json.dumps(
                    {"job_id": job, "sharded_reduction": "passed"}))
            calls, sleeps = [], []

            def runner(cmd, **kwargs):
                calls.append(cmd)
                if cmd[0] == "bash" and sum(c[0] == "bash" for c in calls) == 1:
                    if raise_error:
                        raise OSError("simulated launch failure")
                    return SimpleNamespace(returncode=1)
                return SimpleNamespace(returncode=0)

            def sleeper(seconds):
                phase = read_json(root / f"logs/session-{job}.json")["phase"]
                sleeps.append(phase)
                assert len(sleeps) < 12, "Release/retry loop did not converge"
                if len(sleeps) <= 2:
                    assert phase == "training_failed_retaining_allocation"
                    assert sum(c[0] == "bash" for c in calls) == 1
                elif len(sleeps) == 3:
                    # Wrong-job retry/release cannot affect this allocation.
                    control_path.write_text(json.dumps({"job_id": "other", "retry_revision": 99,
                        "release_approved": True, "approved_by": "user", "approved_at": "now"}))
                elif len(sleeps) == 4:
                    control["retry_revision"] = 1
                    control_path.write_text(json.dumps(control))
                else:
                    assert phase == "completed_awaiting_user_release_approval"
                    assert sum(c[0] == "bash" for c in calls) == 2
                    assert sum(c[0] != "bash" for c in calls) == (0 if raise_error else 1)
                    control["retry_revision"] = 2
                    if len(sleeps) >= 6:
                        control.update(release_approved=True, approved_by="user", approved_at="now")
                    control_path.write_text(json.dumps(control))
            run_session(root, job, Path("python"), runner=runner, sleeper=sleeper, poll_seconds=0)
            assert read_json(root / f"logs/session-{job}.json")["phase"] == "user_approved_release"
            assert not release_allowed({**control, "approved_by": "agent"}, job)
            assert retry_revision({"job_id": job, "retry_revision": True}, job) is None
            assert readiness(manifest, auth, "wrong-job") == (True, False)
    print("SESSION_RETENTION_CHECK_PASSED: failures retained; no blind retry; scoped repair; success waits for user release")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--setting", choices=["ST-Organize", "MT-Uniform", "MT-Organize50"])
    parser.add_argument("--release-approved", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.release_approved:
        root = Path(os.environ["WM_ROOT"])
        job_id = os.environ["SLURM_JOB_ID"]
        raise SystemExit(0 if release_allowed(read_json(root / f"longleaf/session-control-{job_id}.json"), job_id) else 1)
    else:
        main(args.setting)
