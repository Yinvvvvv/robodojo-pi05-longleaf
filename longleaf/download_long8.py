"""Download the frozen eight-task file list with resumable SHA256 receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import threading

from task_sampling import TASKS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retry_delay(exc, attempt):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    limited = status == 429 or 'rate limit' in str(exc).lower()
    if attempt == 11 or (not limited and attempt >= 2):
        return None
    delay = min(900, 60 * (2 ** min(attempt, 4))) if limited else 10 * (attempt + 1)
    if response is not None:
        try:
            delay = max(delay, int(response.headers.get('Retry-After', 0)))
        except ValueError:
            pass
    return delay


def self_check():
    from types import SimpleNamespace
    from tempfile import TemporaryDirectory
    exc = RuntimeError('HTTP rate limit')
    exc.response = SimpleNamespace(status_code=429, headers={'Retry-After': '90'})
    assert retry_delay(exc, 0) == 90
    assert retry_delay(exc, 5) == 900
    assert retry_delay(exc, 11) is None
    assert retry_delay(RuntimeError('network error'), 2) is None
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'checksum'
        path.write_bytes(b'abc')
        assert sha256(path) == hashlib.sha256(b'abc').hexdigest()
    print('DOWNLOAD_CHECK_PASSED: SHA256, rate-limit backoff, Retry-After, bounded failure')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    task = args.task or TASKS[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    root = Path(os.environ["WM_ROOT"])
    folder = root / "data/long8"
    inventory_path = folder / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    files = inventory["tasks"][task]["files"]
    if len([f for f in files if f["path"].endswith(".hdf5")]) != 100:
        raise ValueError("Unexpected official episode count; review inventory")
    if inventory["bytes"] > 600_000_000_000:
        raise ValueError("Inventory exceeds reviewed download budget")
    if any(not f["sha256"] or len(f["sha256"]) != 64 for f in files):
        raise ValueError("Missing source SHA256")
    verified = folder / "download_receipts" / task
    verified.mkdir(parents=True, exist_ok=True)

    throttle = threading.Lock()
    last_request = [0.0]
    stop = threading.Event()

    def download(entry):
        from huggingface_hub import hf_hub_download
        name = entry["path"]
        receipt = verified / (hashlib.sha256(name.encode()).hexdigest() + ".json")
        target = root / "data/raw" / name
        if receipt.exists() and target.exists():
            old = json.loads(receipt.read_text())
            stat = target.stat()
            if (old["sha256"] == entry["sha256"] and old["bytes"] == stat.st_size
                    and old["mtime_ns"] == stat.st_mtime_ns):
                return name, "verified_unchanged"
        force_download = False
        for attempt in range(12):
            if stop.is_set():
                raise RuntimeError('Task stopped after another file failed')
            with throttle:
                time.sleep(max(0.0, 4 - (time.monotonic() - last_request[0])))
                last_request[0] = time.monotonic()
            try:
                path = Path(hf_hub_download(inventory["source_repo"], name,
                    repo_type="dataset", revision=inventory["source_revision"],
                    local_dir=root / "data/raw", force_download=force_download))
                if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                    raise ValueError(f"Downloaded file size or SHA256 mismatch: {name}")
                record = {**entry, "mtime_ns": path.stat().st_mtime_ns,
                          "source_revision": inventory["source_revision"]}
                temp = receipt.with_suffix(".tmp")
                temp.write_text(json.dumps(record) + "\n")
                temp.replace(receipt)
                return name, "downloaded_sha256_verified"
            except Exception as exc:
                if isinstance(exc, ValueError):
                    force_download = True
                delay = retry_delay(exc, attempt)
                if delay is None:
                    stop.set()
                    raise
                response = getattr(exc, 'response', None)
                status = getattr(response, 'status_code', None)
                print('DOWNLOAD_BACKOFF', task, name, 'status', status, 'seconds', delay, flush=True)
                if stop.wait(delay):
                    raise RuntimeError('Task stopped after another file failed') from exc

    # Download HDF5 before preview videos; no other tasks or robot embodiments.
    files = sorted(files, key=lambda item: (not item["path"].endswith(".hdf5"), item["path"]))
    completed = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, entry) for entry in files]
        for future in as_completed(futures):
            path, status = future.result()
            completed += 1
            print("FILE_VERIFIED", task, completed, len(files), status, path, flush=True)
    report = {"task": task, "files": completed, "episodes": 100,
              "bytes": sum(entry["bytes"] for entry in files),
              "source_revision": inventory["source_revision"],
              "inventory_sha256": sha256(inventory_path), "sha256_verified": True}
    target = folder / f"download-{task}.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(target)
    print("TASK_DOWNLOAD_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
