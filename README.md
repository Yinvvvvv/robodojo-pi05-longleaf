# RoboDojo pi0.5 Longleaf training (5000-step snapshot)

Private source/configuration snapshot for ST-Organize, MT-Uniform and MT-Organize50. This export contains the baseline 5000-update training path. The requested 7000/10000-step continuation scripts and their configuration are intentionally excluded.

## Protocol

| Setting | Training data | Task sampling mass |
| --- | --- | --- |
| ST-Organize | Organize only, 80/10/10 episodes | Organize 100% |
| MT-Uniform | Eight selected long-horizon tasks, 640/80/80 episodes | Each task 12.5% |
| MT-Organize50 | Same eight-task dataset and split | Organize 50%; each other task 50%/7 |

All three use seed 0, full finetuning, effective batch 256, global microbatch 8 (4 per GPU), accumulation 32, two A100 40GB GPUs, FSDP=2, FP32 AdamW states on host, EMA 0.99, 1000 warmup steps, peak LR 2.5e-5 and cosine endpoint/floor 5000/2.5e-6. The inherited pi0.5 backbone uses bfloat16. Three cameras and action horizon 50 are retained.

Checkpoint label 4999 represents optimizer state 5000; label 2000 represents state 2001. The final printed training metric may be Step4900; no artificial Step5000 loss is created. Fixed validation uses EMA flow-matching loss (not rollout success), 4 samples per selected episode, seed 20261022. ST uses 40 frames/10 episodes; MT uses 320 frames/80 episodes. See the recorded native plans and TrainConfig text under `configs/`.

## Layout

- `longleaf/train_pi05.py`: training/normalization adapter and accepted-manifest gate.
- `longleaf/gradient_accumulation.py`, `optimizer_offloading.py`, `bounded_checkpointing.py`: microbatch accumulation, host optimizer memory, bounded native checkpoint serialization.
- `longleaf/task_sampling.py`, `native_multi_dataset.py`: task weights, train-only frames and native multi-source data loading.
- `longleaf/checkpoint_validation.py`: serial committed-checkpoint EMA validation using the current model.
- `longleaf/prepare_*.py`, `download_long8.py`, `finish_long8.py`: data preparation/QC and train-only normalization.
- `longleaf/env.sh`, `setup.sbatch`, `baselines.sbatch`, `train_session*`: recorded Longleaf environment and baseline launch/retention logic.
- `upstream-overlay/XPolicyLab/`: the training files captured from the runtime plus the pinned base configuration and upstream license.
- `SOURCE_MANIFEST.json`: file digests and provenance. Captured runtime bodies are preserved; exported upstream runtime files have only an added provenance comment.

This is a source snapshot, not a standalone vendoring of all dependencies. Checkpoints, datasets, image/video assets, norm arrays, runtime logs, credentials and live job control files are excluded. The original Longleaf paths and historical owner IDs remain in the recorded configuration. Cloning this repository does not submit, restart or release any job.

## Reconstruct the source environment

The base source is [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/tree/53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2), pinned to `53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2`.

1. Clone that revision into the `XPOLICYLAB_ROOT` chosen for your workspace.
2. Copy `upstream-overlay/XPolicyLab/` over that checkout, retaining the rest of the pinned upstream code.
3. Use Python 3.11 and the upstream locked environment (`uv sync --locked --no-dev --group lerobot --python 3.11`); the recorded setup also installs the XPolicyLab package and its required pytest import.
4. Provide the accepted dataset manifest and train-split norm assets externally. The adapter verifies their original digests; the example manifest is not an accepted production manifest.
5. Adapt the root paths and job-scoped ownership/control files for your own allocation. The checked-in session-control file is an example, not permission to run or release live resources. The original launcher assumes an existing resume checkpoint; first-time training must explicitly choose the corresponding non-resume initialization path.

The bounded checkpoint writer snapshot is the current base training implementation. Its presence in this export does not imply every historical training attempt had already loaded that writer. No active source file, checkpoint, job or evaluation binding was modified to create this repository.

## Minimal export verification

Python sources are parsed without imports or model loads; shell files pass `bash -n`; JSON and SHA256 integrity are checked. These checks do not claim a fresh GPU training run.

Third-party source and changes are described in `THIRD_PARTY_NOTICES.md`.
