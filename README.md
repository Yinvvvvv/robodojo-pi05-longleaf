# RoboDojo pi0.5 training — 5000 steps

Core training code and configuration for three seed-0 baselines.

| Setting | Data | Sampling |
| --- | --- | --- |
| ST-Organize | Organize only; 80/10/10 episodes | Organize 100% |
| MT-Uniform | Eight long-horizon tasks; 640/80/80 episodes | Each task 12.5% |
| MT-Organize50 | Same eight-task split | Organize 50%, other tasks 50%/7 each |

All three use full finetuning, 5000 optimizer updates, effective batch 256, global microbatch 8 (4/GPU), accumulation 32, two A100 40GB GPUs, FSDP=2, FP32 AdamW host states and EMA 0.99. LR uses 1000-step warmup, peak 2.5e-5, cosine endpoint 5000 and floor 2.5e-6. The pi0.5 backbone uses bfloat16, three cameras and action horizon 50.

## Files

- `longleaf/train_pi05.py`: training entry and accepted-data/configuration checks.
- `gradient_accumulation.py`, `optimizer_offloading.py`, `bounded_checkpointing.py`: batch accumulation, host optimizer state and bounded checkpoint serialization.
- `task_sampling.py`, `native_multi_dataset.py`: task weights and native dataset loading.
- `checkpoint_validation.py`: fixed EMA validation at committed checkpoints.
- `parallel_baselines.py`: protocol and per-job ownership guards.
- `longleaf/training-protocol.json`, `parallel-baselines.json`: shared protocol and recorded ownership configuration.
- `configs/<setting>/train-plan.json`: recorded native configuration for each baseline.

Label 4999 is optimizer state 5000; label 2000 is state 2001. Fixed EMA flow-matching validation uses seed 20261022 and 4 frames per selected episode: ST 40 frames/10 episodes, MT 320 frames/80 episodes. Validation loss is not physical success rate.

## External environment

External dependency: [XPolicyLab commit 53c5da5](https://github.com/XPolicyLab/XPolicyLab/tree/53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2) and its locked Python 3.11 environment. XPolicyLab/openpi source and the original runtime changes to upstream training, optimizer and checkpoint code must be supplied outside this repository.

Set `WM_ROOT`, `OPENPI_ROOT` and the appropriate job environment; provide the accepted manifest, train-split normalization assets and model weights externally. Adapt the recorded paths and owner IDs before reuse. The training entry verifies the accepted manifest and normalization digests.

This repository contains project training modules and recorded 5000-step configuration. Dependency source copies, continuation code/configuration, preprocessing and Slurm scripts, runtime logs, weights and datasets are excluded. It is not standalone. The original live environment, source files and running jobs are unchanged.
