#!/usr/bin/env python3
"""Configure the pinned RoboDojo adapter without editing upstream source."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from openpi.training import config


def accepted_manifest(path, repo_id, root, config_name):
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('qc_status') != 'accepted':
        raise ValueError('Formal data QC and episode/layout splits have not been accepted')
    dataset = manifest['datasets'][repo_id]
    episodes = dataset['episodes']
    if len(episodes) < 3 or not {'train', 'val', 'test'} <= {
            entry['split'] for entry in episodes.values()}:
        raise ValueError('Formal data must contain fixed train/val/test episodes')
    if dataset.get('norm_source_split') != 'train':
        raise ValueError('Formal normalization must use training episodes only')
    norm = root / 'assets' / config_name / repo_id / 'norm_stats.json'
    if hashlib.sha256(norm.read_bytes()).hexdigest() != dataset['norm_stats_sha256']:
        raise ValueError('Formal normalization SHA256 does not match the accepted manifest')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['norm', 'train'], required=True)
    parser.add_argument('--mode', choices=['lora', 'full'], default='full')
    parser.add_argument('--repo-id', default='RoboDojo-organize_table-pilot-arx_x5-joint')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--microbatch-size', type=int, default=8,
                        help='Global microbatch, summed across GPUs')
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp-name')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--setting', choices=['pilot', 'ST-Organize', 'MT-Uniform', 'MT-Organize50'],
                        default='pilot')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--check-config', action='store_true',
                        help='Validate configuration without loading or updating model weights')
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 2 if args.phase == 'norm' else 256
    if min(args.steps, args.batch_size, args.gpus, args.save_interval, args.log_interval) < 1:
        parser.error('Steps, batch, GPU count and intervals must be positive')
    if args.num_workers < 0:
        parser.error('num-workers must be nonnegative')
    if args.phase == 'train' and (args.microbatch_size < 1
            or args.batch_size % args.microbatch_size or args.microbatch_size % args.gpus):
        parser.error('Global batch must divide into microbatches; each microbatch must divide across GPUs')
    if args.setting != 'pilot' and (args.phase != 'train' or args.mode != 'full'
            or args.batch_size != 256 or not args.manifest):
        parser.error('Formal baselines require full training, batch 256 and an accepted manifest')

    root = Path(os.environ['WM_ROOT'])
    from parallel_baselines import training_protocol
    protocol = training_protocol(root)
    if args.setting != 'pilot' and args.steps != protocol['optimizer_updates']:
        parser.error('Formal step target must match the user-approved training protocol')
    base = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    model = base.model
    if args.mode == 'lora':
        model = dataclasses.replace(model, paligemma_variant='gemma_2b_lora',
                                    action_expert_variant='gemma_300m_lora')
    cfg = dataclasses.replace(
        base, name=f'pi05_robodojo_longleaf_{args.mode}', model=model, seed=args.seed,
        freeze_filter=model.get_freeze_filter(),
        ema_decay=base.ema_decay if args.mode == 'full' else None,
        data=dataclasses.replace(base.data, repo_id=args.repo_id,
                                assets=config.AssetsConfig(asset_id=args.repo_id)),
        exp_name=args.exp_name or f'pilot-{os.environ.get("SLURM_JOB_ID", "manual")}',
        assets_base_dir=str(root / 'assets'), checkpoint_base_dir=str(root / 'checkpoints'),
        batch_size=args.batch_size, num_workers=args.num_workers, fsdp_devices=args.gpus,
        num_train_steps=args.steps, log_interval=args.log_interval,
        save_interval=min(args.save_interval, args.steps),
        keep_period=None, wandb_enabled=False, overwrite=False, resume=args.resume,
        lr_schedule=dataclasses.replace(base.lr_schedule,
            warmup_steps=protocol['warmup_steps'], decay_steps=protocol['decay_steps'],
            peak_lr=protocol['peak_lr'], decay_lr=protocol['decay_lr']),
    )
    manifest = None
    if args.setting != 'pilot':
        manifest = accepted_manifest(args.manifest, args.repo_id, root, cfg.name)
    plan = {
        'mode': args.mode, 'setting': args.setting, 'repo_id': args.repo_id,
        'protocol_id': protocol['protocol_id'],
        'validation_checkpoint_labels': protocol['validation_checkpoint_labels'],
        'seed': args.seed, 'gpus': args.gpus, 'fsdp_devices': cfg.fsdp_devices,
        'effective_batch_size': cfg.batch_size, 'optimizer_updates': cfg.num_train_steps,
        'global_microbatch_size': args.microbatch_size if args.phase == 'train' else None,
        'per_gpu_microbatch_size': args.microbatch_size // args.gpus if args.phase == 'train' else None,
        'accumulation_steps': cfg.batch_size // args.microbatch_size if args.phase == 'train' else None,
        'checkpoint_dir': str(cfg.checkpoint_dir),
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest() if manifest else None,
        'lr_schedule': repr(cfg.lr_schedule), 'optimizer': repr(cfg.optimizer),
        'ema_decay': cfg.ema_decay,
        'optimizer_state_memory': 'pinned_host',
        'xla_allocator': os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'default'),
        'xla_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'true'),
        'xla_memory_fraction': (None if os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR') == 'platform'
                                else float(os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '.90'))),
    }
    print('TRAIN_PLAN', json.dumps(plan), flush=True)
    if args.check_config:
        return
    if manifest:
        from parallel_baselines import wait_for_owner
        if not wait_for_owner(root, args.setting, args.manifest, os.environ["SLURM_JOB_ID"]):
            return
    config._CONFIGS_DICT[cfg.name] = cfg
    sys.path.insert(0, str(Path(os.environ['OPENPI_ROOT']) / 'scripts'))
    if args.phase == 'norm':
        import compute_norm_stats
        compute_norm_stats.main(cfg.name)
        return

    import jax
    devices = jax.devices()
    if len(devices) != args.gpus or any(d.platform != 'gpu' for d in devices):
        raise RuntimeError(f'Expected {args.gpus} CUDA devices; got {devices}')
    import train
    import optimizer_offloading
    optimizer_offloading.install(train)
    import gradient_accumulation
    gradient_accumulation.install(train, args.microbatch_size)
    if manifest:
        import task_sampling
        from openpi.training import data_loader
        task_sampling.install(data_loader, manifest, args.repo_id, args.setting, args.seed)
        import checkpoint_validation
        checkpoint_validation.install(train, cfg, root, args.setting, manifest, protocol)
    tag = f'{os.environ.get("SLURM_JOB_ID", "manual")}-{cfg.exp_name}'
    (root / f'logs/train-config-{tag}.txt').write_text(repr(cfg) + '\n')
    (root / f'logs/train-plan-{tag}.json').write_text(json.dumps(plan, indent=2) + '\n')
    start = time.monotonic()
    train.main(cfg)
    saved_steps = sorted(path.name for path in cfg.checkpoint_dir.iterdir()
                         if path.is_dir() and path.name.isdigit())
    if not saved_steps:
        raise RuntimeError(f'No saved checkpoint in {cfg.checkpoint_dir}')
    report = {**plan, 'elapsed_seconds_including_initialization': time.monotonic() - start,
              'saved_checkpoint_labels': saved_steps,
              'device_memory': [{'device': str(device), 'stats': device.memory_stats()}
                                for device in devices]}
    target = root / f'logs/train-result-{tag}.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print('TRAIN_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
