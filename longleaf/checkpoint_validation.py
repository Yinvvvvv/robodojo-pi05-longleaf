"""Fixed held-out flow-matching loss at committed checkpoint milestones."""
from collections import defaultdict
import functools
import hashlib
import json
import os
import time


def select_frames(episode_ids, episodes, samples_per_episode):
    groups = defaultdict(list)
    for index, episode in enumerate(episode_ids):
        key = str(int(episode))
        if episodes[key]['split'] == 'val':
            groups[key].append(index)
    expected = {key for key, value in episodes.items() if value['split'] == 'val'}
    if set(groups) != expected or not expected:
        raise ValueError('Missing held-out validation episodes')
    selection = []
    for key in sorted(groups, key=int):
        frames = groups[key]
        if len(frames) < samples_per_episode:
            raise ValueError('Not enough validation frames')
        for i in range(samples_per_episode):
            offset = i * (len(frames) - 1) // (samples_per_episode - 1)
            selection.append({'index': frames[offset], 'episode': key, 'task': episodes[key]['task']})
    return selection


def receipt_path(root, setting, label, job=None):
    return root / f'logs/validation-{job or os.environ["SLURM_JOB_ID"]}-{setting}-seed0-step{label}.json'


def install(training_module, cfg, root, setting, manifest, protocol):
    if not protocol['validation_checkpoint_labels']:
        return
    original_save = training_module._checkpoints.save_state
    original_restore = training_module._checkpoints.restore_state

    def evaluate(manager, state, loader, label):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax import nnx
        from lerobot.datasets import lerobot_dataset
        from openpi.models.model import Observation
        from openpi.training import data_loader, sharding

        jax.block_until_ready(state)
        if int(state.step) != label + 1 or state.ema_params is None:
            raise ValueError('Validation must use the committed EMA state at this label')
        checkpoint = cfg.checkpoint_dir / str(label)
        metadata = json.loads((checkpoint / '_CHECKPOINT_METADATA').read_text())
        commit = metadata['commit_timestamp_nsecs']
        if not commit:
            raise ValueError('Validation requires a committed checkpoint')
        dataset = loader._data_loader.torch_loader.dataset
        raw = dataset
        while not hasattr(raw, 'meta') and not isinstance(raw, lerobot_dataset.MultiLeRobotDataset):
            raw = raw._dataset
        entry = manifest['datasets'][cfg.data.repo_id]
        sources = entry.get('source_datasets')
        if sources:
            ids = [int(ep) + source['episode_offset']
                   for source, part in zip(sources, raw._datasets, strict=True)
                   for ep in part.hf_dataset.with_format(None)['episode_index']]
        else:
            raw[0]
            ids = raw.hf_dataset.with_format(None)['episode_index']
        selected = select_frames(ids, entry['episodes'], protocol['validation_samples_per_episode'])
        selection_sha = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        binding = {'protocol_id': protocol['protocol_id'], 'manifest_sha256': protocol['manifest_sha256'],
                   'checkpoint_label': label, 'optimizer_state_step': int(state.step),
                   'checkpoint_commit_timestamp_nsecs': commit, 'selection_sha256': selection_sha,
                   'seed': protocol['validation_seed']}
        target = receipt_path(root, setting, label)
        if target.exists():
            previous = json.loads(target.read_text())
            if previous.get('status') == 'complete' and all(previous.get(k) == v for k, v in binding.items()):
                return
        print('HELD_OUT_VALIDATION_START', json.dumps(binding), flush=True)
        started = time.monotonic()
        params = state.ema_params
        mesh = jax.tree.leaves(params)[0].sharding.mesh
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        model_def = state.model_def

        def loss(parameters, observation, actions, key):
            model = nnx.merge(model_def, parameters)
            model.eval()
            values = model.compute_loss(key, observation, actions, train=False)
            return jnp.mean(values, axis=tuple(range(1, values.ndim)))

        forward = jax.jit(loss, in_shardings=(jax.tree.map(lambda x: x.sharding, params),
                          data_sharding, data_sharding, replicated), out_shardings=replicated)
        by_task = defaultdict(list)
        # Reuse this process's dataset and EMA. No second model, optimizer, or checkpoint copy.
        batch_size = 8
        random_state = np.random.get_state()
        np.random.seed(protocol['validation_seed'])
        try:
            with sharding.set_mesh(mesh):
                for batch_number, start in enumerate(range(0, len(selected), batch_size)):
                    rows = selected[start:start + batch_size]
                    padded = rows + [rows[-1]] * (batch_size - len(rows))
                    batch = data_loader._collate_fn([dataset[row['index']] for row in padded])
                    batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
                    key = jax.device_put(jax.random.fold_in(jax.random.key(protocol['validation_seed']), batch_number), replicated)
                    values = np.asarray(forward(params, Observation.from_dict(batch), batch['actions'], key))[:len(rows)]
                    if values.shape != (len(rows),) or not np.isfinite(values).all():
                        raise ValueError('Invalid held-out validation loss')
                    for row, value in zip(rows, values, strict=True):
                        by_task[row['task']].append(float(value))
        finally:
            np.random.set_state(random_state)
            forward.clear_cache()
        result = {**binding, 'status': 'complete', 'setting': setting, 'split': 'val',
                  'weights': 'EMA', 'samples': len(selected), 'episodes': len(selected) // protocol['validation_samples_per_episode'],
                  'loss': float(np.mean([v for values in by_task.values() for v in values])),
                  'per_task': {task: {'samples': len(values), 'loss': float(np.mean(values))}
                               for task, values in sorted(by_task.items())},
                  'elapsed_seconds': time.monotonic() - started,
                  'metric': 'held-out flow-matching loss; not rollout success rate',
                  'selection': selected}
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(target)
        print('HELD_OUT_VALIDATION_COMPLETE', json.dumps({k: v for k, v in result.items() if k != 'selection'}), flush=True)

    @functools.wraps(original_save)
    def save(manager, state, loader, step):
        original_save(manager, state, loader, step)
        if step in protocol['validation_checkpoint_labels']:
            manager.wait_until_finished()
            evaluate(manager, state, loader, step)

    @functools.wraps(original_restore)
    def restore(manager, state, loader, step=None):
        restored = original_restore(manager, state, loader, step)
        label = int(restored.step) - 1
        if label in protocol['validation_checkpoint_labels']:
            evaluate(manager, restored, loader, label)
        return restored

    training_module._checkpoints.save_state = save
    training_module._checkpoints.restore_state = restore
