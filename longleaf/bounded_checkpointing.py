"""Stream JAX checkpoint leaves to disk without staging a whole model in RAM.

Pinned to the installed Orbax ArrayHandler API. This changes only serialization;
checkpoint items, array metadata, dtype, sharding and restoration stay native.
"""
import asyncio
import copy
import dataclasses
import fcntl
import functools
import hashlib
import os
from pathlib import Path
import shutil
import json
import threading

import jax
from orbax.checkpoint import future
from orbax.checkpoint._src.serialization import replica_slices, serialization, type_handlers, types

# ponytail: single-host, serial leaf writes; revisit only if save latency matters.
_SAVE_LOCK = threading.Lock()
_INSTALLED = False


class BoundedArrayHandler(type_handlers.ArrayHandler):
    async def serialize(self, values, infos, args=None):
        if jax.process_count() != 1:
            raise ValueError('Bounded checkpoint serialization supports one host only')
        args = args or [types.SaveArgs()] * len(values)
        type_handlers.check_input_arguments(values, infos, args)
        type_handlers.check_array_values(values, infos)
        # Honor Orbax's directory-creation contract before any TensorStore write.
        return [future.CommitFutureAwaitingContractedSignals(
            self._serialize_streaming(values, infos, args), name='bounded_array_writer')]

    async def _serialize_streaming(self, values, infos, args):
        await asyncio.to_thread(_SAVE_LOCK.acquire)
        try:
            # Per-call handler state: params and train_state may enter concurrently.
            handler = copy.copy(self)
            store = handler._array_metadata_store
            handler._array_metadata_store = None
            handler._ext_metadata = {}
            metadata = []
            limiter = serialization.get_byte_limiter(4 * 1024**3)
            largest = 0
            for value, info, arg in zip(values, infos, args, strict=True):
                if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
                    raise ValueError('Training checkpoints must not contain typed PRNG leaves')
                info = dataclasses.replace(info, byte_limiter=limiter)
                # Unlike the original handler, never transfer all leaves together.
                # A temporary pinned-host array avoids caching a NumPy copy on GPU state.
                host = replica_slices.transfer_arrays_to_host(
                    [value], handler._replica_id, handler._use_replica_parallel,
                    enable_pinned_host_transfer=True)
                spec = handler._get_array_write_spec(
                    info, host[0], use_ocdbt=info.is_ocdbt_checkpoint,
                    process_index=type_handlers.get_process_index_for_subdir(
                        info.is_ocdbt_checkpoint), arg=arg)
                metadata.append(spec.metadata)
                largest = max(largest, value.size * value.dtype.itemsize)
                # Await actual writes before releasing the staging buffer or copying next leaf.
                await handler._background_serialize(host, [info], [arg])
                del host, spec
            if store is not None:
                await store.write(checkpoint_dir=infos[0].parent_dir,
                                  array_metadatas=metadata, process_index=jax.process_index())
            print('BOUNDED_CHECKPOINT_ARRAYS_WRITTEN', json.dumps({
                'item': str(infos[0].parent_dir), 'arrays': len(values),
                'max_staged_array_bytes': largest, 'write_limit_bytes': 4 * 1024**3,
                'writes_completed_before_next_array': True}), flush=True)
            return []
        finally:
            _SAVE_LOCK.release()


def pin_inference_checkpoint(root, checkpoint):
    """Keep only immutable EMA inference files, sharing blocks until native deletion."""
    checkpoint = Path(checkpoint)
    if checkpoint.name != '2000' or checkpoint.parent.parent.name != 'pi05_robodojo_longleaf_full':
        return None
    metadata = json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
    assert metadata['commit_timestamp_nsecs']
    target = Path(root)/'inference-milestones'/checkpoint.parent.name/'step2000'
    binding = {'source_checkpoint': str(checkpoint), 'checkpoint_label': 2000,
               'optimizer_state_step': 2001, 'inference_weights': 'EMA',
               'checkpoint_commit_timestamp_nsecs': metadata['commit_timestamp_nsecs'],
               'source_metadata_sha256': hashlib.sha256((checkpoint/'_CHECKPOINT_METADATA').read_bytes()).hexdigest(),
               'inference_only_not_training_resume': True}
    if target.exists():
        assert all(json.loads((target/'source.json').read_text()).get(k) == v for k,v in binding.items())
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.tmp-{os.getpid()}')
    temporary.mkdir()
    try:
        total = 0
        for source in (checkpoint/'params').rglob('*'):
            assert not source.is_symlink()
            destination = temporary/source.relative_to(checkpoint)
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
                total += source.stat().st_size
        shutil.copytree(checkpoint/'assets', temporary/'assets')
        (temporary/'source.json').write_text(json.dumps({**binding, 'params_apparent_bytes': total}, indent=2)+'\n')
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print('INFERENCE_MILESTONE_PINNED', str(target), flush=True)
    return target


def install():
    global _INSTALLED
    if not _INSTALLED:
        type_handlers.register_type_handler(jax.Array, BoundedArrayHandler(), override=True)
        from openpi.training import checkpoints
        original_save = checkpoints.save_state

        @functools.wraps(original_save)
        def serial_save(manager, state, loader, step):
            root = Path(os.environ['WM_ROOT'])
            # One full in-flight checkpoint across the three jobs bounds disk reserve.
            with (root/'logs/checkpoint-write.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                original_save(manager, state, loader, step)
                manager.wait_until_finished()
                if (root/'longleaf/checkpoint-serialization-all-writers-verified.json').is_file():
                    pin_inference_checkpoint(root, Path(str(manager.directory))/str(step))

        checkpoints.save_state = serial_save
        _INSTALLED = True
        print('BOUNDED_CHECKPOINT_SERIALIZATION_ENABLED', flush=True)


def self_check():
    import os
    from pathlib import Path
    import numpy as np
    from flax import nnx
    from openpi.training import checkpoints, config, sharding
    import check_optimizer_offloading as tiny
    import optimizer_offloading

    mesh = sharding.make_mesh(2)
    base = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    cfg = dataclasses.replace(base, model=tiny.TinyConfig(), freeze_filter=nnx.Nothing,
                              fsdp_devices=2, weight_loader=tiny.Loader())
    optimizer_offloading.install(tiny.train)
    state, layout = tiny.train.init_train_state(cfg, jax.random.key(0), mesh, resume=False)
    state = dataclasses.replace(state, step=jax.device_put(np.array(2001, np.int32), layout.step),
        opt_state=jax.tree.map(lambda x, spec: jax.device_put(
            np.full(x.shape, 1 if np.issubdtype(x.dtype, np.integer) else 0.125, x.dtype), spec),
            state.opt_state, layout.opt_state))
    root = Path(os.environ['WM_ROOT'])/'data/long8'/f'bounded-checkpoint-check-{os.getpid()}'
    manager, _ = checkpoints.initialize_checkpoint_dir(root, keep_period=None, overwrite=False, resume=False)
    for label in [2000, 2001]:
        checkpoints.save_state(manager, state, tiny.Data(), label)
        manager.wait_until_finished()
        target = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), state)
        restored = checkpoints.restore_state(manager, target, tiny.Data(), label)
        for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            assert actual.dtype == expected.dtype
        assert all(x.sharding.memory_kind == 'pinned_host' for x in jax.tree.leaves(restored.opt_state))
        assert json.loads((root/str(label)/'_CHECKPOINT_METADATA').read_text())['commit_timestamp_nsecs']
    manager.close()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp)/'checkpoints/pi05_robodojo_longleaf_full/ST-Organize-seed0/2000'
        (probe/'params').mkdir(parents=True); (probe/'assets').mkdir()
        (probe/'params/array').write_bytes(b'immutable-weight-test')
        (probe/'assets/norm_stats.json').write_text('{}')
        (probe/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs': 1}))
        pinned = pin_inference_checkpoint(Path(tmp), probe)
        assert (pinned/'params/array').stat().st_ino == (probe/'params/array').stat().st_ino
        shutil.rmtree(probe)
        assert (pinned/'params/array').read_bytes() == b'immutable-weight-test'
        assert not (pinned/'train_state').exists()
    result = {'status': 'passed', 'inference_hardlinks_survive_source_deletion': True, 'two_consecutive_native_commits': True,
              'exact_params_adamw_ema_step_restore': True, 'pinned_host_restored': True,
              'synthetic_model': True, 'full_model_save_peak_verified': False, 'path': str(root)}
    print('BOUNDED_CHECKPOINT_SELF_CHECK_PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    self_check()
