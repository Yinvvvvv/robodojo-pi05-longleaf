"""Keep FP32 AdamW state in host memory between full-training updates."""
import dataclasses
import functools
import gc
import json
import os
import subprocess

import jax
import openpi.shared.array_typing as at


def install(training_module):
    import bounded_checkpointing
    bounded_checkpointing.install()
    original = training_module.init_train_state

    @functools.wraps(original)
    def init_train_state(*args, **kwargs):
        state, layout = original(*args, **kwargs)
        with at.disable_typechecking():
            layout = dataclasses.replace(layout, opt_state=jax.tree.map(
                lambda spec: spec.with_memory_kind('pinned_host'), layout.opt_state))
            if kwargs.get('resume', False):
                state = jax.tree.map(lambda value, spec: jax.ShapeDtypeStruct(
                    value.shape, value.dtype, sharding=spec, weak_type=value.weak_type),
                    state, layout)
            else:
                state = jax.device_put(state, layout)
                jax.block_until_ready(state)
                gc.collect()
                memory = {'allocator': os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'default'),
                          'device_stats': [device.memory_stats() for device in jax.devices()]}
                try:
                    lines = subprocess.check_output(['nvidia-smi',
                        '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                        '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
                    memory['this_process_gpu_memory'] = [line for line in lines
                        if line.split(',')[0].strip() == str(os.getpid())]
                except (OSError, subprocess.SubprocessError):
                    memory['this_process_gpu_memory'] = None
                print('OPTIMIZER_OFFLOAD_MEMORY', json.dumps(memory), flush=True)
        return state, layout

    original_restore = training_module._checkpoints.restore_state

    @functools.wraps(original_restore)
    def restore_state(manager, state, data_loader, step=None):
        restored = original_restore(manager, state, data_loader, step)
        # Orbax restores numeric values but may discard the host memory kind.
        placements = jax.tree.map(lambda value: value.sharding, state)
        return jax.device_put(restored, placements)

    original_save = training_module._checkpoints.save_state

    @functools.wraps(original_save)
    def save_state(manager, state, data_loader, step):
        original_save(manager, state, data_loader, step)
        # Full checkpoint copies reached the host RAM limit; do not overlap a new update.
        manager.wait_until_finished()

    training_module._checkpoints.save_state = save_state
    training_module.init_train_state = init_train_state
    training_module._checkpoints.restore_state = restore_state
