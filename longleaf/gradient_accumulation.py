"""Full-batch gradients with microbatch activations; one optimizer/EMA update."""
import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import nnx
import optax


def mean_gradients(loss_fn, params, rng, batch, microbatch_size):
    """Average equal-size microbatch gradients before clipping or optimization."""
    batch_size = jax.tree.leaves(batch)[0].shape[0]
    if microbatch_size < 1 or batch_size % microbatch_size:
        raise ValueError("Effective batch must be divisible by global microbatch")
    count = batch_size // microbatch_size
    chunks = jax.tree.map(
        lambda value: value.reshape((count, microbatch_size, *value.shape[1:])), batch
    )
    keys = jax.random.split(rng, count)
    value_and_grad = jax.value_and_grad(loss_fn)

    def accumulate(carry, inputs):
        loss_sum, grad_sum = carry
        key, microbatch = inputs
        loss, grads = value_and_grad(params, key, microbatch)
        return (loss_sum + loss, jax.tree.map(jnp.add, grad_sum, grads)), None

    initial = (jnp.array(0.0, jnp.float32), jax.tree.map(jnp.zeros_like, params))
    (loss, grads), _ = jax.lax.scan(accumulate, initial, (keys, chunks))
    return loss / count, jax.tree.map(lambda value: value / count, grads)


def train_step(config, rng, state, batch, *, microbatch_size):
    trainable = state.params.filter(config.trainable_filter)

    def loss_fn(params, key, microbatch):
        model = nnx.merge(state.model_def, state.params)
        nnx.update(model, params)
        model.train()
        from openpi.training.sharding import activation_sharding_constraint
        observation, actions = activation_sharding_constraint(microbatch)
        return jnp.mean(model.compute_loss(key, observation, actions, train=True))

    key = jax.random.fold_in(rng, state.step)
    loss, grads = mean_gradients(loss_fn, trainable, key, batch, microbatch_size)
    # Fetch host-resident AdamW moments only after computing the mean gradient.
    from openpi.training import sharding
    opt_state = state.opt_state
    if sharding._MeshState.active_mesh is not None:
        opt_state = jax.device_put(opt_state, sharding.fsdp_sharding(
            opt_state, sharding._MeshState.active_mesh))
    updates, opt_state = state.tx.update(grads, opt_state, trainable)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, optax.apply_updates(trainable, updates))
    params = nnx.state(model)
    result = dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state)
    if state.ema_decay is not None:
        result = dataclasses.replace(
            result, ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, params,
            )
        )
    return result, {"loss": loss, "grad_norm": optax.global_norm(grads),
                    "param_norm": optax.global_norm(params)}


def install(training_module, microbatch_size):
    # Replace only the step; preserve upstream loading, FSDP, LR and checkpoints.
    training_module.train_step = functools.partial(train_step, microbatch_size=microbatch_size)


def self_check():
    import numpy as np
    from flax import struct
    from types import SimpleNamespace

    x = jnp.array([[1., 2.], [3., -1.], [2., 4.], [-2., 1.]])
    y = jnp.array([[1.], [-2.], [3.], [.5]])
    initial_weight = jnp.array([[.2], [-.1]])
    loss_fn = lambda weight, key, batch: jnp.mean((batch[0] @ weight - batch[1]) ** 2)
    expected_loss, expected_grad = jax.value_and_grad(loss_fn)(
        initial_weight, jax.random.key(0), (x, y))
    for size in [1, 2, 4]:
        loss, grads = jax.jit(
            functools.partial(mean_gradients, loss_fn, microbatch_size=size)
        )(initial_weight, jax.random.key(0), (x, y))
        np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
        np.testing.assert_allclose(grads, expected_grad, rtol=1e-6)
    try:
        mean_gradients(loss_fn, initial_weight, jax.random.key(0), (x, y), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid microbatch was accepted")

    class Tiny(nnx.Module):
        def __init__(self):
            self.weight = nnx.Param(initial_weight)

        def compute_loss(self, key, observation, actions, train=True):
            return (observation @ self.weight - actions) ** 2

    @struct.dataclass
    class TinyState:
        step: object
        params: object
        model_def: object = struct.field(pytree_node=False)
        tx: object = struct.field(pytree_node=False)
        opt_state: object
        ema_decay: object = struct.field(pytree_node=False)
        ema_params: object

    model = Tiny()
    params = nnx.state(model)
    tx = optax.chain(optax.clip_by_global_norm(.5),
                     optax.adamw(.01, b1=.9, b2=.95, weight_decay=.02))
    state = TinyState(jnp.array(0), params, nnx.graphdef(model), tx, tx.init(params), .9, params)
    cfg = SimpleNamespace(trainable_filter=nnx.Param)
    reference_grads = jax.tree.map(lambda value: expected_grad, params)
    updates, reference_opt = tx.update(reference_grads, state.opt_state, params)
    reference_params = optax.apply_updates(params, updates)
    result, _ = jax.jit(functools.partial(train_step, cfg, microbatch_size=1))(
        jax.random.key(0), state, (x, y))
    assert int(result.step) == 1
    np.testing.assert_allclose(result.params.weight.value,
                               reference_params.weight.value, rtol=1e-6)
    np.testing.assert_allclose(result.ema_params.weight.value,
                               .9 * initial_weight + .1 * reference_params.weight.value, rtol=1e-6)
    for actual, expected in zip(jax.tree.leaves(result.opt_state), jax.tree.leaves(reference_opt)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    if jax.device_count() == 2:
        from openpi.training import sharding
        mesh = sharding.make_mesh(2)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        state_sharding = sharding.fsdp_sharding(jax.eval_shape(lambda: state), mesh, min_size_mbytes=0)
        sharded_state = jax.device_put(state, state_sharding)
        sharded_batch = jax.device_put((x, y), data_sharding)
        step = jax.jit(functools.partial(train_step, cfg, microbatch_size=2),
                       in_shardings=(replicated, state_sharding, data_sharding),
                       out_shardings=(state_sharding, replicated))
        with sharding.set_mesh(mesh):
            distributed, _ = step(jax.random.key(0), sharded_state, sharded_batch)
        np.testing.assert_allclose(distributed.params.weight.value,
                                   reference_params.weight.value, rtol=1e-6)
        assert int(distributed.step) == 1
        print('TWO_DEVICE_FSDP_MICROBATCH_CHECK_PASSED')
    print("ACCUMULATION_SELF_CHECK_PASSED: mean gradient, clip/AdamW once, EMA once, step once")


if __name__ == "__main__":
    self_check()
