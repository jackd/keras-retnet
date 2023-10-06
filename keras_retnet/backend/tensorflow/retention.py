import functools
import typing as tp

import jax
import tensorflow as tf
from jax.experimental import jax2tf

from ..jax import retention as jax_retention


@functools.cache
def _get_retention(**kwargs):
    def retention_fn(query, keys, values, gamma):
        return jax_retention.retention(query, keys, values, gamma, **kwargs)

    return jax2tf.convert(jax.jit(retention_fn))


def retention(
    query: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, gamma: tf.Tensor, **kwargs
) -> tf.Tensor:
    return _get_retention(**kwargs)(query, keys, values, gamma)


retention.__doc__ = jax_retention.retention

# _create_retention_update_cache = jax2tf.convert(
#     jax.jit(jax_retention.create_retention_update_cache)
# )


# def create_retention_update_cache(
#     keys: tf.Tensor, values: tf.Tensor, gamma: tf.Tensor, current_index=None
# ) -> tf.Tensor:
#     return _create_retention_update_cache(keys, values, gamma, current_index)


# create_retention_update_cache.__doc__ = jax_retention.create_retention_update_cache


def _create_retention_update_cache(
    args,
    gamma: tf.Tensor,
    current_index: tp.Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Args:
        keys: [T, d_k]
        values: [T, d_v]
        gamma: []
        current_index: [] int, current_index < T. If None, implied to be T-1

    Returns:
        updated_cache: [B?, d_k, d_v]
    """
    keys, values = args
    assert keys.ndim == 2, keys.shape
    assert values.ndim == 2, values.shape
    assert keys.shape[0] == values.shape[0], (keys.shape, values.shape)
    assert gamma.ndim == 0, gamma.shape

    T = keys.shape[0]

    if current_index is None:
        factors = gamma ** tf.range(T - 1, -1, -1, dtype=gamma.dtype)
    else:
        t_range = tf.range(T)
        padding = t_range > current_index
        factors = gamma ** (tf.range(0, -T, -1, dtype=gamma.dtype) + current_index)
        # factors = gamma ** jnp.arange(current_index, current_index - T, -1)
        factors = tf.where(padding, tf.zeros_like(factors), factors)
    # out = tf.einsum("tk,t,tv->kv", keys, factors, values)
    out = tf.matmul(keys, values * factors[:, None], adjoint_a=True)
    return out


def create_retention_update_cache(
    keys: tf.Tensor,
    values: tf.Tensor,
    gamma: tf.Tensor,
    current_index: tp.Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Args:
        keys: [B?, T, d_k]
        values: [B?, T, d_v]
        gamma: []
        current_index: [] int, current_index < T. If None, implied to be T-1

    Returns:
        updated_cache: [B?, d_k, d_v]
    """
    if keys.ndim == 3:
        return tf.vectorized_map(
            functools.partial(
                _create_retention_update_cache,
                gamma=gamma,
                current_index=current_index,
            ),
            (keys, values),
        )
    return _create_retention_update_cache(
        keys, values, gamma=gamma, current_index=current_index
    )


_retention_update = jax2tf.convert(jax.jit(jax_retention.retention_update))


def retention_update(
    query: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    cache: tf.Tensor,
    gamma: tf.Tensor,
) -> tf.Tensor:
    return _retention_update(query, keys, values, cache, gamma)


retention_update.__doc__ = jax_retention.retention_update


@functools.cache
def _get_multi_head_retention(**kwargs):
    def fn(query, keys, values, gamma):
        return jax_retention.multi_head_retention(query, keys, values, gamma, **kwargs)

    return jax2tf.convert(jax.jit(fn))


def multi_head_retention(
    query: tf.Tensor, keys: tf.Tensor, values: tf.Tensor, gamma: tf.Tensor, **kwargs
) -> tf.Tensor:
    return _get_multi_head_retention(**kwargs)(query, keys, values, gamma)


multi_head_retention.__doc__ = jax_retention.multi_head_retention

_create_multi_head_retention_update_cache = jax2tf.convert(
    jax.jit(jax_retention.create_multi_head_retention_update_cache)
)


def create_multi_head_retention_update_cache(
    keys: tf.Tensor, values: tf.Tensor, gamma: tf.Tensor, current_index=None
) -> tf.Tensor:
    return _create_multi_head_retention_update_cache(keys, values, gamma, current_index)


create_multi_head_retention_update_cache.__doc__ = (
    jax_retention.create_multi_head_retention_update_cache
)

_multi_head_retention_update = jax2tf.convert(
    jax.jit(jax_retention.multi_head_retention_update)
)


def multi_head_retention_update(
    query: tf.Tensor,
    keys: tf.Tensor,
    values: tf.Tensor,
    cache: tf.Tensor,
    gamma: tf.Tensor,
) -> tf.Tensor:
    return _multi_head_retention_update(query, keys, values, cache, gamma)


multi_head_retention_update.__doc__ = jax_retention.multi_head_retention_update
