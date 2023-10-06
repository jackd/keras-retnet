import functools
import typing as tp

import jax
import jax.numpy as jnp

from . import ema


def _get_retention_D(gamma, T):
    # returns: [T, T]
    T_range = jnp.arange(T, dtype=gamma.dtype)
    diff = jnp.expand_dims(T_range, 1) - jnp.expand_dims(T_range, 0)
    D = gamma**diff
    D = jnp.where(diff >= 0, D, jnp.zeros_like(D))
    return D


def _get_retention_coeff(
    query: jnp.ndarray, keys: jnp.ndarray, gamma: jnp.ndarray
) -> jnp.ndarray:
    assert len(query.shape) == 2, query.shape
    assert len(keys.shape) == 2, keys.shape
    assert gamma.shape == (), gamma.shape
    QK = query @ keys.T
    D = _get_retention_D(gamma, query.shape[0])
    return QK * D


def _retention_block(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    rescale: bool = True,
) -> jnp.ndarray:
    """
    Args:
        query: [T, d_k]
        keys: [T, d_k]
        values: [T, d_v]
        gamma: scalar

    Returns:
        [T, d_v]
    """
    QKD = _get_retention_coeff(query, keys, gamma)
    if rescale:
        norm_factor = jnp.abs(jnp.sum(QKD, axis=-1, keepdims=True))
        norm_factor = jnp.maximum(norm_factor, jnp.ones_like(norm_factor))
        QKD = QKD / norm_factor
    return jnp.matmul(QKD, values)


def _retention_scan(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    current_index: tp.Optional[jnp.ndarray] = None,
    *,
    parallel: bool = True,
) -> jnp.ndarray:
    """
    Args:
        query: [T, d_k]
        keys: [T, d_k]
        values: [T, d_v]
        gamma: scalar

    Returns:
        [T, d_v]
    """
    assert len(query.shape) == 2, query.shape
    assert len(keys.shape) == 2, keys.shape
    assert len(values.shape) == 2, values.shape
    assert gamma.shape == (), gamma.shape
    KV = jnp.expand_dims(keys, -1) * jnp.expand_dims(values, -2)  # [T, d_k, d_v]
    gamma = jnp.full((query.shape[0], 1, 1), gamma)
    acc = ema.cumulative_ema(KV, gamma, parallel=parallel)  # [T, d_k, d_v]
    result = jnp.einsum("tk,tkv->tv", query, acc)
    if current_index is None:
        return result
    acc = jnp.expand_dims(acc[current_index], 0)
    return result, acc


def _retention_update(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    cache: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        query: [d_k]
        keys: [d_k]
        values: [d_v]
        cache: [d_k, d_v]
        gamma: scalar

    Returns:
        updated_values: [d_v]
        updated_cache: [d_k, d_v]
    """
    cache = gamma * cache + keys[:, None] * values
    # result = jnp.einsum("k,kv->v", query, cache)
    result = jnp.matmul(query, cache)
    return result, cache


def retention_update(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    cache: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        query: [B?, d_k]
        keys: [B?, d_k]
        values: [B?, d_v]
        cache: [B?, d_k, d_v]
        gamma: scalar

    Returns:
        updated_values: [B?, d_v]
        updated_cache: [B?, d_k, d_v]
    """
    if query.ndim == 2:
        return jax.vmap(functools.partial(_retention_update, gamma=gamma))(
            query, keys, values, cache
        )
    return _retention_update(query, keys, values, cache, gamma)


def _create_retention_update_cache(
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    current_index: tp.Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Args:
        keys: [T, d_k]
        values: [T, d_v]
        gamma: []
        current_index: [] int, current_index < T. If None, implied to be T-1

    Returns:
        updated_cache: [B?, d_k, d_v]
    """
    assert keys.ndim == 2, keys.shape
    assert values.ndim == 2, values.shape
    assert keys.shape[0] == values.shape[0], (keys.shape, values.shape)
    assert gamma.ndim == 0, gamma.shape

    T = keys.shape[0]
    if current_index is None:
        factors = gamma ** jnp.arange(T - 1, -1, -1, dtype=gamma.dtype)
    else:
        t_range = jnp.arange(T)
        padding = t_range > current_index
        factors = gamma ** (jnp.arange(0, -T, -1, dtype=gamma.dtype) + current_index)
        # factors = gamma ** jnp.arange(current_index, current_index - T, -1)
        factors = jnp.where(padding, jnp.zeros_like(factors), factors)
    # out = jnp.einsum("tk,t,tv->kv", keys, factors, values)
    out = jnp.matmul(keys.T, values * factors[:, None])
    return out


def create_retention_update_cache(
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    current_index: tp.Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
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
        return jax.vmap(
            functools.partial(
                _create_retention_update_cache,
                gamma=gamma,
                current_index=current_index,
            )
        )(keys, values)
    return _create_retention_update_cache(
        keys, values, gamma=gamma, current_index=current_index
    )


def _retention_scan_v2(
    query: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, gamma: jnp.ndarray
):
    """
    Args:
        query: [T, d_k]
        keys: [T, d_k]
        values: [T, d_v]
        gamma: scalar

    Returns:
        [T, d_v]
    """
    T, d_k = query.shape
    assert keys.shape == (T, d_k), ((T, d_k), keys.shape)
    assert values.shape[:-1] == (T,), (values.shape, T)
    assert gamma.shape == (), gamma.shape
    d_v = values.shape[-1]
    cache = jnp.zeros((d_k, d_v), dtype=values.dtype)

    def fn(cache, args):
        query, keys, values = args
        y, cache = retention_update(query, keys, values, cache, gamma)
        return cache, y

    _, result = jax.lax.scan(fn, cache, (query, keys, values))
    return result


def _retention_chunkwise_recurrent_v2(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    chunk_size: int = 64,
    parallel: bool = True,
) -> jnp.ndarray:
    T, d_k = query.shape
    assert T % chunk_size == 0, (T, chunk_size)
    assert keys.shape == (T, d_k), ((T, d_k), keys.shape)
    assert values.shape[:-1] == (T,), (values.shape, T)
    assert gamma.shape == (), gamma.shape
    d_v = values.shape[-1]
    num_chunks = T // chunk_size

    query = jnp.reshape(query, (num_chunks, chunk_size, d_k))
    keys = jnp.reshape(keys, (num_chunks, chunk_size, d_k))
    values = jnp.reshape(values, (num_chunks, chunk_size, d_v))

    kv = jnp.einsum(
        "csk,s,csv->ckv",
        keys[:-1],
        gamma ** jnp.arange(chunk_size, 0, -1),
        values[:-1],
    )

    kv = ema.cumulative_ema(
        kv, jnp.full((num_chunks - 1, 1, 1), gamma**chunk_size), parallel=parallel
    )
    kv = jnp.pad(kv, ((1, 0), (0, 0), (0, 0)))

    inner_decay = gamma ** jnp.arange(chunk_size)
    cross_output = jnp.einsum("csk,s,ckv->csv", query, inner_decay, kv)
    inner_output = jax.vmap(
        functools.partial(_retention_block, gamma=gamma, rescale=False)
    )(query, keys, values)
    output = inner_output + cross_output
    return jnp.reshape(output, (T, d_v))


def _retention_chunkwise_recurrent(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """
    Args:
        query: [T, d_k]
        keys: [T, d_k]
        values: [T, d_v]
        gamma: scalar
        chunk_size: int size of each chunk. `T` must be divisible by this.
        parallel: whether to accumulate final chunk elements in parallel or series.

    Returns:
        [T, d_v]
    """
    T, d_k = query.shape
    assert T % chunk_size == 0, (T, chunk_size)
    assert keys.shape == (T, d_k), ((T, d_k), keys.shape)
    assert values.shape[:-1] == (T,), (values.shape, T)
    assert gamma.shape == (), gamma.shape
    d_v = values.shape[-1]
    num_chunks = T // chunk_size

    query = jnp.reshape(query, (num_chunks, chunk_size, d_k))
    keys = jnp.reshape(keys, (num_chunks, chunk_size, d_k))
    values = jnp.reshape(values, (num_chunks, chunk_size, d_v))

    qk_mat = jnp.einsum(  # [num_chunks, chunk_size, chunk_size]
        "csk,ctk->cst", query, keys
    )

    mask = _get_retention_D(gamma, chunk_size)  # [chunk_size, chunk_size]
    scale = jnp.sqrt(jnp.sum(mask, axis=-1, keepdims=True))  # [chunk_size, 1]
    mask = mask / scale
    qk_mat = qk_mat * mask  # [num_chunks, chunk_size, chunk_size]

    inner_decay = gamma ** (
        jnp.arange(chunk_size, dtype=gamma.dtype) + 1
    )  # [chunk_size]
    # NOTE: there's an issue with jax2torch in the line below...
    # inner_decay = inner_decay[:, None] / (scale / scale[-1:])  # [chunk_size, 1]
    inner_decay = inner_decay[:, None] / scale  # [chunk_size, 1]
    inner_scale = jnp.sum(
        jnp.abs(qk_mat), axis=-1, keepdims=True
    )  # [num_chunks, chunk_size, 1]
    inner_scale = jnp.maximum(inner_scale, jnp.ones_like(inner_scale))
    qk_mat = qk_mat / inner_scale
    inner_output = jnp.einsum("cst,ctv->csv", qk_mat, values)

    # reduce kv in one chunk
    kv = jnp.einsum("csk,csv->ckv", keys, values * mask[-1, :, None])

    kv_cache = jnp.zeros((d_k, d_v), dtype=kv.dtype)
    kv_scale = jnp.ones((1, 1), dtype=kv.dtype)
    cross_decay = jnp.reshape(gamma**chunk_size, (1, 1))

    def f(carry, kvi):
        kv_cache, kv_scale = carry
        kv_recurrent = kv_cache / kv_scale
        cross_scale = kv_scale
        kv_cache = kv_cache * cross_decay + kvi
        kv_scale = jnp.max(
            jnp.sum(jnp.abs(kv_cache), axis=0, keepdims=True), axis=1, keepdims=True
        )
        kv_scale = jnp.maximum(kv_scale, jnp.ones_like(kv_scale))

        y = kv_recurrent, cross_scale
        carry = kv_cache, kv_scale

        return carry, y

    _, (kv_recurrent, cross_scale) = jax.lax.scan(f, (kv_cache, kv_scale), kv)

    all_scale = jnp.maximum(inner_scale, cross_scale)  # [num_chunks, chunk_size, 1]
    align_inner_scale = all_scale / inner_scale  # [num_chunks, chunk_size, 1]
    align_cross_scale = all_scale / cross_scale  # [num_chunks, chunk_size, 1]

    cross_output = jnp.einsum("csk,ckv->csv", query * inner_decay, kv_recurrent)
    # line below is a work-around to the NOTE above about jax2torch
    cross_output = cross_output * scale[-1]
    output = inner_output / align_inner_scale + cross_output / align_cross_scale

    output = jnp.reshape(output, (T, d_v))
    return output


def retention(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    implementation: str = "block",
    **kwargs,
) -> jnp.ndarray:
    """
    Retention with optional batch dimension.

    Args:
        query: [B?, T, d_k]
        keys: [B?, T, d_k]
        values: [B?, T, d_v]
        gamma: scalar
        implementation: either "block", "scan" or "chunkwise_recurrent"
        **kwargs: passed to the relevant scan function

    Returns:
        [B?, T, d_v]
    """
    query = jnp.asarray(query)
    keys = jnp.asarray(keys)
    values = jnp.asarray(values)
    gamma = jnp.asarray(gamma)
    func = {
        "block": _retention_block,
        "scan": _retention_scan,
        "chunkwise_recurrent": _retention_chunkwise_recurrent,
        "chunkwise_recurrent_v2": _retention_chunkwise_recurrent_v2,
        "scan_v2": _retention_scan_v2,
    }[implementation]
    func = functools.partial(func, gamma=gamma, **kwargs)

    if len(query.shape) == 3:
        func = jax.vmap(func, in_axes=0, out_axes=0)
    return func(query, keys, values)


def _unbatched_multi_head_retention(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    implementation: str = "block",
    **kwargs,
) -> jnp.ndarray:
    """
    Multi-head retention with optional batch dimension for all args except gamma.

    Args:
        query: [T, H, d_k]
        keys: [T, H, d_k]
        values: [T, H, d_v]
        gamma: [H]

    Returns:
        [T, H, d_v]
    """
    return jax.vmap(
        functools.partial(retention, implementation=implementation, **kwargs),
        in_axes=(1, 1, 1, 0),
        out_axes=1,
    )(query, keys, values, gamma)


def multi_head_retention(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    gamma: jnp.ndarray,
    implementation: str = "block",
    **kwargs,
) -> jnp.ndarray:
    """
    Multi-head retention with optional batch dimension for all args except gamma.

    Args:
        query: [B?, T, H, d_k]
        keys: [B?, T, H, d_k]
        values: [B?, T, H, d_v]
        gamma: [H]

    Returns:
        [B?, T, H, d_v]
    """
    ndim = query.ndim
    if ndim == 4:
        # batched
        return jax.vmap(
            functools.partial(
                _unbatched_multi_head_retention,
                gamma=gamma,
                implementation=implementation,
                **kwargs,
            )
        )(query, keys, values)

    assert ndim == 3, query.shape
    return _unbatched_multi_head_retention(
        query, keys, values, gamma, implementation=implementation, **kwargs
    )


def multi_head_retention_update(
    query: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    cache: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        query: [B?, H, d_k]
        keys: [B?, H, d_k]
        values: [B?, H, d_v]
        cache: [B?, H, d_k, d_v]
        gamma: [H]

    Returns:
        updated_values: [B?, H, d_v]
        updated_cache: [B?, H, d_k, d_v]
    """
    if query.ndim == 3:
        return jax.vmap(functools.partial(multi_head_retention_update, gamma=gamma))(
            query, keys, values, cache
        )
    assert query.ndim == 2, query.shape
    assert keys.ndim == 2, keys.shape
    assert values.ndim == 2, values.shape
    assert cache.ndim == 3, cache.shape
    assert gamma.ndim == 1, gamma.shape
    return jax.vmap(retention_update)(query, keys, values, cache, gamma)


def create_multi_head_retention_update_cache(
    keys: jnp.ndarray, values: jnp.ndarray, gamma: jnp.ndarray, current_index=None
) -> jnp.ndarray:
    """
    Args:
        keys: [B?, T, H, d_k]
        values: [B?, T, H, d_v]
        gamma: [H]

    Returns:
        cache: [B?, H, d_k, d_v]
    """
    if keys.ndim == 4:
        return jax.vmap(
            functools.partial(
                create_multi_head_retention_update_cache,
                gamma=gamma,
                current_index=current_index,
            )
        )(keys, values)

    assert keys.ndim == 3, keys.shape
    assert values.ndim == 3, values.shape
    assert gamma.ndim == 1, gamma.shape

    return jax.vmap(
        functools.partial(create_retention_update_cache, current_index=current_index),
        in_axes=(1, 1, 0),
    )(keys, values, gamma)
