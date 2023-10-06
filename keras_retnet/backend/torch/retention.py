import functools
import os

import jax
import jax2torch
import torch

from ..jax import retention as jax_retention

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@functools.cache
def _get_retention(**kwargs):
    def retention_fn(query, keys, values, gamma):
        return jax_retention.retention(query, keys, values, gamma, **kwargs)

    return jax2torch.jax2torch(jax.jit(retention_fn))


def retention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    gamma: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    return _get_retention(**kwargs)(query, keys, values, gamma)


retention.__doc__ = jax_retention.retention

_create_retention_update_cache = jax2torch.jax2torch(
    jax.jit(jax_retention.create_retention_update_cache)
)


def create_retention_update_cache(
    keys: torch.Tensor, values: torch.Tensor, gamma: torch.Tensor, current_index=None
) -> torch.Tensor:
    return _create_retention_update_cache(keys, values, gamma, current_index)


create_retention_update_cache.__doc__ = jax_retention.create_retention_update_cache

_retention_update = jax2torch.jax2torch(jax.jit(jax_retention.retention_update))


def retention_update(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cache: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    return _retention_update(query, keys, values, cache, gamma)


retention_update.__doc__ = jax_retention.retention_update


@functools.cache
def _get_multi_head_retention(**kwargs):
    def fn(query, keys, values, gamma):
        return jax_retention.multi_head_retention(query, keys, values, gamma, **kwargs)

    return jax2torch.jax2torch(jax.jit(fn))


def multi_head_retention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    gamma: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    return _get_multi_head_retention(**kwargs)(query, keys, values, gamma)


multi_head_retention.__doc__ = jax_retention.multi_head_retention

_create_multi_head_retention_update_cache = jax2torch.jax2torch(
    jax.jit(jax_retention.create_multi_head_retention_update_cache)
)


def create_multi_head_retention_update_cache(
    keys: torch.Tensor, values: torch.Tensor, gamma: torch.Tensor, current_index=None
) -> torch.Tensor:
    return _create_multi_head_retention_update_cache(keys, values, gamma, current_index)


create_multi_head_retention_update_cache.__doc__ = (
    jax_retention.create_multi_head_retention_update_cache
)

_multi_head_retention_update = jax2torch.jax2torch(
    jax.jit(jax_retention.multi_head_retention_update)
)


def multi_head_retention_update(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cache: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    return _multi_head_retention_update(query, keys, values, cache, gamma)


multi_head_retention_update.__doc__ = jax_retention.multi_head_retention_update
