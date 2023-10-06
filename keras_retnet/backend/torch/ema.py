import os

import jax
import jax2torch
import torch

from ..jax import ema as jax_ema

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_cumulative_ema = jax2torch.jax2torch(
    jax.jit(jax_ema.cumulative_ema, static_argnames=("reverse", "axis"))
)


def cumulative_ema(
    values: torch.Tensor, factors: torch.Tensor, reverse: bool = False, axis: int = 0
) -> torch.Tensor:
    result = _cumulative_ema(values, factors, reverse, axis)
    return result


cumulative_ema.__doc__ = jax_ema.cumulative_ema.__doc__


_segment_cumulative_ema = jax2torch.jax2torch(
    jax.jit(jax_ema.segment_cumulative_ema, static_argnames=("reverse", "axis"))
)


def segment_cumulative_ema(
    values: torch.Tensor,
    factors: torch.Tensor,
    segment_ids: torch.Tensor,
    reverse: bool = False,
    axis: int = 0,
) -> torch.Tensor:
    return _segment_cumulative_ema(values, factors, segment_ids, reverse, axis)


segment_cumulative_ema.__doc__ = jax_ema.segment_cumulative_ema.__doc__

_reduce_ema = jax2torch.jax2torch(
    jax.jit(jax_ema.segment_cumulative_ema, static_argnames=("reverse", "axis"))
)


def reduce_ema(
    values: torch.Tensor,
    factors: torch.Tensor,
    reverse: bool = False,
    axis: int = 0,
) -> torch.Tensor:
    return _reduce_ema(values, factors, reverse, axis)


reduce_ema.__doc__ = jax_ema.reduce_ema.__doc__
