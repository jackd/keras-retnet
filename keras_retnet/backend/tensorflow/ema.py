import tensorflow as tf

from ..jax import ema as jax_ema
from .jax2tf_utils import convert_and_compile


def cumulative_ema(
    values: tf.Tensor, factors: tf.Tensor, reverse: bool = False, axis: int = 0
) -> tf.Tensor:
    ndims = len(values.shape)
    if axis < 0:
        axis += ndims
    assert 0 <= axis < ndims, (axis, values.shape)
    return convert_and_compile(jax_ema.cumulative_ema, reverse=reverse, axis=axis)(
        values, factors
    )


cumulative_ema.__doc__ = jax_ema.cumulative_ema.__doc__


def segment_cumulative_ema(
    values: tf.Tensor,
    factors: tf.Tensor,
    segment_ids: tf.Tensor,
    reverse: bool = False,
    axis: int = 0,
) -> tf.Tensor:
    ndims = len(values.shape)
    if axis < 0:
        axis += ndims
    assert 0 <= axis < ndims, (axis, values.shape)
    assert values.shape == factors.shape, (values.shape, factors.shape)
    return convert_and_compile(
        jax_ema.segment_cumulative_ema, reverse=reverse, axis=axis
    )(values, factors, segment_ids)


segment_cumulative_ema.__doc__ = jax_ema.segment_cumulative_ema.__doc__


def reduce_ema(
    values: tf.Tensor,
    factors: tf.Tensor,
    reverse: bool = False,
    axis: int = 0,
) -> tf.Tensor:
    ndims = len(values.shape)
    if axis < 0:
        axis += ndims
    assert 0 <= axis < ndims, (axis, values.shape)
    return convert_and_compile(jax_ema.reduce_ema, reverse=reverse, axis=axis)(
        values, factors
    )


reduce_ema.__doc__ = jax_ema.reduce_ema.__doc__
