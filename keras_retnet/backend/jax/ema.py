"""ema ops in jax."""
import typing as tp

import jax
import jax.numpy as jnp

from . import scan as scan_lib

Pair = tp.Tuple[jnp.ndarray, jnp.ndarray]


def _cumulative_ema_op(a: Pair, b: Pair) -> Pair:
    xa, fa = a
    xb, fb = b
    return xa * fb + xb, fa * fb


def _pad_on_axis(
    x: jnp.ndarray, pad_width: tp.Tuple[int, int], axis: int
) -> jnp.ndarray:
    padding = [(0, 0) for _ in x.shape]
    padding[axis] = pad_width
    return jnp.pad(x, padding)


def reduce_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    reverse: bool = False,
    axis: int = 0,
) -> jnp.ndarray:
    """
    Compute exponential moving average.

    Same as `cumulative_ema` but returns only the last (or first if revers is True)
    entry along the specified axis.

    Args:
        values: N-D float values
        factors: same shape/dtype as values, or may have 1 fewer elements on axis
            dimension.
        axis: the axis to compute exponential moving average along.
        reverse: if True, perform accumulation in reverse.

    Returns:
        reduced ema values, same shape as values/factors but without `axis` dimension.
    """
    values = jnp.asarray(values)
    factors = jnp.asarray(factors)
    if axis < 0:
        axis += len(values.shape)
    assert values.shape == factors.shape, (values.shape, factors.shape)
    if reverse:
        values = jnp.flip(values, axis)
        factors = jnp.flip(factors, axis)
    if factors.shape[axis] == values.shape[axis] - 1:
        factors = _pad_on_axis(factors, (1, 0), axis)
    element_shape = list(values.shape)
    del element_shape[axis]
    dtype = values.dtype
    init = (jnp.zeros((), dtype), jnp.ones((), dtype))
    f, t = jax.lax.reduce(
        (values, factors), init, _cumulative_ema_op, dimensions=(axis,)
    )
    del t
    return f


def cumulative_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    reverse: bool = False,
    axis: int = 0,
    parallel: bool = True,
) -> jnp.ndarray:
    """
    Compute cumulative exponential moving average.

    When `reverse == False` and axis == 0 and factors.shape == values.shape this
    corresponds to:
        output[0] = values[0]
        output[i+1] = output[i] * factors[i+1] + values[i+1]

    Note that factors[0] is never used, so this function also accepts `factors` one
    shorter in the `axis` dimension, in which case it is padded (either at the start
    if `reverse==False` or at the end if `reverse==True`).

    If `reverse == True`, then the result equivalent to the reverse of the non-reversed
    call on arguments reversed on the given axis, assuming
    `factors.shape == values.shape`.

    Args:
        values: N-D float values
        factors: same dtype as values. shapes must be broadcastable, and consistent
            on `axis` dimension or one less.
        axis: the axis to compute exponential moving average along.
        reverse: if True, perform accumulation in reverse.
        parallel: if True, uses `jax.lax.associative_scan`, otherwise regular
            `jax.lax.scan`.

    Returns:
        cumulative ema values, same shape as values/factors.
    """
    values = jnp.asarray(values)
    factors = jnp.asarray(factors)
    if axis < 0:
        axis += len(values.shape)
    if factors.shape[axis] == values.shape[axis] - 1:
        factors = _pad_on_axis(factors, (0, 1) if reverse else (1, 0), axis)
    assert values.dtype == factors.dtype, (values.dtype, factors.dtype)
    if parallel:
        f, t = jax.lax.associative_scan(
            _cumulative_ema_op, (values, factors), reverse=reverse, axis=axis
        )
    else:

        def f(carry, x):
            out = _cumulative_ema_op(carry, x)
            return out, out

        shape = list(values.shape)
        del shape[axis]
        init = (jnp.zeros(shape, values.dtype), jnp.ones(shape, values.dtype))
        _, (f, t) = jax.lax.scan(f, init, (values, factors), reverse=reverse)
    del t
    return f


def times_to_factors(times: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.exp(-jnp.diff(times, axis=axis))


def factors_to_times(factors: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jnp.cumsum(-jnp.log(factors), axis=axis)


def symmetric_cumulative_ema_from_factors(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    axis: int = 0,
) -> jnp.ndarray:
    values = jnp.asarray(values)
    factors = jnp.asarray(factors)
    assert values.shape[axis] == factors.shape[axis] + 1, (
        values.shape[axis],
        factors.shape[axis],
    )
    forward = cumulative_ema(values, factors, axis=axis)
    reverse = cumulative_ema(values, factors, axis=axis, reverse=True)
    combined = forward + reverse - values
    return combined


# def symmetric_cumulative_ema(
#     values: jnp.ndarray,
#     times: jnp.ndarray,
#     axis: int = 0,
#     times_are_sorted: bool = False,
# ) -> jnp.ndarray:
#     """
#     Compute symmetric cumulative exponential moving average.

#     In the 1D case, the output satisfies:

#     ```python
#     output[i] = jnp.sum(jnp.exp(-jnp.abs(times - times[i])) * values)
#     ```

#     Args:
#         values: values to compute EMA of.
#         times: same shape/dtype as values.
#         axis: axis along which to compute EMA.

#     Returns:
#         output: same shape/dtype as values.
#     """
#     values = jnp.asarray(values)
#     times = jnp.asarray(times)
#     if not times_are_sorted:
#         # sort times/values along times
#         order = jnp.argsort(times, axis=axis)
#         # apply permutation
#         values = jnp.take_along_axis(values, order, axis=axis)
#         times = jnp.take_along_axis(times, order, axis=axis)

#     factors = times_to_factors(times, axis=axis)
#     accumulated = symmetric_cumulative_ema_from_factors(values, factors, axis=axis)

#     if not times_are_sorted:
#         accumulated = array_ops.scatter_along_axis(accumulated, order, axis=axis)
#     return accumulated


def segment_cumulative_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    segment_ids: jnp.ndarray,
    reverse: bool = False,
    axis: int = 0,
) -> jnp.ndarray:
    """
    Compute segmented cumulative exponential moving average.

    Args:
        values: N-D float values
        factors: same shape/dtype as values
        segment_ids: integers with shape [values.shape[axis]]
        reverse: if True, EMA is computed in reverse
        axis: axis of (values, factors) to compute EMA on

    Returns:
        Same shape as values/factors/segment_ids
    """
    values = jnp.asarray(values)
    factors = jnp.asarray(factors)
    segment_ids = jnp.asarray(segment_ids)
    f, t = scan_lib.segment_associative_scan(
        _cumulative_ema_op, segment_ids, (values, factors), reverse=reverse, axis=axis
    )
    del t
    return f
