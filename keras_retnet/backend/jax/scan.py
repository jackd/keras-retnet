import typing as tp

import jax
import jax.numpy as jnp

from ._flax_scan import scan  # pylint:disable=unused-import


def segment_associative_scan(
    fn: tp.Callable, segments: jnp.ndarray, elems, reverse: bool = False, axis: int = 0
):
    """
    Compute associative_scan grouped by segments.

    This is equivalent to partitioning elems by segments, performing an associative_scan
    on each partition and then concatenating the results in the order in which they
    appear in segments (not necessarily their values).

    `segments` are assumed to be grouped, e.g. [0, 0, 0, 2, 2, 1, 1, 1, 1, 3]

    Args:
        fn: associative binary operation.
        segments: [N] values defining partitions. Partitions are assumed to be grouped.
        elems: structure of arrays, where each array has shape[axis] == N.
        reverse: if True, associative scan is performed in reverse.
        axis: axis over which to scan.

    Returns:
        elements with the same structure as `elems`

    See also:
        jax.lax.associative_scan.
    """

    def wrapped_fn(a, b):
        segments_a, elems_a = a
        segments_b, elems_b = b
        elems_c = fn(elems_a, elems_b)
        mask = segments_a == segments_b
        elems_c = jax.tree_util.tree_map(
            lambda elem_b, elem_c: jnp.where(mask, elem_c, elem_b), elems_b, elems_c
        )
        return (segments_b, elems_c)

    args = (segments, elems)
    _, out = jax.lax.associative_scan(wrapped_fn, args, reverse=reverse, axis=axis)
    return out
