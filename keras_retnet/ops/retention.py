import typing as tp

from keras_core import KerasTensor, Operation

from ..backend import Array, ops
from ..backend import retention as retention_backend

# pylint:disable=arguments-differ


class Retention(Operation):
    def __init__(
        self, implementation: str = "block", kwargs=None, name: tp.Optional[str] = None
    ):
        self.implementation = implementation
        self.kwargs = kwargs or {}
        super().__init__(name)

    def compute_output_spec(self, query, keys, values, gamma):
        del keys, gamma
        return KerasTensor((*query.shape[:-1], values.shape[-1]), values.dtype)

    def call(self, query, keys, values, gamma):
        return retention_backend.retention(
            query,
            keys,
            values,
            gamma,
            implementation=self.implementation,
            **self.kwargs,
        )


def retention(
    query: Array,
    keys: Array,
    values: Array,
    gamma: Array,
    *,
    implementation: str = "block",
    **kwargs,
) -> Array:
    return Retention(implementation=implementation, kwargs=kwargs)(
        query, keys, values, gamma
    )


class MultiHeadRetention(Operation):
    def __init__(
        self, implementation: str = "block", kwargs=None, name: tp.Optional[str] = None
    ):
        self.implementation = implementation
        self.kwargs = kwargs or {}
        super().__init__(name)

    def compute_output_spec(self, query, keys, values, gamma):
        del keys, gamma
        return KerasTensor(
            (*query.shape[:-1], values.shape[-1]), values.dtype, name="retention"
        )

    def call(self, query, keys, values, gamma):
        return retention_backend.multi_head_retention(
            query,
            keys,
            values,
            gamma,
            implementation=self.implementation,
            **self.kwargs,
        )


def multi_head_retention(
    query: Array,
    keys: Array,
    values: Array,
    gamma: Array,
    *,
    implementation: str = "block",
    **kwargs,
) -> Array:
    return MultiHeadRetention(implementation=implementation, kwargs=kwargs)(
        query, keys, values, gamma
    )


# class CreateRetentionUpdateCache(Operation):
#     def call(
#         self,
#         keys: Array,
#         values: Array,
#         gamma: Array,
#         current_index: tp.Optional[Array] = None,
#     ):
#         return retention_backend.create_retention_update_cache(
#             keys, values, gamma, current_index=current_index
#         )

#     def compute_output_spec(
#         self,
#         keys: KerasTensor,
#         values: KerasTensor,
#         gamma: KerasTensor,
#         current_index: tp.Optional[KerasTensor] = None,
#     ) -> KerasTensor:
#         if len(keys.shape) == 3:
#             # includes batch dim
#             shape = (keys.shape[0], keys.shape[1], values.shape[1])
#         else:
#             shape = (keys.shape[-1], values.shape[-1])
#         return KerasTensor(shape, dtype=keys.dtype, name="retention_cache")


# def create_retention_update_cache(
#     keys: Array, values: Array, gamma: Array, current_index: tp.Optional[Array] = None
# ) -> Array:
#     return CreateRetentionUpdateCache()(keys, values, gamma, current_index)


def create_retention_update_cache(
    keys: Array,
    values: Array,
    gamma: Array,
    current_index: tp.Optional[Array] = None,
) -> Array:
    """
    Args:
        keys: [B?, T, d_k]
        values: [B?, T, d_v]
        gamma: []
        current_index: [] int, current_index < T. If None, implied to be T-1

    Returns:
        updated_cache: [B?, d_k, d_v]
    """
    assert len(keys.shape) == len(values.shape), (keys.shape, values.shape)
    assert keys.shape[-2] == values.shape[-2], (keys.shape, values.shape)
    assert len(gamma.shape) == 0, gamma.shape

    T = keys.shape[-2]

    if current_index is None:
        factors = gamma ** ops.arange(T - 1, -1, -1, dtype=gamma.dtype)
    else:
        t_range = ops.arange(T)
        padding = t_range > current_index
        factors = gamma ** (ops.arange(0, -T, -1, dtype=gamma.dtype) + current_index)
        # factors = gamma ** jnp.arange(current_index, current_index - T, -1)
        factors = ops.where(padding, ops.zeros_like(factors), factors)
    # out = tf.einsum("tk,t,tv->kv", keys, factors, values)
    out = ops.matmul(ops.swapaxes(keys, -2, -1), values * factors[:, None])
    return out


# class RetentionUpdate(Operation):
#     def call(
#         self, query: Array, keys: Array, values: Array, cache: Array, gamma: Array
#     ) -> Array:
#         return retention_backend.retention_update(query, keys, values, cache, gamma)

#     def compute_output_spec(
#         self, query: Array, keys: Array, values: Array, cache: Array, gamma: Array
#     ):
#         v_out = KerasTensor(shape=values.shape, dtype=values.dtype, name="values_out")
#         s_out = KerasTensor(shape=cache.shape, dtype=cache.dtype, name="cache_out")
#         return v_out, s_out


# def retention_update(
#     query: Array, keys: Array, values: Array, cache: Array, gamma: Array
# ) -> tp.Tuple[Array, Array]:
#     return RetentionUpdate()(query, keys, values, cache, gamma)


def retention_update(
    query: Array,
    keys: Array,
    values: Array,
    cache: Array,
    gamma: Array,
) -> tp.Tuple[Array, Array]:
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
    cache = gamma * cache + ops.expand_dims(keys, -1) * ops.expand_dims(values, -2)
    # result = jnp.einsum("k,kv->v", query, cache)
    if len(query.shape) == 1:
        result = ops.matmul(query, cache)
    else:
        result = ops.einsum("bk,bkv->bv", query, cache)
    return result, cache


# class CreateMultiHeadRetentionUpdateCache(Operation):
#     def call(
#         self,
#         keys: Array,
#         values: Array,
#         gamma: Array,
#         current_index: tp.Optional[Array] = None,
#     ):
#         return retention_backend.create_multi_head_retention_update_cache(
#             keys, values, gamma, current_index=current_index
#         )

#     def compute_output_spec(
#         self,
#         keys: KerasTensor,
#         values: KerasTensor,
#         gamma: KerasTensor,
#         current_index: tp.Optional[KerasTensor] = None,
#     ) -> KerasTensor:
#         if len(keys.shape) == 3:
#             # includes batch dim
#             shape = (keys.shape[0], keys.shape[1], values.shape[1])
#         else:
#             shape = (keys.shape[-1], values.shape[-1])
#         return KerasTensor(shape, dtype=keys.dtype, name="retention_cache")


# def create_multi_head_retention_update_cache(
#     keys: Array, values: Array, gamma: Array, current_index: tp.Optional[Array] = None
# ) -> Array:
#     return CreateMultiHeadRetentionUpdateCache()(
#         keys, values, gamma, current_index=current_index
#     )
def create_multi_head_retention_update_cache(
    keys: Array,
    values: Array,
    gamma: Array,
    current_index: tp.Optional[Array] = None,
) -> Array:
    """
    Args:
        keys: [B?, T, H, d_k]
        values: [B?, T, H, d_v]
        gamma: [H]
        current_index: [] int, current_index < T. If None, implied to be T-1

    Returns:
        updated_cache: [B?, H, d_k, d_v]
    """
    assert len(keys.shape) == len(values.shape), (keys.shape, values.shape)
    assert keys.shape[-3] == values.shape[-3], (keys.shape, values.shape)
    assert keys.shape[-2] == values.shape[-2] == gamma.shape[0], (
        keys.shape,
        values.shape,
        gamma.shape,
    )
    assert len(gamma.shape) == 1, gamma.shape

    T = keys.shape[-3]

    if current_index is None:
        factors = gamma ** ops.arange(T - 1, -1, -1, dtype=gamma.dtype)[:, None]
    else:
        t_range = ops.arange(T)
        padding = t_range > current_index
        factors = gamma ** (
            ops.arange(0, -T, -1, dtype=gamma.dtype)[:, None] + current_index
        )
        # factors = gamma ** jnp.arange(current_index, current_index - T, -1)
        factors = ops.where(padding[:, None], ops.zeros_like(factors), factors)
    if len(values.shape) == 4:
        pattern = "bthk,th,bthv->bhkv"
    else:
        pattern = "thk,th,thv->hkv"
    out = ops.einsum(pattern, keys, factors, values)
    # out = ops.matmul(ops.swapaxes(keys, -2, -1), values * factors[..., None])
    return out


# class MultiHeadRetentionUpdate(Operation):
#     def call(
#         self, query: Array, keys: Array, values: Array, cache: Array, gamma: Array
#     ) -> Array:
#         return retention_backend.multi_head_retention_update(
#             query, keys, values, cache, gamma
#         )

#     def compute_output_spec(
#         self, query: Array, keys: Array, values: Array, cache: Array, gamma: Array
#     ):
#         v_out = KerasTensor(shape=values.shape, dtype=values.dtype, name="values_out")
#         s_out = KerasTensor(shape=cache.shape, dtype=cache.dtype, name="cache_out")
#         return v_out, s_out


# def multi_head_retention_update(
#     query: Array, keys: Array, values: Array, cache: Array, gamma: Array
# ) -> tp.Tuple[Array, Array]:
#     return MultiHeadRetentionUpdate()(query, keys, values, cache, gamma)


def multi_head_retention_update(
    query: Array,
    keys: Array,
    values: Array,
    cache: Array,
    gamma: Array,
) -> tp.Tuple[Array, Array]:
    """
    Args:
        query: [B?, H, d_k]
        keys: [B?, H, d_k]
        values: [B?, H, d_v]
        cache: [B?, H, d_k, d_v]
        gamma: [H]

    Returns:
        updated_values: [B?, d_v]
        updated_cache: [B?, d_k, d_v]
    """
    cache = gamma[:, None, None] * cache + ops.expand_dims(keys, -1) * ops.expand_dims(
        values, -2
    )
    if len(query.shape) == 2:
        result = ops.einsum("hk,hkv->hv", query, cache)
    else:
        result = ops.einsum("bhk,bhkv->bhv", query, cache)
    return result, cache
