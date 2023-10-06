import copy
import math

from ..backend import keras, ops, retention
from ..ops import retention as retention_ops


def _rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = ops.stack((-x2, x1), axis=-1)
    shape = [-1 if d is None else d for d in x.shape[:-2]]
    return ops.reshape(x, (*shape, x.shape[-2] * x.shape[-1]))  # flatten trailing dims


def _theta_shift(x, sin, cos):
    return (x * cos) + (_rotate_every_two(x) * sin)


@keras.utils.register_keras_serializable("keras_retnet")
class MultiScaleRetention(keras.layers.Layer):  # pylint:disable=abstract-method
    def __init__(
        self,
        num_heads: int,
        value_factor: int = 2,
        gate_activation="swish",
        retention_kwargs=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.value_factor = value_factor
        self.retention_kwargs = (
            {} if retention_kwargs is None else copy.deepcopy(retention_kwargs)
        )
        self.gate_activation = keras.activations.get(gate_activation)

    def get_config(self):
        config = super().get_config()
        config.update(
            num_heads=self.num_heads,
            value_factor=self.value_factor,
            retention_kwargs=copy.deepcopy(self.retention_kwargs),
            gate_activation=keras.activations.serialize(self.gate_activation),
        )
        return config

    def build(self, input_shape):
        if self.built:
            return

        super().build(input_shape)
        # pylint:disable=attribute-defined-outside-init
        key_dim = input_shape[-1]
        assert key_dim % (self.num_heads * 2) == 0, (key_dim, self.num_heads)
        value_dim = key_dim * self.value_factor

        self.query_dense = keras.layers.Dense(
            key_dim,
            name="query",
            kernel_initializer=keras.initializers.VarianceScaling(
                mode="fan_avg", distribution="uniform", scale=2**-2.5
            ),
        )
        self.key_dense = keras.layers.Dense(
            key_dim,
            name="key",
            kernel_initializer=keras.initializers.VarianceScaling(
                mode="fan_avg", distribution="uniform", scale=2**-2.5
            ),
        )
        self.value_dense = keras.layers.Dense(
            value_dim,
            name="value",
            kernel_initializer=keras.initializers.VarianceScaling(
                mode="fan_avg", distribution="uniform", scale=2**-2.5
            ),
        )
        self.gate_dense = keras.layers.Dense(
            value_dim,
            name="gate",
            activation=self.gate_activation,
            kernel_initializer=keras.initializers.VarianceScaling(
                mode="fan_avg", distribution="uniform", scale=2**-2.5
            ),
        )
        self.output_dense = keras.layers.Dense(
            key_dim, kernel_initializer="glorot_uniform", name="output"
        )
        self.group_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, center=False, scale=False
        )

        for layer in (
            self.query_dense,
            self.key_dense,
            self.value_dense,
            self.gate_dense,
        ):
            layer.build(input_shape)
        for layer in (self.output_dense, self.group_norm):
            layer.build((*input_shape[:-1], value_dim))
        # pylint:enable=attribute-defined-outside-init

    def _get_retention_args(self, inputs):
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        shape = ops.shape(inputs)
        B = shape[0]
        T = shape[1]
        D_k = inputs.shape[2]
        D_v = D_k * self.value_factor
        H = self.num_heads
        assert D_k % (H * 2) == 0, (D_k, H)
        d_k = D_k // H
        d_v = D_v // H

        k = k / math.sqrt(D_k)

        q = ops.reshape(q, (B, T, H, d_k))
        k = ops.reshape(k, (B, T, H, d_k))

        angle = 1.0 / (10000 ** ops.linspace(0, 1, d_k // 2, dtype=q.dtype))
        angle = ops.reshape(ops.tile(ops.expand_dims(angle, 1), (1, 2)), (-1,))
        s = ops.sin(angle)
        c = ops.cos(angle)
        q = _theta_shift(q, s, c)
        k = _theta_shift(k, s, c)

        gamma = 1 - 2 ** (-5 - ops.arange(self.num_heads, dtype=self.dtype))

        q = ops.reshape(q, (B, T, H, d_k))
        k = ops.reshape(k, (B, T, H, d_k))
        v = ops.reshape(v, (B, T, H, d_v))
        return q, k, v, gamma, B, T, D_v

    def _finalize(self, inputs, output, B, T, D_v):
        output = self.group_norm(output)
        output = ops.reshape(output, (B, T, D_v))
        output = output * self.gate_dense(inputs)
        output = self.output_dense(output)
        return output

    def call(self, inputs):  # pylint:disable=arguments-differ
        q, k, v, gamma, B, T, D_v = self._get_retention_args(inputs)
        output = retention.multi_head_retention(q, k, v, gamma, **self.retention_kwargs)

        return self._finalize(inputs, output, B, T, D_v)

    def compute_output_spec(self, inputs, mask=None):
        return keras.KerasTensor(
            inputs.shape, inputs.dtype, name="multi_scale_retention_output"
        )

    def call_and_create_cache(self, inputs, *, current_index, max_length, mask=None):
        q, k, v, gamma, B, T, D_v = self._get_retention_args(inputs)

        value_out = retention.multi_head_retention(
            q, k, v, gamma, **self.retention_kwargs
        )
        value_out = self._finalize(inputs, value_out, B, T, D_v)
        # there's generally some redundancy between this and multi_head_retention
        # but it's difficult to remove and maintain arbitrary multi_head_retention
        # implementation/kwargs
        cache = retention_ops.create_multi_head_retention_update_cache(
            k, v, gamma, current_index=current_index
        )
        cache = ops.expand_dims(cache, axis=1)
        return value_out, cache

    def call_with_cache(self, inputs, *, cache, current_index, mask=None):
        del current_index, mask
        q, k, v, gamma, B, T, D_v = self._get_retention_args(inputs)
        assert T == 1, T
        q, k, v, cache = (ops.squeeze(x, axis=1) for x in (q, k, v, cache))
        value_out, cache = retention_ops.multi_head_retention_update(
            q, k, v, cache, gamma
        )
        value_out = ops.expand_dims(value_out, axis=1)
        cache = ops.expand_dims(cache, axis=1)
        value_out = self._finalize(inputs, value_out, B, T, D_v)
        return value_out, cache
