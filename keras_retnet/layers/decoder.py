import copy

from keras_nlp.utils.keras_utils import clone_initializer

from ..backend import keras
from . import multi_scale_retention


@keras.utils.register_keras_serializable("keras_retnet")
class RetnetDecoder(keras.layers.Layer):  # pylint:disable=abstract-method
    def __init__(
        self,
        intermediate_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        value_factor: int = 2,
        is_moe_layer: bool = False,
        layer_norm_epsilon: float = 1e-05,
        normalize_first: bool = False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        activation="gelu",
        retention_kwargs=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if is_moe_layer:
            raise NotImplementedError()
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.value_factor = value_factor
        self.is_moe_layer = is_moe_layer
        self.layer_norm_epsilon = layer_norm_epsilon
        self.normalize_first = normalize_first
        self.retention_kwargs = (
            {} if retention_kwargs is None else copy.deepcopy(retention_kwargs)
        )
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.activation = keras.activations.get(activation)

    def get_config(self):
        config = super().get_config()
        config.update(
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            value_factor=self.value_factor,
            is_moe_layer=self.is_moe_layer,
            layer_norm_epsilon=self.layer_norm_epsilon,
            normalize_first=self.normalize_first,
            retention_kwargs=copy.deepcopy(self.retention_kwargs),
            kernel_initializer=keras.initializers.serialize(self.kernel_initializer),
            bias_initializer=keras.initializers.serialize(self.bias_initializer),
            activation=keras.activations.serialize(self.activation),
        )
        return config

    def build(self, input_shape):
        # pylint:disable=attribute-defined-outside-init
        if self.built:
            return
        super().build(input_shape)

        self.retention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="retention_layer_norm"
        )
        self.retention = multi_scale_retention.MultiScaleRetention(
            self.num_heads,
            self.value_factor,
            retention_kwargs=copy.deepcopy(self.retention_kwargs),
        )
        self.dropout_layer = keras.layers.Dropout(self.dropout)
        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="final_layer_norm"
        )
        # Feedforward layers.
        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="intermediate_dense",
        )
        self.feedforward_output_dense = keras.layers.Dense(
            input_shape[-1],
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="output_dense",
        )

        for layer in (
            self.retention_layer_norm,
            self.retention,
            self.dropout_layer,
            self.final_layer_norm,
            self.feedforward_intermediate_dense,
        ):
            layer.build(input_shape)
        self.feedforward_output_dense.build(
            self.feedforward_intermediate_dense.compute_output_shape(input_shape)
        )
        # pylint:enable=attribute-defined-outside-init

    def call(self, x):  # pylint:disable=arguments-differ
        residual = x
        if self.normalize_first:
            x = self.retention_layer_norm(x)

        x = self.retention(x)
        x = self.dropout_layer(x)

        x = x + residual
        if not self.normalize_first:
            x = self.retention_layer_norm(x)

        residual = x
        if self.normalize_first:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.feedforward_intermediate_dense(x)
            x = self.feedforward_output_dense(x)
        else:
            raise NotImplementedError()

        x = x + residual
        if not self.normalize_first:
            x = self.final_layer_norm(x)

        return x
