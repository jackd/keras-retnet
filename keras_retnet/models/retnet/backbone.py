import copy
import typing as tp

from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.models.backbone import Backbone
from keras_nlp.utils.python_utils import classproperty

from ...backend import keras
from ...layers.decoder import RetnetDecoder
from .presets import backbone_presets


@keras.utils.register_keras_serializable("keras_retnet")
class RetnetBackbone(Backbone):  # pylint:disable=abstract-method
    def __init__(
        self,
        vocabulary_size: int,
        hidden_dim: int,
        num_layers: int,
        intermediate_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        value_factor: int = 2,
        layer_norm_epsilon: float = 1e-5,
        sequence_length: tp.Optional[int] = None,
        batch_size: tp.Optional[int] = None,
        activation="gelu",
        retention_kwargs=None,
        **kwargs,
    ):
        # Inputs
        token_ids = keras.Input(
            shape=(sequence_length,),
            batch_size=batch_size,
            dtype="int32",
            name="token_ids",
        )
        padding_mask = keras.Input(
            shape=(sequence_length,), dtype="int32", name="padding_mask"
        )

        # Embed tokens, positions.
        token_embedding_layer = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            name="token_embedding",
            embeddings_initializer=keras.initializers.RandomUniform(-1e-4, 1e-4),
            tie_weights=False,
        )
        x = token_embedding_layer(token_ids)

        for i in range(num_layers):
            x = RetnetDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                value_factor=value_factor,
                activation=activation,
                retention_kwargs=retention_kwargs,
                name=f"decoder{i}",
            )(x)
        sequence_output = x

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
            **kwargs,
        )

        # All references to `self` below this line
        self.token_embedding = token_embedding_layer
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.retention_kwargs = (
            {} if retention_kwargs is None else copy.deepcopy(retention_kwargs)
        )
        self.activation = keras.activations.get(activation)

    def get_config(self):
        config = super().get_config()
        config.update(
            vocabulary_size=self.vocabulary_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_heads=self.num_heads,
            layer_norm_epsilon=self.layer_norm_epsilon,
            retention_kwargs=copy.deepcopy(self.retention_kwargs),
            activation=keras.activations.serialize(self.activation),
        )
        return config

    @classproperty
    def presets(cls):  # pylint:disable=no-self-argument
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate RwkvBackbone model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = keras_retnet.models.retnet.RetnetBackbone.from_preset(
            "{{example_preset_name}}"
        )

        # Load randomly initialized model from preset architecture
        model = keras_retnet.models.retnet.RetnetBackbone.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """

        if not cls.presets:
            raise NotImplementedError("No presets have been created for this class.")

        if preset not in cls.presets:  # pylint:disable=unsupported-membership-test
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]  # pylint:disable=unsubscriptable-object
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        raise NotImplementedError("No weights available")

        # pylint:disable=line-too-long

        # import safetensors  # pylint:disable=import-outside-toplevel

        # weights_path = keras.utils.get_file(
        #     "retnet-1m.safetensors",
        #     metadata["weights_url"],
        #     cache_subdir=os.path.join("models", preset),
        #     file_hash=metadata["weights_hash"],
        # )

        # def load_layer_norm(layer: keras.layers.LayerNormalization, key):
        #     layer.gamma.assign(fp.get_tensor(f"{key}.weight"))
        #     layer.beta.assign(fp.get_tensor(f"{key}.bias"))

        # def load_dense(layer: keras.layers.Dense, key: str):
        #     layer.kernel.assign(fp.get_tensor(f"{key}.weight").T)
        #     layer.bias.assign(fp.get_tensor(f"{key}.bias"))

        # with safetensors.safe_open(
        #     weights_path, framework=keras.backend.backend()
        # ) as fp:
        #     embedding_layer: ReversibleEmbedding = model.token_embedding
        #     embedding_layer.embeddings.assign(
        #         fp.get_tensor("decoder.embed_tokens.weight")
        #     )
        #     embedding_layer.reverse_embeddings.assign(
        #         fp.get_tensor("decoder.output_projection.weight").T
        #     )
        #     for i in range(model.num_layers):
        #         layer: RetnetDecoder = model.get_layer(f"decoder{i}")
        #         load_layer_norm(
        #             layer.retention_layer_norm,
        #             f"decoder.layers.{i}.retention_layer_norm",
        #         )
        #         load_layer_norm(
        #             layer.final_layer_norm, f"decoder.layers.{i}.final_layer_norm"
        #         )
        #         load_dense(
        #             layer.feedforward_intermediate_dense, f"decoder.layers.{i}.ffn.fc1"
        #         )
        #         load_dense(
        #             layer.feedforward_output_dense, f"decoder.layers.{i}.ffn.fc2"
        #         )
        #         retention = layer.retention
        #         load_dense(retention.gate_dense, f"decoder.layers.{i}.retention.g_proj")
        #         load_dense(retention.key_dense, f"decoder.layers.{i}.retention.k_proj")
        #         load_dense(
        #             retention.query_dense, f"decoder.layers.{i}.retention.q_proj"
        #         )
        #         load_dense(
        #             retention.value_dense, f"decoder.layers.{i}.retention.v_proj"
        #         )
        #         load_dense(
        #             retention.output_dense, f"decoder.layers.{i}.retention.out_proj"
        #         )
        # return model

        # pylint:enable=line-too-long
