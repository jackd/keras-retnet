import datasets
import tensorflow as tf
import tree
from absl import app, flags
from keras_nlp.models.gpt2.gpt2_causal_lm_preprocessor import GPT2CausalLMPreprocessor
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer

from keras_retnet.backend import keras
from keras_retnet.models.retnet import RetnetBackbone, RetnetCausalLM

flags.DEFINE_integer("batch_size", 2, "Minibatch size.")
flags.DEFINE_integer("epochs", 10, "Number of epochs to train for.")
flags.DEFINE_bool(
    "smoke", False, "Run a smoke test (quick run to make sure things work)"
)
flags.DEFINE_enum(
    "preset",
    default="base",
    enum_values=["base", "medium", "xl", "3b", "7b", "13b", "65b"],
    help="preset suffix",
)
flags.DEFINE_integer("t", 8192, "Sequence length")


def main(_):
    FLAGS = flags.FLAGS
    # Load data.
    data = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    data = data.filter(lambda x: len(x["text"]) > 0 and not x["text"].isspace())
    data_kwargs = {
        "batch_size": FLAGS.batch_size,
        "drop_remainder": True,
        "columns": "text",
        "label_cols": None,
        "prefetch": False,
    }
    train_data = data["train"].to_tf_dataset(shuffle=True, **data_kwargs)
    val_data, test_data = (
        data[key].to_tf_dataset(**data_kwargs) for key in ("validation", "test")
    )

    preset = f"retnet-{FLAGS.preset}"
    # retention_kwargs = {"implementation": "chunkwise_recurrent", "chunk_size": 64}
    retention_kwargs = {"implementation": "chunkwise_recurrent_v2", "chunk_size": 64}
    # retention_kwargs = None
    tokenizer = GPT2Tokenizer.from_preset("gpt2_base_en")
    preprocessor = GPT2CausalLMPreprocessor(tokenizer, sequence_length=FLAGS.t)
    backbone = RetnetBackbone.from_preset(
        preset,
        load_weights=False,
        retention_kwargs=retention_kwargs,
        sequence_length=preprocessor.sequence_length,
        batch_size=FLAGS.batch_size,
    )
    lm = RetnetCausalLM(backbone)

    def map_func(text):
        output = preprocessor(text)
        # ensure static shape info is there - required for jax2tf
        for el in tree.flatten(output):
            el.set_shape((FLAGS.batch_size, *el.shape[1:]))
        return output

    train_data, val_data, test_data = (
        ds.map(map_func).prefetch(tf.data.AUTOTUNE)
        for ds in (train_data, val_data, test_data)
    )

    lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    epochs = FLAGS.epochs
    if FLAGS.smoke:
        train_data = train_data.take(50)
        val_data = val_data.take(50)
        epochs = 2
    lm.fit(train_data, epochs=epochs)


if __name__ == "__main__":
    app.run(main)
