import functools
import os

if os.environ.get("KERAS_BACKEND", None) is None:
    os.environ["KERAS_BACKEND"] = "jax"
    # os.environ["KERAS_BACKEND"] = "torch"

import keras_core as keras
import triton
from absl import app, flags

from keras_retnet.models.retnet.backbone import RetnetBackbone

flags.DEFINE_integer("b", 1, "batch size")
flags.DEFINE_integer("t", 256, "sequence length")

FLAGS = flags.FLAGS

kwargs = {
    "x_names": ["num_heads"],
    "x_vals": [2**i for i in range(8)],
}
line_vals, line_names, styles = (
    list(x)
    for x in zip(
        ["block", "block", ("red", "-")],
        ["chunkwise_recurrent", "chunkwise_recurrent", ("orange", "-")],
        ["parallel_scan", "parallel_scan", ("blue", "-")],
        ["serial_scan", "serial_scan", ("cyan", "-")],
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_log=True,  # x axis is logarithmic.
        line_arg="implementation",
        line_vals=line_vals,  # Possible values for `line_arg`.
        line_names=line_names,  # Label name for the lines.
        styles=styles,  # Line styles.
        ylabel="time (ms)",  # Label name for the y-axis.
        plot_name="RetNet inference",  # Name for the plot. Used also as a file name.
        y_log=True,
        args={},
        **kwargs,
    )
)
def benchmark(implementation, num_heads):
    batch_size = FLAGS.b
    sequence_length = FLAGS.t

    config = RetnetBackbone.presets["base"][  # pylint:disable=unsubscriptable-object
        "config"
    ]
    config.update(
        num_heads=num_heads, implementation=implementation, vocabulary_size=50277
    )

    model = RetnetBackbone(**config)
    token_ids = keras.ops.cast(
        keras.random.uniform(
            (batch_size, sequence_length), maxval=model.vocabulary_size
        ),
        "int32",
    )
    padding_mask = keras.ops.ones((batch_size, sequence_length), dtype=bool)
    inp = {"token_ids": token_ids, "padding_mask": padding_mask}
    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(
        functools.partial(model, inp),
        quantiles=quantiles,
    )

    return ms, min_ms, max_ms


def main(_):
    benchmark.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    app.run(main)
