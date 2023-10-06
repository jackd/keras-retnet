import functools
import math
import os

if os.environ.get("KERAS_BACKEND", None) is None:
    os.environ["KERAS_BACKEND"] = "jax"
    # os.environ["KERAS_BACKEND"] = "torch"
import jax
import keras_core as keras
import triton
from absl import app, flags

from keras_retnet.ops.retention import multi_head_retention

flags.DEFINE_integer("b", 1, "batch size")
flags.DEFINE_integer("t", 2048, "sequence length")
flags.DEFINE_integer("c", 512, "key channels")

FLAGS = flags.FLAGS

line_vals, line_names, styles = (
    list(x)
    for x in zip(
        ["block", "block", ("red", "-")],
        ["chunkwise_recurrent-32", "chunkwise_recurrent-32", ("orange", "-")],
        ["chunkwise_recurrent-64", "chunkwise_recurrent-64", ("orange", "dashed")],
        ["chunkwise_recurrent-128", "chunkwise_recurrent-128", ("orange", "dotted")],
        ["chunkwise_recurrent_v2-32", "chunkwise_recurrent_v2-32", ("purple", "-")],
        [
            "chunkwise_recurrent_v2-64",
            "chunkwise_recurrent_v2-64",
            ("purple", "dashed"),
        ],
        [
            "chunkwise_recurrent_v2-128",
            "chunkwise_recurrent_v2-128",
            ("purple", "dotted"),
        ],
        ["parallel_scan", "parallel_scan", ("blue", "-")],
        ["serial_scan", "serial_scan", ("cyan", "-")],
        ["scan_v2", "scan_v2", ("black", "-")],
    )
)


@functools.cache
def get_fn(implementation):
    kwargs = {"implementation": implementation}
    if implementation.startswith("chunkwise_recurrent"):
        implementation, chunk_size = implementation.split("-")
        kwargs.update(implementation=implementation, chunk_size=int(chunk_size))
    elif implementation.endswith("scan"):
        kwargs["implementation"] = "scan"
        kwargs["parallel"] = implementation.startswith("parallel")
    return jax.jit(functools.partial(multi_head_retention, **kwargs))


def main(_):
    b = FLAGS.b
    t = FLAGS.t
    c = FLAGS.c
    log2_c = int(math.log2(c))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_log=True,  # x axis is logarithmic.
            line_arg="implementation",
            line_vals=line_vals,  # Possible values for `line_arg`.
            line_names=line_names,  # Label name for the lines.
            styles=styles,  # Line styles.
            ylabel="time (ms)",  # Label name for the y-axis.
            plot_name="Multi-scale Retention",
            y_log=True,
            args={},
            x_names=["num_heads"],
            x_vals=[int(2**i) for i in range(log2_c)],
        )
    )
    def benchmark(implementation, num_heads):
        h = num_heads

        q = keras.random.normal((b, t, h, c // h))
        k = keras.random.normal((b, t, h, c // h))
        v = keras.random.normal((b, t, h, c // h * 2))
        gamma = keras.random.normal((h,))

        quantiles = [0.5, 0.2, 0.8]

        func = get_fn(implementation)

        def wrapper():
            output = func(q, k, v, gamma)
            jax.block_until_ready(output)
            return output

        ms, min_ms, max_ms = triton.testing.do_bench(
            wrapper,
            quantiles=quantiles,
        )

        return ms, min_ms, max_ms

    benchmark.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    app.run(main)
