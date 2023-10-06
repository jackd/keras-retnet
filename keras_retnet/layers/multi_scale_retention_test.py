import unittest

import numpy as np
import tree

from keras_retnet.backend import ops
from keras_retnet.layers.multi_scale_retention import MultiScaleRetention


class MultiScaleRetentionTest(unittest.TestCase):
    def test_iterative_cache(
        self,
        seed: int = 0,
        hidden_dim: int = 16,
        max_length: int = 11,
        current_index: int = 5,
        batch_size: int = 2,
        num_heads: int = 4,
    ):
        rng = np.random.default_rng(seed)
        layer = MultiScaleRetention(num_heads)
        layer.build((None, max_length, hidden_dim))

        # ensure reproducible - no way to set keras seed?
        for v in layer.trainable_weights:
            v.assign(rng.normal(size=v.shape).astype(dtype=v.dtype))

        inputs = rng.normal(size=(batch_size, max_length, hidden_dim)).astype("float32")

        outputs, cache = layer.call_and_create_cache(
            inputs, current_index=current_index, max_length=max_length
        )
        # compare to raw call
        raw_outputs = layer(inputs)
        np.testing.assert_allclose(outputs, raw_outputs, rtol=1e-4)

        # iterative implementation
        prev_output, prev_cache = layer.call_and_create_cache(
            inputs, current_index=current_index - 1, max_length=max_length
        )
        np.testing.assert_allclose(
            prev_output[:, current_index - 1], raw_outputs[:, current_index - 1]
        )

        it_outputs, it_cache = layer.call_with_cache(
            ops.expand_dims(inputs[:, current_index], 1),
            cache=prev_cache,
            current_index=current_index,
        )
        np.testing.assert_allclose(
            it_outputs, ops.expand_dims(raw_outputs[:, current_index], 1), rtol=1e-4
        )
        np.testing.assert_allclose(
            it_outputs, ops.expand_dims(outputs[:, current_index], 1), rtol=1e-4
        )
        tree.assert_same_structure(it_cache, cache)
        for itc, c in zip(tree.flatten(it_cache), tree.flatten(cache)):
            np.testing.assert_allclose(itc, c, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
