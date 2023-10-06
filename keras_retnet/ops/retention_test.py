import unittest

import numpy as np
from absl.testing import parameterized
from keras_core.src.testing import TestCase

from keras_retnet.backend import ops
from keras_retnet.ops import retention

IMPLEMENTATIONS = ("block", "chunkwise_recurrent", "parallel_scan", "serial_scan")


def normalize(x, epsilon: float = 1e-5):
    x = x - ops.mean(x, axis=-1, keepdims=True)
    x = x / (ops.std(x, axis=-1, keepdims=True) + epsilon)
    return x


class RetentionCorrectnessTest(TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {"implementation": "scan", "parallel": True},
        {"implementation": "scan", "parallel": False},
        {"implementation": "scan_v2"},
        {
            "implementation": "chunkwise_recurrent",
            "chunk_size": 2,
            "T": 8,
        },
        {
            "implementation": "chunkwise_recurrent_v2",
            "chunk_size": 2,
            "T": 8,
            "parallel": False,
        },
        {
            "implementation": "chunkwise_recurrent_v2",
            "chunk_size": 2,
            "T": 8,
            "parallel": True,
        },
    )
    def test_retention_methods_consistent(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        d_k: int = 3,
        d_v: int = 5,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)
        query = ops.convert_to_tensor(rng.normal(size=(B, T, d_k)), "float32")
        keys = ops.convert_to_tensor(rng.normal(size=(B, T, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=()), "float32")
        block = retention.retention(query, keys, values, gamma, implementation="block")
        other = retention.retention(query, keys, values, gamma, **kwargs)
        block = normalize(block)
        other = normalize(other)
        self.assertAllClose(block, other, rtol=1e-2)

    @parameterized.parameters(
        {"implementation": "scan", "parallel": True},
        {"implementation": "scan", "parallel": False},
        {"implementation": "scan_v2"},
        {
            "implementation": "chunkwise_recurrent",
            "chunk_size": 2,
            "T": 8,
        },
        {
            "implementation": "chunkwise_recurrent_v2",
            "chunk_size": 2,
            "T": 8,
            "parallel": False,
        },
        {
            "implementation": "chunkwise_recurrent_v2",
            "chunk_size": 2,
            "T": 8,
            "parallel": True,
        },
    )
    def test_multi_head_retention_methods_consistent(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        H: int = 4,
        d_k: int = 3,
        d_v: int = 5,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)
        query = ops.convert_to_tensor(rng.normal(size=(B, T, H, d_k)), "float32")
        keys = ops.convert_to_tensor(rng.normal(size=(B, T, H, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T, H, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=(H,)), "float32")

        block = retention.multi_head_retention(
            query, keys, values, gamma, implementation="block"
        )
        other = retention.multi_head_retention(query, keys, values, gamma, **kwargs)
        block = normalize(block)
        other = normalize(other)
        self.assertAllClose(block, other, rtol=1e-2, atol=1e-3)

    def test_single_head_retention_consistent(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        d_k: int = 3,
        d_v: int = 5,
    ):
        rng = np.random.default_rng(seed)
        query = ops.convert_to_tensor(rng.normal(size=(B, T, 1, d_k)), "float32")
        keys = ops.convert_to_tensor(rng.normal(size=(B, T, 1, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T, 1, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=(1,)), "float32")

        multi = retention.multi_head_retention(query, keys, values, gamma)
        multi = ops.squeeze(multi, 2)
        single = retention.retention(
            ops.squeeze(query, 2),
            ops.squeeze(keys, 2),
            ops.squeeze(values, 2),
            ops.squeeze(gamma, 0),
        )
        self.assertAllClose(multi, single)

    def test_cache_creation_and_update_without_index(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        d_k: int = 3,
        d_v: int = 5,
    ):
        rng = np.random.default_rng(seed)
        query = ops.convert_to_tensor(rng.normal(size=(B, T + 1, d_k)), "float32")
        keys = ops.convert_to_tensor(rng.normal(size=(B, T + 1, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T + 1, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=()), "float32")

        expected = retention.retention(
            query, keys, values, gamma, implementation="block", rescale=False
        )
        expected = expected[:, -1]

        cache = retention.create_retention_update_cache(
            keys[:, :-1], values[:, :-1], gamma
        )
        actual, actual_updated_cache = retention.retention_update(
            query[:, -1], keys[:, -1], values[:, -1], cache, gamma
        )
        self.assertAllClose(actual, expected)
        expected_updated_cache = retention.create_retention_update_cache(
            keys, values, gamma
        )
        self.assertAllClose(actual_updated_cache, expected_updated_cache)

        cache_with_index = retention.create_retention_update_cache(
            keys[:, :-1], values[:, :-1], gamma, current_index=T - 1
        )
        self.assertAllClose(cache_with_index, cache)

    def test_cache_creation_and_update_with_index(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        d_k: int = 3,
        d_v: int = 5,
        current_index: int = 3,
    ):
        rng = np.random.default_rng(seed)
        keys = ops.convert_to_tensor(rng.normal(size=(B, T, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=()), "float32")

        actual_cache = retention.create_retention_update_cache(
            keys, values, gamma, current_index=current_index
        )
        expected_cache = retention.create_retention_update_cache(
            keys[:, : current_index + 1], values[:, : current_index + 1], gamma
        )
        self.assertAllClose(actual_cache, expected_cache)

    def test_multi_head_cache_creation_and_update_without_index(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        d_k: int = 3,
        d_v: int = 5,
        H: int = 2,
    ):
        rng = np.random.default_rng(seed)
        query = ops.convert_to_tensor(rng.normal(size=(B, T + 1, H, d_k)), "float32")
        keys = ops.convert_to_tensor(rng.normal(size=(B, T + 1, H, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T + 1, H, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=(H,)), "float32")

        expected = retention.multi_head_retention(
            query, keys, values, gamma, implementation="block", rescale=False
        )
        expected = expected[:, -1]

        cache = retention.create_multi_head_retention_update_cache(
            keys[:, :-1], values[:, :-1], gamma
        )
        actual, actual_updated_cache = retention.multi_head_retention_update(
            query[:, -1], keys[:, -1], values[:, -1], cache, gamma
        )
        expected_updated_cache = retention.create_multi_head_retention_update_cache(
            keys, values, gamma
        )
        self.assertAllClose(actual, expected)
        self.assertAllClose(actual_updated_cache, expected_updated_cache)

    def test_multi_head_cache_creation_and_update_with_index(
        self,
        seed: int = 0,
        B: int = 2,
        T: int = 7,
        d_k: int = 3,
        d_v: int = 5,
        H: int = 2,
        current_index: int = 3,
    ):
        rng = np.random.default_rng(seed)
        keys = ops.convert_to_tensor(rng.normal(size=(B, T, H, d_k)), "float32")
        values = ops.convert_to_tensor(rng.normal(size=(B, T, H, d_v)), "float32")
        gamma = ops.convert_to_tensor(rng.uniform(size=(H,)), "float32")

        actual_cache = retention.create_multi_head_retention_update_cache(
            keys, values, gamma, current_index=current_index
        )
        expected_cache = retention.create_multi_head_retention_update_cache(
            keys[:, : current_index + 1], values[:, : current_index + 1], gamma
        )
        self.assertAllClose(actual_cache, expected_cache)

        query = ops.convert_to_tensor(rng.normal(size=(B, T, H, d_k)), "float32")
        updated, updated_cache = retention.multi_head_retention_update(
            query[:, current_index + 1],
            keys[:, current_index + 1],
            values[:, current_index + 1],
            actual_cache,
            gamma,
        )
        block_values = retention.multi_head_retention(
            query, keys, values, gamma, implementation="block", rescale=False
        )
        self.assertAllClose(updated, block_values[:, current_index + 1])


if __name__ == "__main__":
    unittest.main()
