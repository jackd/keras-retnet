from keras_nlp.backend import keras, ops

if keras.backend.backend() == "tensorflow":
    import tensorflow as tf

    from .tensorflow import ema, retention

    Array = tf.Tensor
elif keras.backend.backend() == "jax":
    import jax.numpy as jnp

    from .jax import ema, retention

    Array = jnp.ndarray
elif keras.backend.backend() == "torch":
    import torch

    Array = torch.Tensor  # must be before next .torch import for some reason...
    from .torch import ema, retention


else:
    raise ValueError(f"Unsupported keras backend {keras.backend.backend()}")


__all__ = ["Array", "ema", "keras", "ops", "retention"]
