import functools

import tensorflow as tf
from jax.experimental import jax2tf


@functools.cache
def convert_and_compile(jax_func, polymorphic_shapes=None, **kwargs):
    if kwargs:
        jax_func = functools.partial(jax_func, **kwargs)
    return tf.function(
        jax2tf.convert(jax_func, polymorphic_shapes=polymorphic_shapes),
        jit_compile=True,
    )
