# Keras RetNet

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[keras](https://keras.io/) implementation of [retention networks](https://arxiv.org/abs/2307.08621).

This has no affiliation with either keras or the authors of the original work.

Note there are currently no publicly available weights to load. If you are aware of any, please let us know in the issues. While considerable effort has gone into ensuring consistency with the original, it's difficult to say if this is the case or not without integration tests with pretrained weights.

## Installation

While this repository supports all three keras backends (tensorflow, pytorch and jax), the former two are supported via `jax2torch` and `jax2tf` - so all need `jax` (most of these functions could be re-written in their native backends, but that's not a high priority right now). Data processing in the examples is done via tensorflow. Once you've install `jax`, `keras-core` and `keras-nlp`, this package can be installed via

```bash
git clone https://github.com/jackd/keras-retnet.git
pip install -e keras-retnet
```

Note getting all backends to work in the same environment is non-trivial. I had success using `conda` to install `jax` and pip for `tensorflow`/`torch` (following conda installation instructions for `tensorflow`/`torch` tends to break `jax` installations).

## Quickstart

See [examples/train.py](./examples/train.py) and [examples/generate.py](./examples/generate.py). Things _might_ work without specifying a keras backend - in which case behaviour reverts to `tf.keras` - this hasn't been tested. `jax` is probably the most stable backend, since it doesn't use `jax2torch` or `jax2tf`.

```bash
KERAS_BACKEND=jax python examples/train.py --smoke
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
