# Installation

## Private dependency

This project uses the private `chebdGE` package (`chebttge` and `chebtuckerge`).
Its source is not included in this repository and is not assumed to be on PyPI.
You must already have authorized access to a local checkout. In a Python 3.11
virtual environment, install it first (replace the example path):

```bash
python -m pip install -e /path/to/private/chebdGE
```

## Install the committor solver

Clone `tt-de` alongside this repository, then run from the `committor-tt` root:

```bash
python -m pip install -e '../tt-de[dev,examples]'
python -m pip install -e '.[dev,notebook]'
```

The `ftde` package supplies the density estimators and Ginzburg–Landau model.
Run examples and tests from the repository root. For GPU use, install the JAX
build appropriate for your CUDA environment. Historical checks are recorded in
[archive-validation.txt](docs/archive-validation.txt).
