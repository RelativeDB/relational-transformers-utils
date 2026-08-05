# Installation

Relational Transformers Utils requires Python 3.10 or newer. It depends on
`relational-transformers`, numpy, torch, and safetensors, and installs two importable
packages: `relational_transformers_utils` and `relben`.

```{eval-rst}
.. tab:: pip

   ::

      pip install -U relational-transformers-utils

.. tab:: uv

   ::

      uv add relational-transformers-utils
```

## Editable Install

For development, install the checkout with the test dependencies and run the suite:

```bash
git clone https://github.com/RelativeDB/relational-transformers-utils
cd relational-transformers-utils
python -m pip install -e '.[dev]'
pytest
```

The suite is deterministic and runs offline on CPU.
