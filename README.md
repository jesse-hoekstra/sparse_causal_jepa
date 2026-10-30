# Sparse causal JEPA

Research code for sparse causal representation learning.

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,analysis,logging]'
.venv/bin/python scripts/smoke.py
```

Implementation is in `src/scjepa` and `diagnostics`; checks are in `tests`.

See [third-party notices](src/scjepa/third_party/README.md) for dependency
licenses and attribution, including SIGReg's non-commercial restrictions.
