# Training-pattern attribution: le-wm

- Upstream: <https://github.com/lucas-maes/le-wm>
- Commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- License: MIT (Lucas Maes, 2026), retained verbatim in `LICENSE`.

The project's plain-PyTorch training loop re-expresses the upstream pattern of
jointly optimizing encoder and predictor with predictive and representation
regularization losses. The unused reference `jepa.py` and `module.py` networks
are no longer shipped. No upstream training framework is required at runtime.
