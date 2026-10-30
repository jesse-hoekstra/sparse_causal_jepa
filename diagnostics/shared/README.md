# Shared diagnostic utilities

Data generation, cache authentication, initialization, sampling and evaluation
helpers used by the masked-MLP runners.

`prepare.py` generates a fresh cache. Its `--help` describes the available options.
Keep the source checkout fixed while jobs are running so checkpoint and cache
identities remain valid.
