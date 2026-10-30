# Verification

Run `python scripts/smoke.py` from an installed editable checkout. It checks
finite forward/backward updates, attached learned-code gradients and stopped
EMA targets for both masked-MLP paths on small CPU inputs.

Long training jobs belong on GPU servers.
