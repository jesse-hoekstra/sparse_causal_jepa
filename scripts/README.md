# Verification

Run `python scripts/smoke.py` from an installed editable checkout. On small CPU inputs it
checks finite forward/backward updates, attached learned-code gradients through both prediction
steps and stopped EMA targets for the two Experiment 1 masked-MLP paths and the vision model.

The test suite (`python -m pytest`) also runs every vision training stage for a few CPU updates
and checks exact resumption and continuation. Long training jobs belong on GPU servers.
