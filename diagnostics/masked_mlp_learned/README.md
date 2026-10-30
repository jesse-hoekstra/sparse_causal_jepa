# Learned parameter codes

Masked-MLP training with a context-derived parameter code.

- `run.py`: training, projection and calibration.
- `acquisition.py`, `selection.py`: initialization and training-loss selection.
- `bound_control.py`, `budget_control.py`: explicit calibration and budget settings.
- `physical_response.py`: evaluation utilities.
- `acquisition_batch.sh`, `experiment_batch.sh`: server batch orchestration.

Use each Python entry point's `--help` for command-line options.
