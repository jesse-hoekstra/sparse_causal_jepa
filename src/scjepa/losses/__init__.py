"""Experiment-independent latent prediction and optional representation penalties."""

from scjepa.losses.predictive import prediction_mse
from scjepa.losses.regularization import RegularizerKind, RepresentationRegularizer

__all__ = ["RegularizerKind", "RepresentationRegularizer", "prediction_mse"]
