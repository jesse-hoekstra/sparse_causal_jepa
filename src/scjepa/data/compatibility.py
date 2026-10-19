"""Validate observed interfaces before training, without using scientific labels as inputs."""

from collections.abc import Sized
from typing import cast

from torch import Tensor
from torch.utils.data import Dataset

from scjepa.models import SCJEPA, ImageEncoder, VectorEncoder


def validate_model_dataset(model: SCJEPA, dataset: Dataset[dict[str, Tensor]]) -> None:
    """Check shapes, action alignment, dtype, and declared minimum memory/context.

    Latent dimensions remain model choices. Ground-truth states, graph locations,
    parameters, and recovery utilities never enter the model or its objective.
    """
    if len(cast(Sized, dataset)) < 1:
        raise ValueError("dataset must contain at least one episode")
    item = dataset[0]
    if set(item) - {"observations", "actions", "episode_ids"}:
        raise ValueError("training items must not contain ground-truth or metadata keys")
    if "observations" not in item:
        raise ValueError("training items require observations")
    observations = item["observations"]
    encoder = model.online_encoder
    paired = observations.ndim == encoder.observation_ndim + 2
    if observations.ndim != encoder.observation_ndim + 1 and not paired:
        raise ValueError("dataset observation rank is incompatible with the encoder")
    if paired and (not isinstance(encoder, VectorEncoder) or observations.shape[0] != 2):
        raise ValueError("paired observations require two vector episodes")
    if isinstance(encoder, VectorEncoder) and observations.shape[-1] != encoder.observation_dim:
        raise ValueError("dataset observation dimension is incompatible with the vector encoder")
    if isinstance(encoder, ImageEncoder) and observations.shape[1] != encoder.in_channels:
        raise ValueError("dataset channel count is incompatible with the image encoder")
    if observations.dtype != next(encoder.parameters()).dtype:
        raise ValueError("dataset observation dtype must match model dtype before training")
    length = observations.shape[1 if paired else 0]
    if length <= max(model.context_len, model.predictor.history_len):
        raise ValueError("trajectory must contain context/history plus a prediction target")
    actions = item.get("actions")
    if model.predictor.action_dim:
        expected_actions = (length - 1, model.predictor.action_dim)
        if paired:
            expected_actions = (2, *expected_actions)
        if actions is None or actions.shape != expected_actions:
            raise ValueError(
                "dataset actions must match the episode/time axes and model.action_dim"
            )
        if actions.dtype != observations.dtype:
            raise ValueError("dataset action dtype must match observation dtype")
    elif actions is not None:
        raise ValueError("action-free model cannot consume an action-conditioned dataset")
    specification: object = getattr(dataset, "specification", {})
    if isinstance(specification, dict):
        spec = cast(dict[str, object], specification)
        history = spec.get("history_len", 1)
        context = spec.get("minimum_context_len", 1)
        first = spec.get("minimum_prediction_start", 0)
        if isinstance(history, int) and model.predictor.history_len < history:
            raise ValueError("model history is shorter than the DGP's finite memory")
        if isinstance(context, int) and model.context_len < context:
            raise ValueError("model context is shorter than the DGP's parameter-recovery minimum")
        if (
            isinstance(first, int)
            and max(model.context_len, model.predictor.history_len) - 1 < first
        ):
            raise ValueError("model would supervise an invalid initialization transition")
