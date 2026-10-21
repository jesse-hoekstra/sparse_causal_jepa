"""Measure frozen simplified-Gaussian context-code variance without training.

Load an experiment's own trusted checkpoint and authoritative observation map.
Sample each longest context once, then compare nested prefixes no longer than
the trained context. This diagnostic never averages codes before measuring them.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from torch import Tensor

from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.training.factory import build_model


def load_frozen_model(
    checkpoint: Path, device: torch.device
) -> tuple[SCJEPA, VectorObservationMap, dict[str, object]]:
    """Validate recorded model/run/data/map identities and freeze checkpoint weights."""
    checkpoint = checkpoint.resolve()
    checkpoint_bytes = checkpoint.read_bytes()
    payload = cast(
        dict[str, Any],
        torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False),
    )
    record = read_json(checkpoint.parent / "run_config.json")
    root = checkpoint.parents[3]
    manifest = read_json(root / "experiment.json")
    body = {key: value for key, value in manifest.items() if key != "identity"}
    data = cast(dict[str, object], record["data"])
    model_config = cast(dict[str, object], payload["model_config"])
    if data.get("family") != "simplified_gaussian":
        raise ValueError("only simplified_gaussian checkpoints are supported")
    if (
        payload.get("format_version") != 1
        or canonical_fingerprint(model_config) != canonical_fingerprint(record["model"])
        or payload["dataset_identity"] != record["dataset_identity"]
        or payload["run_metadata"].get("experiment_run_identity") != record["run_identity"]
        or record["experiment_identity"] != manifest["identity"]
        or canonical_fingerprint(body) != manifest["identity"]
        or canonical_fingerprint(manifest["definition"]) != manifest["config_fingerprint"]
        or canonical_fingerprint(
            [manifest["identity"], record["graph"], record["seed"], record["condition"]]
        )
        != record["run_identity"]
    ):
        raise ValueError("checkpoint/run/experiment provenance differs")
    map_path = root / "observation_map.pt"
    map_payload = cast(
        dict[str, Tensor], torch.load(map_path, map_location="cpu", weights_only=True)
    )
    observation_map = VectorObservationMap(matrix=map_payload["matrix"], dimension=dgp.DIMENSION)
    dataset = build_dgp_dataset(OmegaConf.create(data), observation_matrix=observation_map.matrix)
    if (
        observation_map.fingerprint != manifest["observation_fingerprint"]
        or dataset.fingerprint != record["dataset_identity"]
    ):
        raise ValueError("observation map or dataset generator differs from the recorded run")
    model = build_model(OmegaConf.create(model_config))
    model.load_state_dict(payload["model"])
    if model.context_len != model_config["context_len"]:
        raise ValueError("checkpoint context length differs from its model configuration")
    model.to(device).eval().requires_grad_(False)
    return (
        model,
        observation_map,
        {
            "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "step": payload["step"],
            "trained_context_length": model.context_len,
            "run_identity": record["run_identity"],
            "dataset_identity": dataset.fingerprint,
            "recorded_source_fingerprint": record["source_fingerprint"],
            "observation_map": str(map_path),
            "observation_map_fingerprint": observation_map.fingerprint,
            "generator_version": dgp.GENERATOR_VERSION,
        },
    )


def sample_contexts(
    observation_map: VectorObservationMap, groups: int, contexts: int, length: int, seed: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw exact scientific laws locally, returning theta and flattened contexts."""
    if min(groups, contexts, length) < 2:
        raise ValueError("need at least two groups, contexts per theta, and context frames")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    theta = dgp.PARAMETER_BOUND * (
        2 * torch.rand(groups, dgp.DIMENSION, dtype=torch.float64, generator=generator) - 1
    )
    initial = torch.randn(groups, contexts, dgp.DIMENSION, dtype=torch.float64, generator=generator)
    gaussian = torch.randn(
        groups, contexts, length - 1, 10, dtype=torch.float64, generator=generator
    )
    coupling = (
        2
        * torch.rand(
            groups, contexts, length - 1, dgp.DIMENSION, dtype=torch.float64, generator=generator
        )
        - 1
    )
    actions = dgp.encode_controls(torch.cat((gaussian, coupling), dim=-1))
    states = dgp.simulate(initial, theta[:, None], actions)
    return (
        theta,
        observation_map.forward(states).float().flatten(0, 1),
        actions.float().flatten(0, 1),
    )


@torch.inference_mode()
def encode_contexts(
    model: SCJEPA, observations: Tensor, actions: Tensor, lengths: list[int], batch_size: int
) -> dict[int, Tensor]:
    """Infer each prefix directly with frozen online states and contextual pooling."""
    if batch_size < 1 or not lengths or any(not 2 <= k <= model.context_len for k in lengths):
        raise ValueError("batch size must be positive; context lengths must lie in [2, trained K]")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model must be in evaluation mode with all parameters frozen")
    if max(lengths) > observations.shape[1] or actions.shape[1] < max(lengths) - 1:
        raise ValueError("sampled trajectories are shorter than requested contexts")
    reference = next(model.parameters())
    codes: dict[int, Tensor] = {}
    for length in lengths:
        pieces: list[Tensor] = []
        for start in range(0, observations.shape[0], batch_size):
            latent = model.online_encoder.encode_sequence(
                observations[start : start + batch_size, :length].to(reference)
            )
            pieces.append(
                model.parameter_encoder(
                    latent, actions[start : start + batch_size, : length - 1].to(reference)
                )
                .cpu()
                .double()
            )
        codes[length] = torch.cat(pieces)
    return codes


def variance_components(codes: Tensor) -> dict[str, float | None]:
    """Decompose [theta, context, coordinate] variance, retaining negative estimates."""
    if codes.ndim != 3 or min(codes.shape[:2]) < 2 or not torch.isfinite(codes).all():
        raise ValueError(
            "codes require finite [at least 2 groups, at least 2 contexts, coordinates]"
        )
    codes = codes.double()
    within = float(codes.var(dim=1, correction=1).mean())
    group_means = float(codes.mean(dim=1).var(dim=0, correction=1).mean())
    between = group_means - within / codes.shape[1]
    total = within + between
    return {
        "within": within,
        "group_mean_variance": group_means,
        "between": between,
        "total": total,
        "ratio": within / total if total > 0 else None,
    }


def main() -> None:
    """Run a read-only checkpoint diagnostic and save raw codes plus provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-lengths", type=int, nargs="+")
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--contexts-per-theta", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.groups, args.contexts_per_theta) < 2 or min(args.batch_size, args.threads) < 1:
        parser.error("need >=2 groups and contexts; batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    model, observation_map, provenance = load_frozen_model(
        args.checkpoint, torch.device(args.device)
    )
    lengths = sorted(set(cast(list[int], args.context_lengths or [model.context_len])))
    if any(not 2 <= length <= model.context_len for length in lengths):
        parser.error("context lengths must lie in [2, checkpoint trained context length]")
    theta, observations, actions = sample_contexts(
        observation_map, args.groups, args.contexts_per_theta, max(lengths), args.seed
    )
    flat_codes = encode_contexts(model, observations, actions, lengths, args.batch_size)
    codes = {
        length: value.reshape(args.groups, args.contexts_per_theta, dgp.DIMENSION)
        for length, value in flat_codes.items()
    }
    result = provenance | {
        "groups": args.groups,
        "contexts_per_theta": args.contexts_per_theta,
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "torch_version": str(torch.__version__),
        "rows": [{"K": length, **variance_components(code)} for length, code in codes.items()],
        "interpretation": (
            "Fresh independent scientific draws, nested prefixes of one max-K trajectory per "
            "context. Between variance subtracts within/context-count from variance of group "
            "means; negative estimates are retained. Ratio can exceed one and is not a fraction "
            "of parameter information; null means zero total variance. Shorter contexts were "
            "not used in training. K=2 is below the theoretical recovery minimum. All lengths "
            "use trained positional rows; no weights or positional embeddings are changed. "
            "Recorded model/data/map identities are validated, not historical source-byte equality."
        ),
    }
    output = cast(Path, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    torch.save({"codes": codes, "theta": theta, "seed": args.seed}, output / "codes.pt")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
