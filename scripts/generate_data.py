"""Inspect deterministic DGP episodes or stream a small versioned dataset bundle."""

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from scjepa.data.bundle import load_dataset_bundle, save_dataset_bundle
from scjepa.data.controlled_vector import ParameterGraph, sufficient_influence_diagnostics
from scjepa.data.factory import build_dgp_dataset


def main() -> None:
    """Report shapes/provenance by default; write only when --output is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="DGP YAML or a supplied configs/data preset")
    parser.add_argument(
        "--bundle", type=Path, help="Inspect an existing bundle instead of generating"
    )
    parser.add_argument(
        "--output", type=Path, help="New bundle directory; existing paths are refused"
    )
    parser.add_argument("--diagnostics", action="store_true", help="Small vector influence checks")
    parser.add_argument("overrides", nargs="*", help="DGP dotlist overrides, e.g. num_episodes=2")
    args = parser.parse_args()
    overrides = cast(list[str], args.overrides)
    if args.bundle is not None:
        if args.output is not None or args.config is not None or overrides:
            parser.error("--bundle is read-only and cannot combine with generation settings")
        dataset = load_dataset_bundle(cast(Path, args.bundle))
    else:
        if args.config is None:
            config = OmegaConf.create({})
        else:
            loaded = OmegaConf.load(cast(Path, args.config))
            config = loaded.data if "data" in loaded else loaded
        if not isinstance(config, DictConfig):
            parser.error("DGP config must be a mapping")
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(overrides)))
        generated = build_dgp_dataset(config)
        if args.output is not None:
            save_dataset_bundle(generated, cast(Path, args.output))
        dataset = generated
    first = dataset[0]
    graphs = dataset.graphs.aggregate_lags()
    report: dict[str, object] = {
        "fingerprint": dataset.fingerprint,
        "config": OmegaConf.to_container(OmegaConf.structured(dataset.config), resolve=True),
        "episodes": len(dataset),
        "training_shapes": {key: list(value.shape) for key, value in first.items()},
        "first_episode_id": int(first["episode_ids"]),
        "first_frame_range": [
            float(first["observations"][0].min()),
            float(first["observations"][0].max()),
        ],
        "aggregated_edges": {
            "state": int(graphs.state.sum()),
            "action": int(graphs.action.sum()),
            "parameter": int(graphs.parameter.sum()),
        },
        "specification": dataset.specification,
    }
    if args.diagnostics:
        if dataset.config.family != "controlled_vector":
            parser.error("--diagnostics currently reports the controlled-vector constructions")
        from dataclasses import asdict

        report["influence_checks"] = asdict(
            sufficient_influence_diagnostics(cast(ParameterGraph, dataset.config.parameter_graph))
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
