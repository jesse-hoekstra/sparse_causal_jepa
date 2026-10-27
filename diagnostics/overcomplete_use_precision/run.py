"""Predeclared8192-group precision audit of a fixed twelve-code predictor."""

import argparse
import copy
import hashlib
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.overcomplete_acquisition.model import OvercompletePredictor
from diagnostics.overcomplete_acquisition.run import code_identity as source_code_identity
from diagnostics.overcomplete_use_precision.data import build_validation, verify_independence
from diagnostics.overcomplete_use_precision.evaluation import evaluate
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan
from scjepa.training.factory import build_model
from scjepa.training.logging import WandbConfig, WandbLogger

ENDPOINT_SHA256 = "1fd806736bec69528715aa57e0afe1d972436732a769d9706c9f845100509bd3"


@dataclass(frozen=True)
class Config:
    """One fixed sample size/seed, independent of the observed evaluation outcomes."""

    endpoint_dir: str
    source_dir: str
    output_dir: str
    device: str = "cpu"
    groups: int = 8192
    seed: int = 202610271901
    chunk_size: int = 128
    bootstrap_samples: int = 1000
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Prevent accidental changes to the declared scientific sample or source files."""
        if self.verification:
            if not 2 <= self.groups <= 8:
                raise ValueError("Verification requires2..8 fresh groups")
        elif (self.groups, self.seed, self.bootstrap_samples) != (8192, 202610271901, 1000):
            raise ValueError(
                "Scientific precision audit fixes8192groups/seed202610271901/bootstrap1000"
            )
        if min(self.chunk_size, self.threads) < 1 or self.bootstrap_samples < 2 or self.seed < 0:
            raise ValueError("Invalid numeric audit settings")
        output = Path(self.output_dir).resolve()
        for name in (self.endpoint_dir, self.source_dir):
            source = Path(name).resolve()
            if output == source or source in output.parents or output in source.parents:
                raise ValueError("Output must be disjoint from both immutable source roots")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def code_identity() -> str:
    """Bind this audit and the complete unchanged source/evaluation dependency chain."""
    digest = hashlib.sha256(source_code_identity().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_source(
    config: Config,
) -> tuple[SCJEPA, EvaluationData, dict[str, Any], dict[str, Any], dict[str, str]]:
    """Authenticate completed endpoint and original charts; load no TRAIN cache or labels."""
    endpoint, source = Path(config.endpoint_dir), Path(config.source_dir)
    completion = authenticated_directory(endpoint, {"protocol.json", "results.json", "last.pt"})
    protocol = cast(dict[str, Any], read_json(endpoint / "protocol.json"))
    if protocol["code_identity"] != source_code_identity():
        raise ValueError("Current source/evaluation code differs from completed control")
    if not config.verification and sha(endpoint / "last.pt") != ENDPOINT_SHA256:
        raise ValueError("Require the selected fixed100k endpoint")
    hashes = {
        str((endpoint / name).resolve()): value for name, value in completion["hashes"].items()
    }
    hashes[str((endpoint / "completed.json").resolve())] = sha(endpoint / "completed.json")
    for name in ("protocol.json", "evaluation_data.pt"):
        if sha(source / name) != protocol["source_hashes"][name]:
            raise ValueError("Archived evaluation data/source protocol changed")
        hashes[str((source / name).resolve())] = sha(source / name)
    payload = cast(
        dict[str, Any], torch.load(endpoint / "last.pt", map_location="cpu", weights_only=False)
    )
    expected_step = protocol["config"]["steps"] if config.verification else 100000
    if payload["identity"] != protocol["identity"] or payload["step"] != expected_step:
        raise ValueError("Checkpoint identity or fixed-final step differs")
    source_protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    model_config = copy.deepcopy(source_protocol["model_config"])
    model_config["param_dim"] = 12
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        model = build_model(cast(DictConfig, OmegaConf.create(model_config)))
        model.predictor = cast(Spartan, OvercompletePredictor())
    model.load_state_dict(payload["model"], strict=True)
    for name, expected in protocol["frozen_hashes"].items():
        if module_fingerprint(getattr(model, name)) != expected:
            raise ValueError("Frozen state-chart bytes changed")
    data = cast(
        EvaluationData,
        torch.load(source / "evaluation_data.pt", map_location="cpu", weights_only=False),
    )
    return model.requires_grad_(False).eval(), data, protocol, payload["records"][-1], hashes


def verify_replay(actual: dict[str, Any], archived: dict[str, Any]) -> None:
    """Require the same original64-group factual MSE and all original use intervals."""
    torch.testing.assert_close(
        torch.tensor(actual["ordinary_mse_by_horizon"]),
        torch.tensor(archived["ordinary_mse_by_horizon"]),
        atol=1e-8,
        rtol=1e-6,
    )
    for first, second in zip(actual["per_parameter"], archived["per_parameter"], strict=True):
        for name in ("normalized_response_error", "predictive_benefit", "response_energy_ratio"):
            for field in ("estimate", "interval_95"):
                torch.testing.assert_close(
                    torch.tensor(first[name][field]),
                    torch.tensor(second[name][field]),
                    atol=1e-8,
                    rtol=1e-6,
                )


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Evaluate a frozen model once over the predeclared independent validation sample."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out = Path(config.output_dir)
    try:
        with file_lock(out / ".lock"):
            model, old_data, source_protocol, archived, hashes = load_source(config)
            source_model_hash = model_fingerprint(model)
            protocol: dict[str, Any] = {
                "config": asdict(config),
                "source_hashes": hashes,
                "source_model_sha256": source_model_hash,
                "source_identity": source_protocol["identity"],
                "code_identity": code_identity(),
                "command": shlex.join([sys.executable, *sys.argv]),
                "hypothesis": (
                    "More independent groups distinguish sampling noise in the "
                    "residual-response cross term from persistent mean bias."
                ),
                "protocol": (
                    "One new8192-group validation sample; original "
                    "donor0/query5..9, positive min(.01,.5*(.75-abs(theta))) "
                    "interventions, two attached horizons, exact original group "
                    "bootstrap1000/source seed. No optimization, probes or "
                    "adaptive subsets."
                ),
                "interpretation": (
                    "Secondary adaptive precision diagnostic; does not replace or "
                    "relabel the completed64-group final-test criterion."
                ),
                "source_test_access": (
                    "Only exact theta-row identity checks establish no repeated "
                    "old test group; no old-test prediction, fitting or "
                    "selection."
                ),
            }
            evidence = {key: value for key, value in protocol.items() if key != "command"}
            protocol["identity"] = canonical_fingerprint(evidence)
            if (out / "completed.json").exists():
                completed = authenticated_directory(
                    out,
                    {
                        "protocol.json",
                        "results.json",
                        "validation_data.pt",
                        "cache_receipt.json",
                        "measurements.pt",
                        "replay.json",
                    },
                )
                if completed["identity"] != protocol["identity"]:
                    raise ValueError("Completed precision audit identity changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            if (out / "protocol.json").exists():
                if (
                    not resume
                    or read_json(out / "protocol.json")["identity"] != protocol["identity"]
                ):
                    raise ValueError("Existing output requires matching explicit resume")
            else:
                write_json(out / "protocol.json", protocol)
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"overcomplete-use-precision-{out.parent.name}",
                    job_type="diagnostic",
                ),
                resolved_config=protocol,
                dataset_identity=source_protocol["identity"],
                out_dir=out,
                resume_state=read_json(out / "tracking.json")
                if (out / "tracking.json").exists()
                else None,
            )
            write_json(out / "tracking.json", logger.metadata)
            step = 0

            def progress(phase: str, done: int) -> None:
                nonlocal step
                step += 1
                metrics = {f"progress/{phase}_groups": float(done)}
                logger.log(step, metrics)
                print({"step": step, **metrics}, flush=True)

            try:
                model = model.double().to(config.device)
                fixed = model_fingerprint(model)
                count = source_protocol["config"]["groups"]
                replay_data: EvaluationData = {
                    "metadata": old_data["metadata"],
                    "validation": {
                        key: value[:count] for key, value in old_data["validation"].items()
                    },
                }
                replay, _, _ = evaluate(
                    model,
                    replay_data,
                    chunk_size=config.chunk_size,
                    bootstrap_samples=source_protocol["config"]["bootstrap_samples"],
                    seed=source_protocol["config"]["seed"],
                )
                verify_replay(replay, archived["response"])
                write_json(
                    out / "replay.json",
                    {"identity": protocol["identity"], "matched": True, "report": replay},
                )
                progress("replay", count)
                if (out / "validation_data.pt").exists():
                    receipt = read_json(out / "cache_receipt.json")
                    if receipt["identity"] != protocol["identity"] or receipt["sha256"] != sha(
                        out / "validation_data.pt"
                    ):
                        raise ValueError("Cached fresh evaluation data changed")
                    data = cast(
                        EvaluationData,
                        torch.load(
                            out / "validation_data.pt", map_location="cpu", weights_only=False
                        ),
                    )
                else:
                    data = build_validation(
                        model,
                        old_data["metadata"],
                        groups=config.groups,
                        seed=config.seed,
                        batch_size=config.chunk_size,
                        progress=lambda n: progress("cache", n),
                    )
                    atomic_save(out / "validation_data.pt", data)
                    write_json(
                        out / "cache_receipt.json",
                        {
                            "identity": protocol["identity"],
                            "sha256": sha(out / "validation_data.pt"),
                        },
                    )
                independent = verify_independence(data, old_data)
                del old_data
                report, raw, vectors = evaluate(
                    model,
                    data,
                    chunk_size=config.chunk_size,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=source_protocol["config"]["seed"],
                    progress=lambda n: progress("response", n),
                )
                if model_fingerprint(model) != fixed or any(
                    p.requires_grad for p in model.parameters()
                ):
                    raise RuntimeError("Frozen evaluation changed model bytes or enabled gradients")
                for name, expected in hashes.items():
                    if sha(Path(name)) != expected:
                        raise RuntimeError("Immutable source changed during evaluation")
                atomic_save(
                    out / "measurements.pt",
                    {"identity": protocol["identity"], "raw": raw, "vectors": vectors},
                )
                result = {
                    "identity": protocol["identity"],
                    "groups": config.groups,
                    "independence": independent,
                    "report": report,
                    "tracking": logger.metadata,
                    "scope": protocol["interpretation"],
                    "evaluation_model_sha256": fixed,
                }
                write_json(out / "results.json", result)
                metrics: dict[str, float] = {"diagnostic/complete": 1.0}
                for row in report["per_parameter"]:
                    for name in ("normalized_response_error", "predictive_benefit"):
                        key = f"validation/theta_{row['parameter']}/{name}"
                        metrics[key] = float(row[name]["estimate"])
                        metrics[key + "/lower95"], metrics[key + "/upper95"] = row[name][
                            "interval_95"
                        ]
                for horizon, value in enumerate(report["ordinary_mse_by_horizon"], 1):
                    metrics[f"validation/ordinary_mse/horizon_{horizon}"] = float(value)
                for name, values in report["decomposition_means"].items():
                    for factor, value in enumerate(values, 1):
                        metrics[f"validation/theta_{factor}/{name}"] = float(value)
                logger.log(step + 1, metrics)
                for name in ("protocol.json", "results.json", "replay.json"):
                    logger.log_artifact(out / name, name=f"overcomplete-use-{Path(name).stem}")
                write_json(
                    out / "completed.json",
                    {
                        "identity": protocol["identity"],
                        "hashes": {
                            name: sha(out / name)
                            for name in (
                                "protocol.json",
                                "results.json",
                                "replay.json",
                                "validation_data.pt",
                                "cache_receipt.json",
                                "measurements.pt",
                                "tracking.json",
                            )
                        },
                    },
                )
                logger.finish()
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Expose the single fixed scientific run with optional authenticated restart."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-dir", "source-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
