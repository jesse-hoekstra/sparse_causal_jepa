"""Retrospective code-content audit on authenticated, already-trained parameter groups.

Reconstructed physical parameters are evaluation-only. This module never trains
models, optimizes group codes, or writes into a source/cache/curriculum directory.
Readout test groups were all used by the teacher, so this is not unseen-system
acquisition evidence.
"""

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import SourceBundle, build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.multiquery_control.data import TensorSplit, encode_states, module_fingerprint
from diagnostics.multiquery_control.evaluation import parameter_readouts
from diagnostics.multiquery_control.run import atomic_save, sha
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng

READOUT_SEED = 2026102701
READOUT_COUNTS = (4096, 2048, 2048)
STATE_TOLERANCE = 2e-5


def verified_teacher_snapshot(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture immutable teacher-phase inputs without claiming parent-run completion.

    No parent completion manifest is read or synthesized. The receipt records
    byte hashes observed now, bound to the protocol and fixed teacher counters.
    These same hashes must still match after the read-only audit finishes.
    """
    names = ("protocol.json", "initial_group_codes.pt", "teacher_final.pt")
    hashes = {name: sha(output / name) for name in names}
    protocol = cast(dict[str, Any], read_json(output / "protocol.json"))
    config = protocol["config"]
    if (
        config.get("initialization", "warm") not in ("warm", "fresh_zero")
        or config["verification"]
        or tuple(config[name + "_steps"] for name in ("teacher", "amortization", "washout"))
        != (30000, 10000, 10000)
    ):
        raise ValueError("Teacher snapshot requires the scientific 30000/10000/10000 protocol")
    settings = dict(config)
    for key in ("source_dir", "output_dir", "training_source_root", "wandb_mode", "wandb_entity"):
        settings.pop(key)
    evidence = {
        key: protocol[key]
        for key in (
            "settings",
            "source_hashes",
            "initialization",
            "source_step",
            "initial_table_hash",
            "initial_encoder_hash",
            "frozen_hashes",
            "code_identity",
            "torch",
        )
    }
    if settings != protocol["settings"] or canonical_fingerprint(evidence) != protocol["identity"]:
        raise ValueError("Teacher snapshot protocol identity or settings differ")
    initial = cast(
        dict[str, Any],
        torch.load(output / "initial_group_codes.pt", map_location="cpu", weights_only=False),
    )
    teacher = cast(
        dict[str, Any],
        torch.load(output / "teacher_final.pt", map_location="cpu", weights_only=False),
    )
    if (
        initial["identity"] != protocol["identity"]
        or teacher["identity"] != protocol["identity"]
        or teacher["phase"] != "teacher"
        or teacher["phase_step"] != 30000
        or teacher["global_step"] != 30000
    ):
        raise ValueError("Teacher snapshot identity or fixed phase counters differ")
    table = nn.Embedding.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        initial["weight"], freeze=True, sparse=True
    )
    if module_fingerprint(table) != protocol["initial_table_hash"]:
        raise ValueError("Teacher snapshot initial table hash differs")
    verify_snapshot_hashes(output, hashes)
    return protocol, {
        "kind": "read_only_teacher_phase_snapshot",
        "parent_completion_required": False,
        "parent_completion_status": "not_asserted",
        "teacher_global_step": 30000,
        "hashes": hashes,
    }


def verify_snapshot_hashes(output: Path, hashes: dict[str, str]) -> None:
    """Reject mutation of any captured phase input, even if its internal IDs still match."""
    for name, expected in hashes.items():
        if sha(output / name) != expected:
            raise ValueError(f"Teacher snapshot input changed during audit: {name}")


def verified_curriculum(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a completed immutable curriculum before retrospective label access."""
    completed = cast(dict[str, Any], read_json(output / "completed.json"))
    required = {
        "protocol.json",
        "results.json",
        "initial_group_codes.pt",
        "teacher_final.pt",
        "amortization_final.pt",
        "washout_final.pt",
    }
    if not required.issubset(completed["hashes"]):
        raise ValueError("Curriculum completion does not authenticate all audit inputs")
    for name, expected in completed["hashes"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path) != expected:
            raise ValueError(f"Curriculum artifact changed: {name}")
    protocol = cast(dict[str, Any], read_json(output / "protocol.json"))
    result = read_json(output / "results.json")
    if protocol["identity"] != completed["identity"] or result["identity"] != completed["identity"]:
        raise ValueError("Curriculum identity mismatch")
    config = protocol["config"]
    if config.get("initialization", "warm") not in ("warm", "fresh_zero") or config["verification"]:
        raise ValueError("This audit requires a completed warm or fresh_zero scientific control")
    if (
        tuple(config[name + "_steps"] for name in ("teacher", "amortization", "washout"))
        != (30000, 10000, 10000)
        or result["global_step"] != 50000
    ):
        raise ValueError("Curriculum did not finish its original fixed budget")
    return protocol, completed


def prepare_audit_initialization(
    model: SCJEPA, bundle: SourceBundle, curriculum: dict[str, Any]
) -> tuple[Tensor | None, dict[str, str]]:
    """Restore the authenticated initial learned modules on a disposable audit model.

    Warm controls retain the original source model and reconstruct donor-zero
    codes later. Fresh controls reuse source validation and copy only verified
    initial E/P weights; no optimizer or source artifact is changed.
    """
    mode = curriculum["config"].get("initialization", "warm")
    expected_codes: Tensor | None = None
    source_hashes = dict(bundle.hashes)
    if mode == "fresh_zero":
        table, initialization, source_hashes = initialize_control(model, bundle, "fresh_zero")
        if curriculum.get("initialization") != initialization:
            raise ValueError("Fresh curriculum initialization evidence differs from source")
        expected_codes = table.weight.detach().cpu().clone()
        if bool(torch.count_nonzero(expected_codes)):
            raise ValueError("Fresh curriculum must initialize its train table at zero")
    elif mode != "warm":
        raise ValueError("Unsupported curriculum initialization")
    if source_hashes != curriculum["source_hashes"]:
        raise ValueError("Curriculum source/initialization hashes differ from verified source")
    if module_fingerprint(model.parameter_encoder) != curriculum["initial_encoder_hash"]:
        raise ValueError("Curriculum initial encoder differs from authenticated initialization")
    return expected_codes, source_hashes


def verify_initial_table(
    initial: dict[str, Any], curriculum: dict[str, Any], expected_codes: Tensor
) -> Tensor:
    """Authenticate the saved train table and its warm-context or exact-zero origin."""
    if initial["identity"] != curriculum["identity"]:
        raise ValueError("Initial group-code identity differs from curriculum")
    values = cast(Tensor, initial["weight"])
    if values.shape != expected_codes.shape or not bool(torch.isfinite(values).all()):
        raise ValueError("Initial group codes have invalid shape or values")
    table = nn.Embedding.from_pretrained(values, freeze=True, sparse=True)  # pyright: ignore[reportUnknownMemberType]
    if module_fingerprint(table) != curriculum["initial_table_hash"]:
        raise ValueError("Initial group-code table hash differs from curriculum")
    if curriculum["config"].get("initialization", "warm") == "fresh_zero":
        if not torch.equal(values, expected_codes) or bool(torch.count_nonzero(values)):
            raise ValueError("Fresh initial group codes must be exactly zero")
    elif float((values - expected_codes).abs().max()) > STATE_TOLERANCE:
        raise ValueError("Initial group codes do not match original donor-zero group ordering")
    return values


@torch.no_grad()
def reconstruct_training_truth(
    model: SCJEPA,
    train: TensorSplit,
    data_config: dict[str, Any],
    observation_matrix: Tensor,
    expected_dataset_fingerprint: str,
    *,
    batch_size: int = 32,
) -> tuple[Tensor, dict[str, Any]]:
    """Bind labels only after all addressed actions and frozen states reproduce.

    No seed search, inverse fitting, truncation, or tolerance fallback is allowed.
    Every group and every cached frame is checked. Returned labels cannot exist
    at the caller until the complete cache check has passed.
    """
    if set(train) != {"online", "target", "actions"} or batch_size < 1:
        raise ValueError("Require the original label-free training cache and positive batch size")
    config = {**data_config, "split": "train", "num_episodes": len(train["online"])}
    dataset = build_dgp_dataset(OmegaConf.create(config), observation_matrix=observation_matrix)
    if dataset.fingerprint != expected_dataset_fingerprint:
        raise ValueError("Reconstructed training dataset fingerprint differs from recorded cache")
    if (
        config["family"] != "simplified_gaussian"
        or config["episodes_per_parameter"] != 10
        or config["trajectory_length"] != 12
    ):
        raise ValueError("Require the original simplified Gaussian R10/T12 generator")
    observation_map = VectorObservationMap(matrix=observation_matrix, dimension=6)
    labels: list[Tensor] = []
    maxima = {"online": 0.0, "target": 0.0}
    action_hash = hashlib.sha256()
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            episodes = [dataset.process.generate(index) for index in range(start, end)]
            for episode in episodes:
                truth = episode.ground_truth
                physical = dgp.simulate(
                    truth["initial_state"], truth["parameters"], truth["actions"]
                )
                observed = observation_map.forward(physical).to(episode.observations)
                if not torch.equal(observed, episode.observations):
                    raise ValueError("Reconstructed parameters do not reproduce their observations")
                if not torch.equal(truth["actions"].to(episode.actions), episode.actions):
                    raise ValueError("Reconstructed physical actions differ from stored actions")
            actions = torch.stack([episode.actions for episode in episodes])
            if not torch.equal(actions, train["actions"][start:end]):
                raise ValueError(f"Regenerated actions differ from cached groups {start}:{end}")
            action_hash.update(actions.contiguous().numpy().tobytes())
            online, target = encode_states(
                model, torch.stack([episode.observations for episode in episodes]), batch_size
            )
            for name, actual in (("online", online), ("target", target)):
                cached = train[name][start:end]
                if (
                    actual.shape != cached.shape
                    or not bool(torch.isfinite(actual).all())
                    or not bool(torch.isfinite(cached).all())
                ):
                    raise ValueError(f"Regenerated {name} states are invalid")
                maximum = float((actual.double() - cached.double()).abs().max())
                maxima[name] = max(maxima[name], maximum)
                if maximum > STATE_TOLERANCE:
                    raise ValueError(f"Regenerated {name} states differ from cache: {maximum}")
            labels.extend(episode.ground_truth["parameters"].detach().cpu() for episode in episodes)
    finally:
        model.train(was_training)
        model.online_encoder.eval()
        model.target_encoder.eval()
    theta = torch.stack(labels)
    if theta.shape != (len(dataset), 6) or not bool(torch.isfinite(theta).all()):
        raise ValueError("Reconstructed training labels have invalid shape or values")
    return theta, {
        "dataset_fingerprint": dataset.fingerprint,
        "data_config": config,
        "groups_verified": len(dataset),
        "all_actions_equal": True,
        "all_parameter_payloads_reproduce_observations": True,
        "ordered_action_bytes_sha256": action_hash.hexdigest(),
        "state_max_absolute_error": maxima,
        "state_tolerance": STATE_TOLERANCE,
        "theta_tensor_fingerprint": tensor_fingerprint(theta),
        "binding": (
            "Every addressed group verified against all cached actions and state encodings "
            "before labels were returned"
        ),
    }


@torch.no_grad()
def donor_zero_codes(model: SCJEPA, train: TensorSplit, batch_size: int) -> Tensor:
    """Infer each training group's donor-zero context without any latent-table access."""
    if set(train) != {"online", "target", "actions"}:
        raise ValueError("Code inference requires label-free cached inputs only")
    reference = next(model.parameters())
    pieces: list[Tensor] = []
    for start in range(0, len(train["online"]), batch_size):
        states = train["online"][start : start + batch_size, 0, :10].to(reference)
        actions = train["actions"][start : start + batch_size, 0, :9].to(reference)
        pieces.append(model.parameter_encoder(states, actions).detach().cpu())
    return torch.cat(pieces)


def audit(
    curriculum_output: Path,
    output: Path,
    source_run_config: Path | None,
    *,
    device: str = "cpu",
    wandb_mode: str = "disabled",
    teacher_snapshot: bool = False,
) -> dict[str, Any]:
    """Read immutable artifacts and evaluate code contents using group-disjoint probes."""
    curriculum_output, output = curriculum_output.resolve(), output.resolve()
    curriculum, completion = (
        verified_teacher_snapshot(curriculum_output)
        if teacher_snapshot
        else verified_curriculum(curriculum_output)
    )
    config = curriculum["config"]
    source = Path(config["source_dir"]).resolve()
    if any(output == path or output.is_relative_to(path) for path in (source, curriculum_output)):
        raise ValueError("Audit output must be outside source and curriculum directories")
    if output.exists():
        raise FileExistsError("Use a fresh audit output directory")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    target_device = torch.device(device)
    try:
        with preserve_rng(target_device):
            bundle = load_primary_source(
                source,
                config["seed"],
                training_source_root=Path(config["training_source_root"])
                if config["training_source_root"]
                else None,
            )
            if bundle.train_groups != sum(READOUT_COUNTS):
                raise ValueError(
                    "Audit readout partition requires all 8192 original training groups"
                )
            primary, _ = build_source_model(bundle, target_device)
            primary.eval().requires_grad_(False)
            expected_initial, source_hashes = prepare_audit_initialization(
                primary, bundle, curriculum
            )
            provenance = bundle.protocol["source"]
            record_path = (
                source_run_config or Path(provenance["checkpoint"]).parent / "run_config.json"
            )
            record = cast(dict[str, Any], read_json(record_path))
            if (
                record["run_identity"] != provenance["run_identity"]
                or record["dataset_identity"] != provenance["dataset_identity"]
                or record["model"] != bundle.protocol["model_config"]
            ):
                raise ValueError("Source run configuration differs from recorded provenance")
            metadata = bundle.evaluation["metadata"]
            dataset = build_dgp_dataset(
                cast(DictConfig, OmegaConf.create(record["data"])),
                observation_matrix=metadata["observation_matrix"],
            )
            if dataset.fingerprint != provenance["dataset_identity"]:
                raise ValueError("Source run data settings do not reproduce its original identity")
            evidence: dict[str, Any] = {
                "curriculum_identity": curriculum["identity"],
                "curriculum_artifact_hashes": completion["hashes"],
                "source_hashes": source_hashes,
                "initialization": config.get("initialization", "warm"),
                "initialization_code_sha256": sha(
                    Path(__file__).parent / "group_latent_control/run.py"
                ),
                "source_run_config_sha256": sha(record_path),
                "training_dataset_fingerprint": metadata["training_dataset_fingerprint"],
                "training_source_root": config["training_source_root"],
                "readout_seed": READOUT_SEED,
                "readout_group_counts": READOUT_COUNTS,
                "context": "donor 0; complete individual-context code, no context averaging",
                "scope": (
                    "All 8192 groups were used to train teacher codes; this is retrospective "
                    "code-content evidence, NOT unseen-system acquisition"
                ),
                "labels": (
                    "Reconstructed only in this evaluation-only audit after full cache "
                    "verification; never passed to model training"
                ),
                "audit_code_sha256": sha(Path(__file__)),
                "readout_code_sha256": sha(
                    Path(__file__).parent / "multiquery_control/evaluation.py"
                ),
                "cache_builder_code_sha256": sha(
                    Path(__file__).parent / "multiquery_control/data.py"
                ),
            }
            if teacher_snapshot:
                evidence["phase_snapshot"] = completion
                evidence["completion_scope"] = (
                    "Completion means this read-only teacher-phase audit finished; "
                    "the parent curriculum need not have completed its later phases."
                )
            identity = canonical_fingerprint(evidence)
            output.mkdir(parents=True)
            write_json(output / "protocol.json", {"identity": identity, **evidence})
            logger = WandbLogger(
                WandbConfig(
                    mode=wandb_mode,
                    project=config["wandb_project"],
                    entity=config["wandb_entity"],
                    name=output.parent.name,
                    job_type="diagnostic",
                ),
                resolved_config=evidence,
                dataset_identity=bundle.protocol["identity"],
                out_dir=output,
            )
            write_json(output / "tracking.json", logger.metadata)
            try:
                logger.log(0, {"audit/reconstruction_complete": 0.0})
                theta, reconstruction = reconstruct_training_truth(
                    primary,
                    bundle.train,
                    record["data"],
                    metadata["observation_matrix"],
                    metadata["training_dataset_fingerprint"],
                )
                logger.log(
                    0,
                    {
                        "audit/reconstruction_complete": 1.0,
                        "audit/groups_verified": float(len(theta)),
                    },
                )
                initial = cast(
                    dict[str, Any],
                    torch.load(
                        curriculum_output / "initial_group_codes.pt",
                        map_location="cpu",
                        weights_only=False,
                    ),
                )
                if expected_initial is None:
                    expected_initial = donor_zero_codes(primary, bundle.train, bundle.batch_size)
                initial_values = verify_initial_table(initial, curriculum, expected_initial)
                variants: dict[str, Tensor] = {"initial_group_table": initial_values}
                expected_global = (
                    {"teacher": 30000}
                    if teacher_snapshot
                    else {"teacher": 30000, "amortization": 40000, "washout": 50000}
                )
                frozen = {
                    name: module_fingerprint(module)
                    for name, module in (
                        ("online", primary.online_encoder),
                        ("target", primary.target_encoder),
                    )
                }
                for phase in expected_global:
                    checkpoint = cast(
                        dict[str, Any],
                        torch.load(
                            curriculum_output / f"{phase}_final.pt",
                            map_location="cpu",
                            weights_only=False,
                        ),
                    )
                    if (
                        checkpoint["identity"] != curriculum["identity"]
                        or checkpoint["phase"] != phase
                        or checkpoint["phase_step"] != config[phase + "_steps"]
                        or checkpoint["global_step"] != expected_global[phase]
                    ):
                        raise ValueError("Curriculum phase checkpoint identity/counters differ")
                    candidate = copy.deepcopy(primary)
                    candidate.load_state_dict(checkpoint["model"], strict=True)
                    if (
                        any(
                            module_fingerprint(module) != frozen[name]
                            for name, module in (
                                ("online", candidate.online_encoder),
                                ("target", candidate.target_encoder),
                            )
                        )
                        or candidate.get_extra_state() != primary.get_extra_state()
                    ):
                        raise ValueError(
                            "Curriculum checkpoint changed frozen state maps or rollout"
                        )
                    if phase == "teacher":
                        if module_fingerprint(candidate.parameter_encoder) != module_fingerprint(
                            primary.parameter_encoder
                        ):
                            raise ValueError("Teacher changed its frozen context encoder")
                        variants["learned_teacher_table"] = checkpoint["table"]["weight"]
                    else:
                        if phase == "amortization" and not torch.equal(
                            checkpoint["table"]["weight"], variants["learned_teacher_table"]
                        ):
                            raise ValueError("Amortization changed its teacher target table")
                        if phase == "washout" and checkpoint["table"] is not None:
                            raise ValueError("Washout retained the teacher table")
                        variants[phase + "_encoder_donor0"] = donor_zero_codes(
                            candidate, bundle.train, bundle.batch_size
                        )
                generator = torch.Generator().manual_seed(READOUT_SEED)
                ordered = torch.randperm(len(theta), generator=generator)
                indices: dict[str, Tensor] = {}
                start = 0
                for name, count in zip(
                    ("calibration", "validation", "test"), READOUT_COUNTS, strict=True
                ):
                    indices[name] = ordered[start : start + count]
                    start += count
                reports: dict[str, Any] = {}
                for name, values in variants.items():
                    if values.shape != theta.shape or not bool(torch.isfinite(values).all()):
                        raise ValueError(f"Invalid audited code table: {name}")
                    codes = {key: values[groups, None] for key, groups in indices.items()}
                    splits = {key: {"theta": theta[groups]} for key, groups in indices.items()}
                    reports[name] = parameter_readouts(codes, splits)
                    reports[name]["scope"] = evidence["scope"]
                    logger.log(
                        0,
                        {
                            f"trained_groups/{name}/theta_{i + 1}/readout_test_r2": float(value)
                            for i, value in enumerate(reports[name]["selected_test_r2"])
                        },
                    )
                result = {
                    "identity": identity,
                    "curriculum_identity": curriculum["identity"],
                    "reconstruction": reconstruction,
                    "reports": reports,
                    "scope": evidence["scope"],
                    "tracking": logger.metadata,
                }
                if teacher_snapshot:
                    result["phase_snapshot"] = completion
                    result["completion_scope"] = evidence["completion_scope"]
                write_json(output / "results.json", result)
                atomic_save(
                    output / "audit_inputs.pt",
                    {
                        "theta_evaluation_only": theta,
                        "codes": variants,
                        "readout_group_indices": indices,
                        "identity": identity,
                    },
                )
                logger.log_artifact(
                    output / "results.json", name=f"group-table-audit-{identity[:12]}-results"
                )
                if teacher_snapshot:
                    verify_snapshot_hashes(curriculum_output, completion["hashes"])
                    if sha(record_path) != evidence["source_run_config_sha256"]:
                        raise RuntimeError("Source run configuration changed during snapshot audit")
                else:
                    verified_curriculum(curriculum_output)
                for name, expected in source_hashes.items():
                    if sha(source / name) != expected:
                        raise RuntimeError("Source changed during retrospective audit")
                logger.finish()
                write_json(
                    output / "completed.json",
                    {
                        "identity": identity,
                        **(
                            {
                                "completion_scope": "teacher_phase_audit_only",
                                "parent_complete": None,
                            }
                            if teacher_snapshot
                            else {}
                        ),
                        "hashes": {
                            name: sha(output / name)
                            for name in ("results.json", "audit_inputs.pt", "protocol.json")
                        },
                    },
                )
                return result
            except BaseException:
                logger.log(0, {"audit/failed": 1.0})
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Audit a completed warm or fresh-zero curriculum on a fresh output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teacher-snapshot", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    args = parser.parse_args()
    result = audit(
        args.curriculum_output,
        args.output_dir,
        args.source_run_config,
        device=args.device,
        wandb_mode=args.wandb_mode,
        teacher_snapshot=args.teacher_snapshot,
    )
    print(result["identity"])


if __name__ == "__main__":
    main()
