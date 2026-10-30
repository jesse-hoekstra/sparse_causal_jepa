"""Validation-only physical-response assay for the Paley24 experiment.

True parameters and physical states enter only this frozen-model evaluation.
The donor context is regenerated after changing one physical parameter; its
entire inferred code predicts the unchanged recipient history and actions.
No code coordinate is assumed to correspond to a physical parameter.
The explicit oracle-control mode supplies true parameters and bypasses the
unused parameter encoder; it does not fit or modify any model.
"""

import argparse
import math
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import dgp
from diagnostics.balanced_action_oracle.data import StateData, load_data
from diagnostics.joint_acquisition.baseline import MatchedBatch, MatchedPredictions, response_report
from diagnostics.masked_mlp_learned import run
from diagnostics.masked_mlp_oracle import from_dense
from diagnostics.masked_mlp_oracle import run as oracle_run
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from diagnostics.overcomplete_use_precision.evaluation import direct_decomposition, join_batches
from diagnostics.signal_path.geometry import predict_rollout
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger

BOOTSTRAP_SEED = 202610261301


@torch.no_grad()
def matched_batch(
    model: SCJEPA,
    data: StateData,
    observation: VectorObservationMap,
    indices: tuple[int, ...],
) -> MatchedBatch:
    """Regenerate matched donor0/query5-9 responses on explicit validation groups."""
    if data.split != "validation" or data.states is None:
        raise ValueError("Physical responses require validation physical references")
    if (
        model.training
        or model.context_len != 10
        or not model.autoregressive
        or model.predictor.history_len != 1
        or next(model.parameters()).dtype != torch.float64
    ):
        raise ValueError("Require a float64 eval-mode K10/L1 autoregressive model")
    if (
        not indices
        or min(indices) < 0
        or max(indices) >= len(data)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Require distinct in-range validation group indices")
    if data.metadata["action_mix"] != "paley24":
        raise ValueError("Require the balanced Paley24 action chart")
    reference = next(model.parameters())
    theta = data.tensors["oracle_parameters"][list(indices)].to(reference)
    states = data.states[list(indices)].to(reference)
    actions = data.tensors["actions"][list(indices)].to(reference)
    observations = data.tensors["observations"][list(indices)].to(reference)
    groups = len(indices)
    remaining = dgp.PARAMETER_BOUND - theta.abs()
    if not bool((remaining > 0).all()):
        raise ValueError("Parameters must lie strictly inside their support")
    steps = (remaining * 0.5).clamp(max=0.01)
    changed_theta = theta[:, None] + torch.diag_embed(steps)
    rho = float(data.metadata["rho"])
    rotation_scale = float(data.metadata["rotation_scale"])
    queries = torch.tensor([[5, 6, 7, 8, 9]], device=reference.device)
    donor_actions = actions[:, :1, :9]
    initial = states[:, :1, 0]
    physical = dgp.simulate(
        initial, theta[:, None], donor_actions, rho=rho, rotation_scale=rotation_scale
    )
    changed_physical = dgp.simulate(
        initial[:, :, None].expand(-1, -1, 6, -1),
        changed_theta[:, None],
        donor_actions[:, :, None].expand(-1, -1, 6, -1, -1),
        rho=rho,
        rotation_scale=rotation_scale,
    )
    donor = model.online_encoder.encode_sequence(
        observation.forward(physical).flatten(0, 1)
    ).reshape(groups, 1, 10, 6)
    changed_donor = model.online_encoder.encode_sequence(
        observation.forward(changed_physical).flatten(0, 2)
    ).reshape(groups, 1, 6, 10, 6)
    query_states, query_actions = states[:, queries, 9], actions[:, queries, 9:11]
    history = model.online_encoder.encode_sequence(
        observation.forward(query_states).reshape(-1, 1, 6)
    ).reshape(groups, 1, 5, 1, 6)
    futures = dgp.simulate(
        query_states, theta[:, None, None], query_actions, rho=rho, rotation_scale=rotation_scale
    )[..., 1:, :]
    targets = model.target_encoder.encode_sequence(
        observation.forward(futures).flatten(0, 2)
    ).reshape(groups, 1, 5, 2, 6)
    changed_futures = dgp.simulate(
        query_states[:, :, None].expand(-1, -1, 6, -1, -1),
        changed_theta[:, None, :, None],
        query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1),
        rho=rho,
        rotation_scale=rotation_scale,
    )[..., 1:, :]
    changed_targets = model.target_encoder.encode_sequence(
        observation.forward(changed_futures).flatten(0, 3)
    ).reshape(groups, 1, 6, 5, 2, 6)
    cached_online = model.online_encoder.encode_sequence(observations.flatten(0, 1)).reshape(
        groups, 10, 12, 6
    )
    cached_target = model.target_encoder.encode_sequence(observations.flatten(0, 1)).reshape(
        groups, 10, 12, 6
    )
    errors = (
        (physical - states[:, :1, :10]).abs().max(),
        (futures - states[:, queries, 10:12]).abs().max(),
        (observation.forward(states) - observations).abs().max(),
        (donor - cached_online[:, :1, :10]).abs().max(),
        (history - cached_online[:, queries, 9:10]).abs().max(),
        (targets - cached_target[:, queries, 10:12]).abs().max(),
    )
    error = max(float(value) for value in errors)
    if not math.isfinite(error) or error > 2e-5:
        raise ValueError(f"DGP/observation/state reconstruction differs: {error}")
    return MatchedBatch(
        donor,
        donor_actions,
        changed_donor,
        history,
        query_actions,
        targets,
        changed_targets,
        steps,
        indices,
        (0,),
        queries.cpu(),
        "validation",
        0,
        error,
    )


def predict_matched(
    model: SCJEPA, batch: MatchedBatch, *, oracle_parameters: Tensor | None = None
) -> MatchedPredictions:
    """Use complete donor codes, or explicit oracle-control inputs, at both forecast steps."""
    width = model.predictor.param_dim
    if width not in (6, 12) or model.parameter_encoder.param_dim != width:
        raise ValueError("Require matching six- or twelve-dimensional code interfaces")
    batch = batch.to(next(model.parameters()))
    groups, donors, queries = batch.history.shape[:3]
    if oracle_parameters is None:
        codes = cast(
            Tensor,
            model.parameter_encoder(
                batch.donor_states.flatten(0, 1), batch.donor_actions.flatten(0, 1)
            ),
        ).reshape(groups, donors, width)
        changed_codes = cast(
            Tensor,
            model.parameter_encoder(
                batch.changed_donor_states.flatten(0, 2),
                batch.donor_actions[:, :, None].expand(-1, -1, 6, -1, -1).flatten(0, 2),
            ),
        ).reshape(groups, donors, 6, width)
    else:
        if width != 6 or oracle_parameters.shape != (groups, 6) or batch.split != "validation":
            raise ValueError("Oracle response requires six parameters per validation group")
        codes = oracle_parameters.to(batch.history)[:, None].expand(-1, donors, -1)
        changed_codes = codes[:, :, None] + torch.diag_embed(batch.finite_steps)[:, None]
    base = predict_rollout(
        model,
        batch.history.flatten(0, 2),
        codes[:, :, None].expand(-1, -1, queries, -1).flatten(0, 2),
        batch.query_actions.flatten(0, 2),
    ).reshape(groups, donors, queries, 2, 6)
    changed = predict_rollout(
        model,
        batch.history[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
        changed_codes[:, :, :, None].expand(-1, -1, -1, queries, -1).flatten(0, 3),
        batch.query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
    ).reshape(groups, donors, 6, queries, 2, 6)
    return MatchedPredictions(base, changed, codes, changed_codes)


@torch.no_grad()
def assess(
    model: SCJEPA,
    data: StateData,
    observation: VectorObservationMap,
    *,
    chunk_size: int = 16,
    parameter_source: Literal["learned", "oracle"] = "learned",
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Assess all existing validation groups with one fixed independent-group bootstrap."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    if parameter_source not in {"learned", "oracle"}:
        raise ValueError("Unknown parameter source")
    batches: list[MatchedBatch] = []
    predictions: list[Tensor] = []
    changes: list[Tensor] = []
    for start in range(0, len(data), chunk_size):
        batch = matched_batch(
            model, data, observation, tuple(range(start, min(start + chunk_size, len(data))))
        )
        parameters = (
            data.tensors["oracle_parameters"][list(batch.group_indices)]
            if parameter_source == "oracle"
            else None
        )
        result = predict_matched(model, batch, oracle_parameters=parameters)
        batches.append(batch.to(torch.zeros((), dtype=torch.float64)))
        predictions.append(result.predictions.cpu())
        changes.append(result.changed_predictions.cpu())
    joined = join_batches(batches)
    base, changed = torch.cat(predictions), torch.cat(changes)
    report, raw = response_report(
        base, changed, joined, bootstrap_samples=1000, seed=BOOTSTRAP_SEED
    )
    variance = joined.targets.permute(3, 0, 1, 2, 4).flatten(1, 3).var(1, correction=0).mean(-1)
    report["ordinary_variance_normalized_mse_by_horizon"] = [
        float(error / scale)
        for error, scale in zip(report["ordinary_mse_by_horizon"], variance, strict=True)
    ]
    report["target_variance_definition"] = (
        "Mean per-coordinate population variance across validation groups and recipients"
    )
    decomposition = direct_decomposition(base, changed, joined.targets, joined.changed_targets)
    torch.testing.assert_close(
        raw["predictive_benefit"],
        decomposition["benefit_from_direct_vectors"],
        atol=2e-15,
        rtol=1e-9,
    )
    for row in report["per_parameter"]:
        nre, benefit = row["normalized_response_error"], row["predictive_benefit"]
        row["response_fidelity_pass"] = bool(nre["defined"] and nre["interval_95"][1] < 1)
        row["positive_benefit_pass"] = bool(benefit["defined"] and benefit["interval_95"][0] > 0)
    report.update(
        parameter_source=parameter_source,
        all_six_response_fidelity_pass=all(
            row["response_fidelity_pass"] for row in report["per_parameter"]
        ),
        historical_all_six_predictive_use_pass=all(
            row["response_fidelity_pass"] and row["positive_benefit_pass"]
            for row in report["per_parameter"]
        ),
        reconstruction_error=joined.reconstruction_error,
        decomposition_means={
            key: [float(value) for value in tensor.mean(0)] for key, tensor in decomposition.items()
        },
        interpretation=(
            "Frozen final-budget checkpoint; validation only. "
            + (
                "Complete single-context learned codes. "
                if parameter_source == "learned"
                else "Explicit oracle-control parameter inputs; context encoder bypassed. "
            )
            + "NRE and signed benefit remain separate. Factorwise intervals are not simultaneous. "
            "No probe, code-coordinate alignment, fitting, or checkpoint selection enters."
        ),
    )
    return report, raw | decomposition | {
        "factual_predictions": base,
        "counterfactual_predictions": changed,
        "factual_targets": joined.targets,
        "counterfactual_targets": joined.changed_targets,
        "finite_steps": joined.finite_steps,
    }


def main() -> None:
    """Authenticate an immutable endpoint and save a new validation-only evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--parameter-source", choices=("learned", "oracle"), default="learned")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="disabled"
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="masked-mlp-physical-response")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = cast(Path, args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = cast(Path, args.run_dir).resolve()
    if args.parameter_source == "oracle":
        verifiers = {
            oracle_run.SCHEMA: oracle_run.verify_completion,
            from_dense.SCHEMA: from_dense.verify_completion,
        }
        schema = read_json(source / "completion.json").get("schema")
        if schema not in verifiers:
            raise ValueError("Oracle response requires an authenticated oracle checkpoint")
        completion = verifiers[schema](source, kind="train")
    else:
        completion = run.verify_completion(source)
    if completion["kind"] not in {"train", "projection"}:
        raise ValueError("Require a completed final-budget model")
    data, cache = load_data(args.data_dir)
    if cache["identity"] != completion["data_identity"]:
        raise ValueError("Endpoint and evaluation data identities differ")
    checkpoint = cast(
        dict[str, Any], torch.load(source / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        checkpoint["step"] != completion["steps"]
        or checkpoint["identity"] != completion["identity"]
    ):
        raise ValueError("Require the authenticated fixed-budget checkpoint")
    model = (
        oracle_run.build_control(completion["condition"], completion["seed"])
        if args.parameter_source == "oracle"
        else run.build_control(completion["condition"], completion["seed"], completion["code_dim"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=args.device, dtype=torch.float64).requires_grad_(False).eval()
    matrix = cast(
        dict[str, Tensor], torch.load(args.data_dir / "observation_map.pt", weights_only=True)
    )["matrix"]
    report, raw = assess(
        model,
        data["validation"],
        VectorObservationMap(matrix=matrix, dimension=6),
        parameter_source=args.parameter_source,
    )
    output.mkdir(parents=True)
    save_torch(output / "responses.pt", raw)
    write_json(
        output / "report.json",
        {
            "source": str(source),
            "source_identity": completion["identity"],
            "source_code_identity": completion["code_identity"],
            "checkpoint_sha256": sha(source / "last.pt"),
            "data_identity": cache["identity"],
            "evaluator_sha256": sha(Path(__file__)),
            "runtime_code_identity": run.code_identity(),
            "torch": str(torch.__version__),
            "evaluation_dependency_sha256": {
                name: sha(Path(__file__).resolve().parents[1] / name)
                for name in (
                    "joint_acquisition/baseline.py",
                    "overcomplete_use_precision/evaluation.py",
                    "signal_path/geometry.py",
                )
            },
            "responses_sha256": sha(output / "responses.pt"),
            "code_dimension": model.predictor.param_dim,
            "parameter_source": args.parameter_source,
            "split": "validation",
            "response": report,
        },
    )
    logger = WandbLogger(
        WandbConfig(
            mode=args.wandb_mode,
            project="sparse_causal_jepa",
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=output.name,
            job_type="validation-physical-response",
        ),
        resolved_config={
            "source_identity": completion["identity"],
            "split": "validation",
            "parameter_source": args.parameter_source,
        },
        dataset_identity=cache["identity"],
        out_dir=output,
    )
    try:
        write_json(output / "wandb.json", logger.metadata)
        logger.log(
            completion["steps"],
            {
                f"theta_{row['parameter']}/{metric}": float(row[metric]["estimate"])
                for row in report["per_parameter"]
                for metric in ("normalized_response_error", "predictive_benefit")
                if row[metric]["defined"]
            },
        )
        logger.log_artifact(output / "report.json", name=output.name)
        logger.finish()
    except BaseException:
        logger.finish(exit_code=1)
        raise
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
