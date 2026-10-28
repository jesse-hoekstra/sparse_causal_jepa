"""Authenticate admission, calibrate one shared GECO bound, and verify disposable native updates."""

import argparse
import copy
import math
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.global_code_calibration.source import verify_files
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.sparse_comparison.artifacts import code_identity, disjoint
from diagnostics.sparse_comparison.source import load_inputs
from diagnostics.sparse_comparison.training import CONDITIONS, assert_trainer, make_trainer, step
from diagnostics.table_settling_control.source import authenticated_directory, model_fingerprint
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.evaluation import calibrate_tau
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """One immutable raw-source admission and full-base calibration shared by all three arms."""

    endpoint_dir: str
    output_dir: str
    device: str = "cpu"
    threads: int = 1
    verification: bool = False
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Protect source artifacts and validate operational settings."""
        if self.threads < 1:
            raise ValueError("Require positive threads")
        disjoint(Path(self.output_dir), [Path(self.endpoint_dir)])
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Prepare no trained candidate: all three one-update copies are discarded."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out, device = Path(config.output_dir), torch.device(config.device)
    try:
        inputs = load_inputs(
            Path(config.endpoint_dir), device=device, verification=config.verification
        )
        disjoint(out, [Path(name).parent for name in inputs.hashes])
        with file_lock(out / ".lock"):
            evidence = {
                "source_hashes": inputs.hashes,
                "source_provenance": inputs.provenance,
                "native_evaluation": asdict(inputs.native_evaluation),
                "ordinary_objective": asdict(inputs.objective_config),
                "code_identity": code_identity(),
                "verification": config.verification,
                "torch": str(torch.__version__),
            }
            identity = canonical_fingerprint(evidence)
            protocol: dict[str, Any] = {
                "identity": identity,
                **evidence,
                "config": asdict(config),
                "command": shlex.join([sys.executable, *sys.argv]),
                "source_step": inputs.source_step,
                "stream_seed": inputs.seed,
                "admission": (
                    "All-six complete-code retention R2>=.8 and nonzero physical response "
                    "energy with NREupper95<1 on authenticated existing8192 validation. "
                    "Historical one-sided benefit failures remain reported; no "
                    "calibrated-source selection."
                ),
                "comparison": (
                    "Three fresh AdamW initializations from identical RAW200k weights, "
                    "fixed30000 additional updates per arm; no automatic extension, "
                    "checkpoint selection, source offsets or post-fit corrections. "
                    "Online/E/P train, target EMA.996; native dense/baseline/full gate "
                    "policies and ordinary objective."
                ),
                "rng": (
                    "Bootstrap seed0 only for native Trainer initialization. Original "
                    "large stream seed governs global groups/donors/gates; isolated "
                    "forward seed=stream*100000+globalstep, SIGRegseed=forwardseed+2**40. "
                    "Identical native objective projections across conditions; "
                    "baseline/full share gate uniforms."
                ),
                "tau": (
                    "Native512cal/B16/seed42 full ordinary base loss on exact raw "
                    "initializer; multiplier1.0, no feasibility tolerance or "
                    "outcome-driven adjustment. Fresh sparse "
                    "lambda1000/step.1/momentum.99/logbounds[-12,12]. Native calibration "
                    "cohort differs from old diagnostic readout/response cohort; both are "
                    "reused, not fresh confirmation."
                ),
            }
            if (out / "protocol.json").exists():
                if not resume or read_json(out / "protocol.json")["identity"] != identity:
                    raise ValueError(
                        "Matching explicit preparation resume required; source/code changed"
                    )
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a fresh preparation output or matching resume")
            else:
                write_json(out / "protocol.json", protocol)
            if (out / "completed.json").exists():
                completion = authenticated_directory(
                    out,
                    {
                        "protocol.json",
                        "results.json",
                        "tau.json",
                        "startup.json",
                        "preflight.json",
                        "initial_model.pt",
                        "tracking.json",
                    },
                )
                if completion["identity"] != identity:
                    raise ValueError("Preparation completion identity differs")
                return cast(dict[str, Any], read_json(out / "results.json"))
            tracking = (
                read_json(out / "tracking.json") if (out / "tracking.json").exists() else None
            )
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=out.parent.name,
                    job_type="diagnostic-prepare",
                ),
                resolved_config=protocol,
                dataset_identity=inputs.provenance["dataset_identity"],
                out_dir=out,
                resume_state=tracking if tracking and tracking.get("id") else None,
            )
            write_json(out / "tracking.json", logger.metadata)
            event = 0

            def log(metrics: dict[str, float]) -> None:
                nonlocal event
                event += 1
                logger.log(event, metrics)
                print({"event": event, **metrics}, flush=True)

            try:
                log(
                    {
                        "admission/revised_pass": float(inputs.admission["passed"]),
                        "admission/historical_all_six_use_pass": float(
                            inputs.admission["historical_all_six_use_passed"]
                        ),
                    }
                )
                if not config.verification and not inputs.admission["passed"]:
                    raise ValueError(
                        "Authenticated raw-source retention/physical-response admission failed"
                    )
                initial_hash = model_fingerprint(inputs.model)
                atomic_save(
                    out / "initial_model.pt",
                    {
                        "identity": identity,
                        "model": inputs.model.state_dict(),
                        "model_sha256": initial_hash,
                    },
                )
                with preserve_rng(device):
                    bound = calibrate_tau(
                        inputs.model,
                        inputs.data_config,
                        inputs.evaluation["metadata"]["observation_matrix"],
                        inputs.native_evaluation,
                        objective=inputs.objective_config,
                    )
                tau = bound["calibration/base_loss"]
                if not math.isfinite(tau) or tau <= 0:
                    raise ValueError(
                        "Native full-base GECO calibration must be finite and positive"
                    )
                tau_record = {
                    "identity": identity,
                    "source_checkpoint_sha256": inputs.provenance["checkpoint_sha256"],
                    "source_model_sha256": initial_hash,
                    "calibration": asdict(inputs.native_evaluation),
                    "objective": asdict(inputs.objective_config),
                    "tau_multiplier": 1.0,
                    "tau": tau,
                    **bound,
                }
                write_json(out / "tau.json", tau_record)
                log({"calibration/tau": tau, **bound})
                startup: dict[str, Any] = {}
                preflight: dict[str, Any] = {}
                for condition in CONDITIONS:
                    with preserve_rng(device):
                        trainer = make_trainer(
                            inputs, condition, out / ("discarded_" + condition), tau=tau, steps=1
                        )
                        assert_trainer(trainer, inputs, condition, tau=tau)
                        if model_fingerprint(trainer.model) != initial_hash:
                            raise RuntimeError("Condition initialization tensors differ")
                        bases = {}
                        for sampled in (False, True) if condition != "dense" else (False,):
                            report = calibrate_tau(
                                trainer.model,
                                inputs.data_config,
                                inputs.evaluation["metadata"]["observation_matrix"],
                                inputs.native_evaluation,
                                objective=trainer.config.objective,
                                training_gates=sampled,
                            )
                            base = report["calibration/base_loss"]
                            if not all(math.isfinite(value) for value in report.values()):
                                raise FloatingPointError("Nonfinite condition startup base")
                            label = "sampled" if sampled else "deterministic"
                            bases[label] = {
                                **report,
                                "base_minus_tau": base - tau,
                                "feasible": base <= tau,
                            }
                            log(
                                {
                                    f"startup/{condition}/{label}/base": base,
                                    f"startup/{condition}/{label}/base_minus_tau": base - tau,
                                }
                            )
                        startup[condition] = bases
                        before = copy.deepcopy(trainer.model.target_encoder.state_dict())
                        metrics = step(trainer, inputs)
                        assert_trainer(trainer, inputs, condition, tau=tau)
                        for name, value in trainer.model.target_encoder.state_dict().items():
                            expected = (
                                before[name] * 0.996
                                + trainer.model.online_encoder.state_dict()[name] * 0.004
                            )
                            torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)
                        first_moments = {
                            name: float(trainer.optimizer.state[parameter]["exp_avg"].norm())
                            for name, parameter in trainer.model.named_parameters()
                            if parameter.requires_grad
                        }
                        if any(
                            not any(
                                value > 0
                                for name, value in first_moments.items()
                                if name.startswith(prefix)
                            )
                            for prefix in ("online_encoder.", "parameter_encoder.", "predictor.")
                        ):
                            raise RuntimeError(
                                "Discarded native update did not reach every trainable block"
                            )
                        preflight[condition] = {
                            "global_step": inputs.source_step + 1,
                            "metrics": metrics,
                            "nonzero_first_moments": first_moments,
                            "fresh_optimizer": True,
                            "ema_verified": True,
                            "geco_updates": int(trainer.geco.num_updates)
                            if trainer.geco is not None
                            else 0,
                            "discarded": True,
                        }
                        log(
                            {
                                f"preflight/{condition}/{name}": value
                                for name, value in metrics.items()
                            }
                        )
                write_json(out / "startup.json", {"identity": identity, "conditions": startup})
                write_json(out / "preflight.json", {"identity": identity, "conditions": preflight})
                verify_files(inputs.hashes)
                if model_fingerprint(inputs.model) != initial_hash:
                    raise RuntimeError("Preparation modified the source model")
                result = {
                    "identity": identity,
                    "admission": inputs.admission,
                    "tau": tau_record,
                    "startup": startup,
                    "preflight": preflight,
                    "tracking": logger.metadata,
                    "source_unchanged": True,
                }
                write_json(out / "results.json", result)
                log({"diagnostic/complete": 1.0})
                for name in ("protocol.json", "results.json", "tau.json"):
                    logger.log_artifact(
                        out / name, name=f"sparse-preparation-{identity[:12]}-{Path(name).stem}"
                    )
                names = (
                    "protocol.json",
                    "initial_model.pt",
                    "tau.json",
                    "startup.json",
                    "preflight.json",
                    "results.json",
                    "tracking.json",
                )
                write_json(
                    out / "completed.json",
                    {"identity": identity, "hashes": {name: sha(out / name) for name in names}},
                )
                logger.finish()
                return result
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Create the shared immutable calibration with online W&B by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    values = vars(parser.parse_args())
    resume = bool(values.pop("resume"))
    run(Config(**values), resume=resume)


if __name__ == "__main__":
    main()
