"""Staged label-free training for the vision experiment (the schedule of Experiment 1).

Commands (``python -m diagnostics.vision_dimer.train <command>``; one stage per call):

- ``train --stage joint``: online CNN, twice-parameter-dim context code and dense predictor;
  EMA targets.
- ``cache``: encode every split once with the joint stage's frozen encoders.
- ``train --stage acquire``: encoders frozen (cached encodings); fresh code encoder/predictor.
- ``select``: pick the acquisition with the lowest mean TRAIN prediction loss over its final
  5,000 updates (no labels).
- ``project``: fold the TRAIN shared-covariance projection to the parameter dimension.
- ``train --stage adapt``: encoders frozen; adapt the parameter-dim code encoder and predictor
  (the dense initializer).
- ``calibrate``: calibration prediction MSE of the dense initializer (``tau`` is a multiple).
- ``train --stage baseline`` / ``--stage full``: joint sparse training under GECO; baseline
  learns state and action gates, full additionally learns parameter gates.
- ``fork``: continue a sparse run exactly to a larger update budget.
- ``evaluate``: score a saved checkpoint on validation or (once, at the end) test.

The defaults of ``StageConfig`` are the reported settings; ``run_seed.sh`` lists every stage's
arguments. Only rendered frames, actions and group membership enter learning. Physical labels
are read by evaluation of calibration/validation (and test) only.
"""

import argparse
import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from diagnostics.masked_mlp_learned.acquisition import group_covariances, shared_modes
from diagnostics.shared.sampling import GroupSampler, resolve_device
from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import dgp
from diagnostics.vision_dimer import evaluate as veval
from diagnostics.vision_dimer.model import (
    CachedEncoder,
    LaggedCodeEncoder,
    LaggedMaskedPredictor,
    assemble,
    build_model,
    fresh_code_modules,
)
from scjepa.losses.regularization import RepresentationRegularizer
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.lagrangian import SparsityLagrangian

type Stage = Literal["joint", "acquire", "adapt", "baseline", "full"]
SCHEMA = "vision-dimer-stage-v1"
FROZEN: tuple[str, ...] = ("acquire", "adapt")
SPARSE: tuple[str, ...] = ("baseline", "full")


@dataclass(frozen=True)
class StageConfig:
    """Complete settings of one training stage; defaults are the reported vision settings.

    The image encoder trains at ``encoder_learning_rate``; the parameter encoder and
    predictor at ``learning_rate``; the gate logits at ``gate_learning_rate``. SIGReg
    regularizes the online states directly. Sparse stages use the constant GECO
    tolerance ``tau`` (twice the dense initializer's calibration MSE in the paper).
    """

    stage: Stage
    data_dir: str
    output_dir: str
    seed: int = 0
    steps: int = 100_000
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-4
    gate_learning_rate: float = 1e-3
    code_dim: int = 2 * dgp.PARAM_DIM
    parent: str | None = None
    acquisition_seed: int | None = None
    tau: float | None = None
    initial_logit: float = 3.0
    batch_groups: int = 16
    sigreg_weight: float = 0.05
    eval_every: int = 25_000
    log_every: int = 500
    checkpoint_every: int = 10_000
    device: str = "cuda"
    wandb_mode: str = "disabled"
    wandb_group: str = "vision-dimer"

    def __post_init__(self) -> None:
        """Reject inconsistent stage settings early."""
        if self.stage not in ("joint", "acquire", "adapt", "baseline", "full"):
            raise ValueError("unknown stage")
        if self.stage != "joint" and self.parent is None:
            raise ValueError("every stage after joint needs a parent checkpoint")
        if self.stage in SPARSE and (self.tau is None or not self.tau > 0):
            raise ValueError("sparse stages need a positive calibrated tau")
        if self.stage == "acquire" and self.acquisition_seed is None:
            raise ValueError("acquisition needs an explicit acquisition seed")

    @classmethod
    def from_saved(cls, record: dict[str, Any]) -> "StageConfig":
        """Rebuild a config stored in a checkpoint.

        Runs made during development also stored a tolerance schedule and a SIGReg
        target; they are accepted only at their neutral values (constant tolerance,
        SIGReg on the raw states), which is how every reported run was trained.
        """
        values = dict(record)
        if values.get("stage") in FROZEN:  # SIGReg does not act while the encoders are frozen
            values.pop("sigreg_target", None)
        for key, neutral in LEGACY_NEUTRAL.items():
            if key in values and values.pop(key) != neutral:
                raise ValueError(f"stored {key} differs from the reported setting {neutral!r}")
        if values.get("encoder_learning_rate") is None:
            values["encoder_learning_rate"] = values["learning_rate"]
        return cls(**values)

    def tolerance(self) -> float:
        """The constant GECO tolerance of a sparse stage."""
        if self.tau is None:
            raise ValueError("only sparse stages have a tolerance")
        return self.tau


# Development-era config fields and the values every reported run used.
LEGACY_NEUTRAL: dict[str, Any] = {
    "tau_relax": 1.0,
    "tau_relax_steps": 0,
    "tau_anneal_steps": 0,
    "sigreg_target": "raw",
}


def _sigreg_samples(states: Tensor) -> Tensor:
    """``(B, R, T, D) -> (R*T, B, D)``: each view/time is one group of B systems."""
    return states.flatten(1, 2).transpose(0, 1)


# --------------------------------------------------------------------------- data


class FrozenCache:
    """Frozen online and EMA encodings of complete episodes (frozen stages only).

    Windows are sliced from the cache exactly as ``RenderedInputs.batch`` slices
    frames, so cached and rendered training see identical windows.
    """

    def __init__(
        self,
        model: SCJEPA,
        inputs: dict[str, vdata.RenderedInputs],
        cache_file: Path | None = None,
    ) -> None:
        """Encode every frame of every episode once (or load an identical saved cache)."""
        self.online: dict[str, Tensor] = {}
        self.target: dict[str, Tensor] = {}
        self.actions: dict[str, Tensor] = {}
        self.first_target = max(model.context_len, cast(Any, model.predictor).history_len)
        self.frames = next(iter(inputs.values())).frames
        if cache_file is not None and cache_file.exists():
            device = next(model.parameters()).device
            saved = cast(dict[str, Any], torch.load(cache_file, map_location=device))
            for split, source in inputs.items():
                self.online[split] = saved["online"][split]
                self.target[split] = saved["target"][split]
                self.actions[split] = source.actions
            return
        with torch.no_grad():
            for split, source in inputs.items():
                online: list[Tensor] = []
                target: list[Tensor] = []
                for begin in range(0, len(source), 8):
                    indices = torch.arange(begin, min(begin + 8, len(source)))
                    x = source.episodes(indices)["observations"]
                    groups, views, length = x.shape[:3]
                    flat = x.flatten(0, 1)
                    online.append(
                        model.online_encoder.encode_sequence(flat).reshape(
                            groups, views, length, dgp.STATE_DIM
                        )
                    )
                    target.append(
                        model.target_encoder.encode_sequence(flat).reshape(
                            groups, views, length, dgp.STATE_DIM
                        )
                    )
                self.online[split] = torch.cat(online)
                self.target[split] = torch.cat(target)
                self.actions[split] = source.actions
        if cache_file is not None:
            _save(cache_file, {"online": self.online, "target": self.target})

    def batch(self, split: str, indices: Tensor, starts: Tensor | None = None) -> dict[str, Tensor]:
        """Cached window inputs and EMA targets of a group batch (default: first window)."""
        indices = indices.to(self.online[split].device)
        online = vdata.window(self.online[split][indices], starts, self.frames)
        target = vdata.window(self.target[split][indices], starts, self.frames)
        return {
            "observations": online,
            "actions": vdata.window(self.actions[split][indices], starts, self.frames - 1),
            "targets": target[:, :, self.first_target :],
        }

    def batches(self, split: str, size: int = 16) -> Iterator[dict[str, Tensor]]:
        """Iterate the whole split in order, first window of each episode."""
        total = len(self.online[split])
        for begin in range(0, total, size):
            yield self.batch(split, torch.arange(begin, min(begin + size, total)))


def frozen_cache_file(config: "StageConfig", data_dir: Path) -> Path | None:
    """Shared cache path keyed by the parent checkpoint and the data manifest hashes."""
    if config.parent is None:
        return None
    parent = Path(config.parent)
    key = vdata.sha256(parent)[:16] + "-" + vdata.sha256(data_dir / "manifest.json")[:12]
    return parent.parent / f"frozen-cache-{key}.pt"


def build_frozen_cache(parent: Path, data_dir: Path, device: str = "cuda") -> Path:
    """Precompute the frozen-encoder cache once, before launching parallel frozen stages."""
    dev = resolve_device(device)
    payload = load_checkpoint(parent)
    model = model_from_checkpoint(payload).to(dev)
    model.eval()
    inputs = {
        split: vdata.RenderedInputs(data_dir, split, dev)
        for split in ("train", "calibration", "validation")
    }
    config = StageConfig(stage="adapt", data_dir=str(data_dir), output_dir="-", parent=str(parent))
    path = cast(Path, frozen_cache_file(config, data_dir))
    FrozenCache(model, inputs, path)
    return path


def rendered_batches(source: vdata.RenderedInputs, size: int = 16) -> Iterator[dict[str, Tensor]]:
    """First-window batches of ``size`` groups, in order (the evaluation convention)."""
    for begin in range(0, len(source), size):
        yield source.batch(torch.arange(begin, min(begin + size, len(source))))


def replay_model(model: SCJEPA) -> SCJEPA:
    """Share the code encoder and predictor; read cached encodings instead of frames."""
    replay = assemble(
        CachedEncoder(),
        cast(LaggedCodeEncoder, model.parameter_encoder),
        cast(LaggedMaskedPredictor, model.predictor),
        model.context_len,
    )
    replay.train(model.training)
    return replay


# --------------------------------------------------------------------- initializers


@torch.no_grad()
def train_context_features(model: SCJEPA, states: Tensor, actions: Tensor) -> Tensor:
    """Penultimate code features ``(G, R, 128)`` of every TRAIN context (no labels)."""
    encoder = cast(LaggedCodeEncoder, model.parameter_encoder)
    context = model.context_len
    pieces: list[Tensor] = []
    for begin in range(0, len(states), 64):
        s = states[begin : begin + 64, :, :context]
        a = actions[begin : begin + 64, :, : context - 1]
        groups, views = s.shape[:2]
        feats = encoder.features(s.flatten(0, 1), a.flatten(0, 1))
        pieces.append(feats.reshape(groups, views, -1).double().cpu())
    return torch.cat(pieces)


@torch.no_grad()
def initialize_shared_output(model: SCJEPA, states: Tensor, actions: Tensor) -> dict[str, Any]:
    """Experiment 1's label-free shared-feature initialization of the code output layer."""
    features = train_context_features(model, states, actions)
    rows = features.flatten(0, 1)
    layer = cast(nn.Linear, cast(LaggedCodeEncoder, model.parameter_encoder).head[-1])
    weight0, bias0 = layer.weight.double().cpu(), layer.bias.double().cpu()
    original = rows @ weight0.T + bias0
    weight, values = shared_modes(features, layer.out_features)
    mean = rows.mean(0)
    proposed = (rows - mean) @ weight.T
    weight = (
        weight
        * original.var(0, correction=0).mean().sqrt()
        / proposed.var(0, correction=0).mean().sqrt()
    )
    layer.weight.copy_(weight.to(layer.weight))
    layer.bias.copy_((original.mean(0) - weight @ mean).to(layer.bias))
    return {"shared_generalized_eigenvalues": [float(v) for v in values]}


# --------------------------------------------------------------------- checkpoints


def _save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def upgrade_checkpoint(saved: dict[str, Any]) -> dict[str, Any]:
    """Bring a training checkpoint written during development to the current layout.

    Development runs also carried an auxiliary SIGReg head (``saved["head"]``), unused when
    SIGReg acts on the states as in every reported run. In stages that train the image
    encoder, its tensors closed the encoder's optimizer group without ever receiving a
    gradient; they are dropped, so continuing such a checkpoint is exact. The stored
    config is normalized by ``StageConfig.from_saved``.
    """
    if "head" not in saved:
        return saved
    upgraded = {key: value for key, value in saved.items() if key != "head"}
    upgraded["config"] = asdict(StageConfig.from_saved(saved["config"]))
    optimizer = saved.get("optimizer")
    if optimizer is not None and saved["config"]["stage"] not in FROZEN:
        head = len(saved["head"])
        groups = [dict(group) for group in optimizer["param_groups"]]
        dropped = groups[1]["params"][-head:]  # groups: continuous, encoder[, gates]
        if any(index in optimizer["state"] for index in dropped):
            raise ValueError("the auxiliary head was trained: not a reported setting")
        groups[1]["params"] = groups[1]["params"][:-head]
        upgraded["optimizer"] = {**optimizer, "param_groups": groups}
    return upgraded


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted local checkpoint dictionary."""
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def model_from_checkpoint(payload: dict[str, Any], *, phase: str = "dense") -> SCJEPA:
    """Rebuild a model (any condition) from a checkpoint's recorded settings."""
    meta = payload["meta"]
    model = build_model(
        meta["seed"],
        phase=cast(Any, phase),
        code_dim=meta["code_dim"],
        context=meta["context"],
        shape=tuple(meta["shape"]),
        initial_logit=meta.get("initial_logit", 3.0),
    )
    result = cast(Any, model.load_state_dict(payload["model"], strict=False))
    missing, unexpected = (
        cast(list[str], result.missing_keys),
        cast(list[str], result.unexpected_keys),
    )
    allowed = {f"predictor.{name}_logits" for name in ("state", "action", "parameter")}
    if unexpected and not set(unexpected) <= allowed:
        raise ValueError(f"unexpected checkpoint tensors: {unexpected}")
    if missing and not set(missing) <= allowed:
        raise ValueError(f"missing checkpoint tensors: {missing}")
    return model


# --------------------------------------------------------------------------- train


def train(config: StageConfig, *, resume: bool = False) -> dict[str, Any]:
    """Run one stage to its fixed update budget; return the final evaluation."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(config.device)
    output = Path(config.output_dir)
    if output.exists() and not resume and (output / "final.json").exists():
        raise FileExistsError(f"{output} already completed")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.data_dir)
    data_config = vdata.load_config(data_dir)
    shape = data_config.renderer.shape
    frozen = config.stage in FROZEN
    sparse = config.stage in SPARSE
    phase = config.stage if sparse else "dense"
    inputs = {
        split: vdata.RenderedInputs(data_dir, split, device)
        for split in ("train", "calibration", "validation")
    }
    # Evaluation always scores the first window of each episode.
    labels: dict[str, dict[str, Tensor]] = {}
    for split in ("calibration", "validation"):
        raw = vdata.load_labels(data_dir, split)
        labels[split] = {"states": raw["states"][:, :, : data_config.frames], "theta": raw["theta"]}

    # ---- model
    parent = load_checkpoint(config.parent) if config.parent else None
    if config.stage == "joint":
        model = build_model(config.seed, phase="dense", code_dim=config.code_dim,
                            context=data_config.context, shape=shape)  # fmt: skip
    else:
        assert parent is not None
        model = model_from_checkpoint(parent, phase=phase)
        if model.parameter_encoder.param_dim != config.code_dim:  # type: ignore[attr-defined]
            raise ValueError("parent code dimension differs from the stage setting")
        if sparse:
            with torch.no_grad():
                for logits in (
                    cast(LaggedMaskedPredictor, model.predictor).learned_logits().values()
                ):
                    logits.fill_(config.initial_logit)
    if config.stage == "acquire":
        assert config.acquisition_seed is not None
        code_encoder, predictor = fresh_code_modules(
            config.acquisition_seed, config.code_dim, data_config.context
        )
        model.parameter_encoder = cast(Any, code_encoder)
        model.predictor = cast(Any, predictor)
    model.to(device)
    model.online_encoder.requires_grad_(not frozen)

    cache = FrozenCache(model, inputs, frozen_cache_file(config, data_dir)) if frozen else None
    runner = replay_model(model) if frozen else model
    regularizer = RepresentationRegularizer("sigreg", 256).to(device)
    meta = {
        "schema": SCHEMA, "stage": config.stage, "seed": config.seed, "code_dim": config.code_dim,
        "episode_frames": data_config.episode_frames,
        "context": data_config.context, "shape": list(shape), "initial_logit": config.initial_logit,
    }  # fmt: skip

    # ---- shared-feature initialization of fresh code heads (label-free, TRAIN only)
    init_record: dict[str, Any] = {}
    if config.stage in ("joint", "acquire") and not resume:
        if cache is not None:
            states = cache.online["train"]
        else:
            with torch.no_grad():
                states = torch.cat(
                    [
                        model.online_encoder.encode_sequence(
                            b["observations"][:, :, : data_config.context].flatten(0, 1)
                        ).reshape(
                            len(b["observations"]),
                            data_config.views,
                            data_config.context,
                            dgp.STATE_DIM,
                        )
                        for b in rendered_batches(inputs["train"])
                    ]
                )
        init_record = initialize_shared_output(model, states, inputs["train"].actions)

    # ---- optimizer
    gate_parameters = list(cast(LaggedMaskedPredictor, model.predictor).learned_logits().values())
    gate_ids = {id(p) for p in gate_parameters}
    # The image encoder uses its own rate; the code encoder and predictor use ``learning_rate``.
    encoder_ids = {id(p) for p in model.online_encoder.parameters()}
    continuous = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in gate_ids and id(p) not in encoder_ids
    ]  # fmt: skip
    encoder_parameters = [p for p in model.online_encoder.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = [{"params": continuous, "lr": config.learning_rate}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": config.encoder_learning_rate})
    if gate_parameters:
        groups.append({"params": gate_parameters, "lr": config.gate_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    geco = (
        # Paper convention: normalized violation e = L_pred / tau - 1 (controller tau = 1).
        SparsityLagrangian(tau=1.0, step_size=0.1, lambda_init=1000.0, momentum=0.99).to(device)
        if sparse
        else None
    )
    sampler = GroupSampler(
        len(inputs["train"]),
        config.batch_groups,
        config.seed + 7919 * (config.acquisition_seed or 0),
    )
    torch.manual_seed(config.seed + 31 * (config.acquisition_seed or 0))  # pyright: ignore[reportUnknownMemberType]
    # Stationary episodes longer than one window: every update draws an independent
    # window start per (system, view); evaluation always uses the first window.
    spare_frames = inputs["train"].episode_frames - inputs["train"].frames
    windows = torch.Generator().manual_seed(
        config.seed + 104_651 * (config.acquisition_seed or 0) + 17
    )
    start = 0
    history_path = output / "history.jsonl"
    if resume and (output / "last.pt").exists():
        saved = upgrade_checkpoint(load_checkpoint(output / "last.pt"))
        if saved["config"] != asdict(config):
            raise ValueError("resume configuration differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load_state_dict(saved["sampler"])
        windows.set_state(saved["windows"])
        if geco is not None:
            geco.load_state_dict(saved["geco"])
        torch.set_rng_state(saved["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        start = int(saved["step"])
        if cache is not None:  # cached encoders are frozen copies of the parent; rebuild
            cache = FrozenCache(model, inputs, frozen_cache_file(config, data_dir))
            runner = replay_model(model)
        lines = history_path.read_text().splitlines() if history_path.exists() else []
        kept = [line for line in lines if json.loads(line)["step"] <= start]
        history_path.write_text("".join(line + "\n" for line in kept))
    (output / "config.json").write_text(
        json.dumps({"config": asdict(config), "meta": meta, "data": data_config.as_record(),
                    "initialization": init_record}, indent=2) + "\n"
    )  # fmt: skip

    wandb_run = None
    if config.wandb_mode != "disabled":
        import wandb

        run_name = f"{output.parent.name}-{output.name}"
        run_id = "".join(
            c if c.isalnum() or c in "-_" else "-" for c in f"{config.wandb_group}-{run_name}"
        )
        wandb_run = wandb.init(
            project="sparse_causal_jepa",
            group=config.wandb_group, name=run_name, mode=cast(Any, config.wandb_mode),
            config={"config": asdict(config), "meta": meta}, dir=str(output),
            resume="allow", id=run_id[:120],
        )  # fmt: skip

    def save(step: int) -> None:
        payload = {
            "config": asdict(config), "meta": meta, "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
            "windows": windows.get_state(),
            "geco": geco.state_dict() if geco is not None else None,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }  # fmt: skip
        _save(output / "last.pt", payload)

    def evaluation(step: int) -> dict[str, Any]:
        if cache is not None:
            cal = veval.collect(runner, cache.batches("calibration"))
            val = veval.collect(runner, cache.batches("validation"))
            shuffled = veval.collect(runner, cache.batches("validation"), shuffle=True)
        else:
            cal = veval.collect(model, rendered_batches(inputs["calibration"]))
            val = veval.collect(model, rendered_batches(inputs["validation"]))
            shuffled = veval.collect(model, rendered_batches(inputs["validation"]), shuffle=True)
        result = veval.evaluate(
            model, cal, labels["calibration"], val, labels["validation"], shuffled
        )
        result["calibration_prediction_mse"] = veval.prediction_mse(cal)
        result["step"] = step
        with (output / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        if wandb_run is not None:
            wandb_run.log(
                {f"val/{k}": v for k, v in result.items() if isinstance(v, int | float)}, step=step
            )
        return result

    model.train()
    totals: dict[str, float] = {}
    count = 0
    clock = time.time()
    for step in range(start + 1, config.steps + 1):
        indices = sampler.next()
        starts = (
            torch.randint(0, spare_frames + 1, (len(indices), data_config.views), generator=windows)
            if spare_frames
            else None
        )
        if cache is not None:
            batch = cache.batch("train", indices, starts)
            out = cast(ModelOutput, runner(batch["observations"], batch["actions"]))
            out = out._replace(targets=batch["targets"])
        else:
            batch = inputs["train"].batch(indices, starts)
            out = cast(ModelOutput, model(batch["observations"], batch["actions"]))
        prediction = (out.predictions - out.targets).square().mean()
        loss = prediction
        terms = {
            "prediction": prediction.detach(),
            "state_std_min": out.online_states.detach().reshape(-1, dgp.STATE_DIM).std(0).min(),
            "state_std_max": out.online_states.detach().reshape(-1, dgp.STATE_DIM).std(0).max(),
        }
        if not frozen and config.sigreg_weight:
            sigreg = regularizer(_sigreg_samples(out.online_states))
            loss = loss + config.sigreg_weight * sigreg
            terms["sigreg"] = sigreg.detach()
        if sparse:
            assert geco is not None
            penalties = {
                name: torch.stack([getattr(d.sparsity, name) for d in out.dynamics]).mean()
                for name in ("state", "action", "parameter")
            }
            sparsity = penalties["state"] + penalties["action"]
            if config.stage == "full":
                sparsity = sparsity + penalties["parameter"]
            loss = loss + geco.penalty_weight * sparsity
            terms.update({f"open_{k}": v.detach() for k, v in penalties.items()})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError(f"nonfinite gradient at step {step}")
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        if not frozen:
            model.update_target()
        if geco is not None:
            geco.update(prediction.detach() / config.tolerance())
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if step % config.log_every == 0 or step == config.steps:
            record = {"step": step, **{k: v / count for k, v in totals.items()},
                      "seconds_per_step": (time.time() - clock) / count}  # fmt: skip
            if geco is not None:
                record["log_lambda"] = float(geco.log_lambda)
                record["tolerance"] = config.tolerance()
            with history_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in record.items() if k != "step"}, step=step
                )
            print(json.dumps(record), flush=True)
            totals, count, clock = {}, 0, time.time()
        if step % config.checkpoint_every == 0 or step == config.steps:
            save(step)
        if step % config.eval_every == 0 and step != config.steps:
            print(json.dumps({"eval": evaluation(step)}), flush=True)
    final = evaluation(config.steps)
    final_payload = {
        "meta": meta,
        "config": asdict(config),
        "model": model.state_dict(),
    }
    _save(output / "final.pt", final_payload)
    (output / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.finish()
    return final


# ------------------------------------------------------------------ select/project


def select(runs: list[Path], output: Path, window: int = 5000) -> dict[str, Any]:
    """Pick the acquisition with the lowest mean logged TRAIN prediction over its last window."""
    scores: dict[str, float] = {}
    for run in runs:
        rows = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        last = rows[-1]["step"]
        tail = [r["prediction"] for r in rows if r["step"] > last - window]
        scores[str(run)] = sum(tail) / len(tail)
    best = min(scores, key=scores.__getitem__)
    record = {"selected": best, "scores": scores, "window": window, "labels_used": False}
    output.write_text(json.dumps(record, indent=2) + "\n")
    return record


@torch.no_grad()
def project(
    source: Path,
    data_dir: Path,
    output: Path,
    dimension: int = dgp.PARAM_DIM,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fold the TRAIN shared-covariance projection and OLS reconstruction into the weights."""
    dev = resolve_device(device)
    payload = load_checkpoint(source / "final.pt")
    model = model_from_checkpoint(payload).to(dev)
    model.eval()
    inputs = {"train": vdata.RenderedInputs(data_dir, "train", dev)}
    cache = FrozenCache(model, inputs)
    encoder = cast(LaggedCodeEncoder, model.parameter_encoder)
    context = model.context_len
    codes: list[Tensor] = []
    for begin in range(0, len(cache.online["train"]), 64):
        s = cache.online["train"][begin : begin + 64, :, :context]
        a = cache.actions["train"][begin : begin + 64, :, : context - 1]
        g, r = s.shape[:2]
        codes.append(encoder(s.flatten(0, 1), a.flatten(0, 1)).reshape(g, r, -1).double().cpu())
    codes_t = torch.cat(codes)
    rows = codes_t.flatten(0, 1)
    mean = rows.mean(0)
    shared, total = group_covariances(codes_t)
    values, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(shared))  # pyright: ignore[reportUnknownMemberType]
    selected = vectors[:, -dimension:].clone()
    for column in range(dimension):
        if bool(selected[selected[:, column].abs().argmax(), column] < 0):
            selected[:, column].neg_()
    projection = selected.T
    inverse = cast(Tensor, torch.linalg.inv(projection @ total @ projection.T))  # pyright: ignore[reportUnknownMemberType]
    reconstruction = total @ projection.T @ inverse
    width = codes_t.shape[-1]
    new = build_model(payload["meta"]["seed"], phase="dense", code_dim=dimension, context=context,
                      shape=tuple(payload["meta"]["shape"]))  # fmt: skip
    state = {k: v.clone() if isinstance(v, Tensor) else v for k, v in model.state_dict().items()}
    w, b = state["parameter_encoder.head.4.weight"], state["parameter_encoder.head.4.bias"]
    state["parameter_encoder.head.4.weight"] = (projection @ w.double().cpu()).to(w)
    state["parameter_encoder.head.4.bias"] = (projection @ (b.double().cpu() - mean)).to(b)
    first, first_bias = state["predictor.first.weight"], state["predictor.first.bias"]
    prefix = first.shape[1] - width
    code_columns = first[:, prefix:].double().cpu()
    state["predictor.first.weight"] = torch.cat(
        (first[:, :prefix], (code_columns @ reconstruction).to(first)), 1
    )
    state["predictor.first.bias"] = (first_bias.double().cpu() + code_columns @ mean).to(first_bias)
    new.load_state_dict(state, strict=True)
    projected = (rows - mean) @ projection.T
    residual = rows - mean - projected @ reconstruction.T
    meta = dict(payload["meta"], code_dim=dimension, stage="project")
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "final.pt", {"meta": meta, "model": new.state_dict()})
    record = {
        "source": str(source), "shared_eigenvalues": [float(v) for v in values],
        "selected_shared_fraction": float(values[-dimension:].sum() / values.clamp_min(0).sum()),
        "reconstructed_fraction": float(1 - residual.square().sum() / (rows - mean).square().sum()),
        "labels_used": False,
    }  # fmt: skip
    (output / "final.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


@torch.no_grad()
def calibrate(source: Path, data_dir: Path, output: Path, device: str = "cuda") -> dict[str, Any]:
    """Tau = deterministic calibration prediction MSE of the adapted dense model."""
    dev = resolve_device(device)
    payload = load_checkpoint(source / "final.pt")
    model = model_from_checkpoint(payload).to(dev)
    cal = vdata.RenderedInputs(data_dir, "calibration", dev)
    samples = veval.collect(model, rendered_batches(cal))
    record = {"tau": veval.prediction_mse(samples), "source": str(source), "groups": len(cal)}
    output.write_text(json.dumps(record, indent=2) + "\n")
    return record


# ------------------------------------------------------------------- fork/evaluate

FORKABLE = frozenset({"output_dir", "steps", "wandb_group", "wandb_mode"})


def fork(source: Path, config: StageConfig) -> dict[str, Any]:
    """Prepare ``config.output_dir`` to continue a sparse checkpoint to a larger budget.

    The continuation is equivalent to having run ``config`` from the start: only the
    output location, budget and W&B bookkeeping may differ. Model, optimizer, sampler,
    window, GECO and RNG states carry over unchanged; the caller then resumes with
    ``train(config, resume=True)``. The reported 500,000-update endpoints were made
    this way from 300,000-update runs.
    """
    saved = upgrade_checkpoint(load_checkpoint(source))
    original = StageConfig.from_saved(saved["config"])
    step = int(saved["step"])
    if config.stage not in SPARSE or original.stage != config.stage:
        raise ValueError("a fork continues a sparse stage in the same condition")
    old, new = asdict(original), asdict(config)
    differing = {key for key in new if old[key] != new[key]}
    if not differing <= FORKABLE:
        raise ValueError(f"a fork may not change {sorted(differing - FORKABLE)}")
    if config.steps < step:
        raise ValueError("the fork budget is shorter than the source progress")
    output = Path(config.output_dir)
    # Launcher files (log.txt, exit_code) may already exist; run artifacts may not.
    artifacts = ("last.pt", "final.pt", "history.jsonl", "evaluations.jsonl", "fork.json")
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError(f"{output} already holds a run")
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    saved["config"] = new
    _save(output / "last.pt", saved)
    history = source.parent / "history.jsonl"
    if history.exists():
        kept = [
            line for line in history.read_text().splitlines() if json.loads(line)["step"] <= step
        ]
        (output / "history.jsonl").write_text("".join(line + "\n" for line in kept))
    record = {
        "source": str(source), "source_sha256": digest, "source_step": step,
        "changed": {key: [old[key], new[key]] for key in sorted(differing)},
    }  # fmt: skip
    (output / "fork.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def evaluate_checkpoint(
    checkpoint: Path, data_dir: Path, output: Path, device: str = "cuda", split: str = "validation"
) -> dict[str, Any]:
    """Score a saved checkpoint with the current evaluation.

    Every alignment and probe is fitted on calibration; ``split`` (validation during
    development, test once at the end) is only scored. PyTorch's default precision settings
    apply (cuDNN TF32 convolutions, as in the reported test evaluation); training disables TF32.
    """
    if split not in ("validation", "test"):
        raise ValueError("the scored split is validation or test")
    dev = resolve_device(device)
    payload = load_checkpoint(checkpoint)
    stage = payload.get("config", {}).get("stage", payload["meta"].get("stage"))
    model = model_from_checkpoint(payload, phase=stage if stage in SPARSE else "dense").to(dev)
    data_config = vdata.load_config(data_dir)
    samples: dict[str, dict[str, Tensor]] = {}
    labels: dict[str, dict[str, Tensor]] = {}
    with torch.no_grad():
        for name in ("calibration", split):
            inputs = vdata.RenderedInputs(data_dir, name, dev)
            samples[name] = veval.collect(model, rendered_batches(inputs))
            raw = vdata.load_labels(data_dir, name)
            labels[name] = {
                "states": raw["states"][:, :, : data_config.frames], "theta": raw["theta"]
            }  # fmt: skip
        scored = vdata.RenderedInputs(data_dir, split, dev)
        shuffled = veval.collect(model, rendered_batches(scored), shuffle=True)
    result = veval.evaluate(
        model, samples["calibration"], labels["calibration"], samples[split], labels[split],
        shuffled,
    )  # fmt: skip
    result["calibration_prediction_mse"] = veval.prediction_mse(samples["calibration"])
    result["scored_split"] = split
    result["step"] = payload.get("step")
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fields = (
        ("stage", str), ("data_dir", str), ("output_dir", str), ("seed", int), ("steps", int),
        ("learning_rate", float), ("encoder_learning_rate", float),
        ("gate_learning_rate", float), ("code_dim", int),
        ("parent", str), ("acquisition_seed", int), ("tau", float), ("initial_logit", float),
        ("batch_groups", int), ("sigreg_weight", float),
        ("eval_every", int), ("log_every", int),
        ("checkpoint_every", int), ("device", str), ("wandb_mode", str), ("wandb_group", str),
    )  # fmt: skip
    run = commands.add_parser("train")
    branch = commands.add_parser("fork", help="continue a sparse checkpoint to a larger budget")
    branch.add_argument("--source", type=Path, required=True)
    for field_name, field_type in fields:
        run.add_argument(f"--{field_name.replace('_', '-')}", type=field_type)
        branch.add_argument(f"--{field_name.replace('_', '-')}", type=field_type)
    run.add_argument("--resume", action="store_true")
    scored = commands.add_parser("evaluate", help="score a saved checkpoint")
    scored.add_argument("--checkpoint", type=Path, required=True)
    scored.add_argument("--data-dir", type=Path, required=True)
    scored.add_argument("--output", type=Path, required=True)
    scored.add_argument("--device", default="cuda")
    scored.add_argument("--split", choices=("validation", "test"), default="validation")
    sel = commands.add_parser("select")
    sel.add_argument("--runs", nargs="+", type=Path, required=True)
    sel.add_argument("--output", type=Path, required=True)
    proj = commands.add_parser("project")
    proj.add_argument("--source", type=Path, required=True)
    proj.add_argument("--data-dir", type=Path, required=True)
    proj.add_argument("--output", type=Path, required=True)
    proj.add_argument("--device", default="cuda")
    cache_cmd = commands.add_parser("cache")
    cache_cmd.add_argument("--parent", type=Path, required=True)
    cache_cmd.add_argument("--data-dir", type=Path, required=True)
    cache_cmd.add_argument("--device", default="cuda")
    cal = commands.add_parser("calibrate")
    cal.add_argument("--source", type=Path, required=True)
    cal.add_argument("--data-dir", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "train":
        values = {
            k: v for k, v in vars(args).items() if k not in {"command", "resume"} and v is not None
        }
        print(json.dumps(train(StageConfig(**values), resume=args.resume)), flush=True)
    elif args.command == "fork":
        updates = {
            k: v for k, v in vars(args).items() if k not in {"command", "source"} and v is not None
        }
        saved = StageConfig.from_saved(load_checkpoint(args.source)["config"])
        config = StageConfig(**{**asdict(saved), **updates})
        receipt = Path(config.output_dir) / "fork.json"
        if receipt.exists():  # restart of an interrupted fork: resume, never re-fork
            if json.loads(receipt.read_text())["source"] != str(args.source):
                raise ValueError("existing fork has a different source")
        else:
            print(json.dumps(fork(args.source, config)), flush=True)
        print(json.dumps(train(config, resume=True)), flush=True)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            args.checkpoint, args.data_dir, args.output, args.device, args.split
        )
        print(json.dumps(result))
    elif args.command == "select":
        print(json.dumps(select(args.runs, args.output)))
    elif args.command == "cache":
        print(
            json.dumps({"cache": str(build_frozen_cache(args.parent, args.data_dir, args.device))})
        )
    elif args.command == "project":
        print(json.dumps(project(args.source, args.data_dir, args.output, device=args.device)))
    else:
        print(json.dumps(calibrate(args.source, args.data_dir, args.output, device=args.device)))


if __name__ == "__main__":
    main()
