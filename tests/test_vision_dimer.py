"""Behavioural tests for the vision experiment: DGP, camera, model, training stages, analyses."""

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.vision_dimer import analysis, dgp, downstream
from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as vtrain
from diagnostics.vision_dimer.model import CachedEncoder, assemble, build_model
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.lagrangian import SparsityLagrangian

PHYSICS = dgp.Physics()
CAMERA = dgp.Renderer()

# The configuration of the reported cohort (seed 20261115), as stored in its manifest.
REPORTED_COHORT: dict[str, Any] = {
    "seed": 20261115,
    "groups": {"train": 8192, "calibration": 512, "validation": 256, "test": 512},
    "views": 10, "context": 10, "forecast": 2, "episode_frames": 40,
    "physics": {
        "dt": 0.5, "field_scale": 0.6, "charge_to_mass": [[1.0, -0.6], [0.75, -0.4]],
        "trap_stiffness": 1.0, "bond_stiffness": 1.0, "site_separation": 1.0,
        "site_angle_degrees": 20.0, "medium_angle_degrees": 45.0,
        "dimer_centres": [[-1.3, 0.6], [1.3, -0.6]], "drag_range": [0.1, 1.2],
        "action_std": 1.3, "burn_in": 300, "initial_position_std": 0.5,
    },
    "renderer": {
        "size": 64, "sigma": 2.6, "span": 28.0, "scale": 2.0, "medium_angle_degrees": 45.0,
        "imaging": "rgb", "sigma_small": 1.8, "dyes": [[1.0, 0.55, 0.1], [0.1, 0.45, 1.0]],
    },
}  # fmt: skip


def _trajectory(groups: int = 8, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    theta = dgp.sample_theta(groups, PHYSICS, generator)
    states, actions = dgp.simulate(theta, 3, 12, PHYSICS, generator)
    return theta, states, actions


def _small_cohort(directory: Path, episode_frames: int = 12, **groups: int) -> vdata.DataConfig:
    sizes = {"train_groups": 8, "calibration_groups": 8, "validation_groups": 4, "test_groups": 4}
    sizes.update(groups)
    config = vdata.DataConfig(
        episode_frames=episode_frames,
        train_groups=sizes["train_groups"],
        calibration_groups=sizes["calibration_groups"],
        validation_groups=sizes["validation_groups"],
        test_groups=sizes["test_groups"],
    )
    vdata.prepare(directory, config)
    return config


def _same_tensors(first: dict[str, Any], second: dict[str, Any]) -> None:
    assert first.keys() == second.keys()
    for key, value in first.items():
        if isinstance(value, Tensor):
            assert torch.equal(value, second[key]), key


# ---------------------------------------------------------------------------- DGP and camera


def test_reported_cohort_settings_are_the_defaults() -> None:
    record = json.loads(json.dumps(vdata.DataConfig().as_record()))
    assert record == REPORTED_COHORT
    assert vdata.DataConfig.from_record(record) == vdata.DataConfig()


def test_recurrence_and_inverse_are_exact() -> None:
    theta, states, actions = _trajectory()
    s = dgp.unflatten_state(states)
    th = theta[:, None, None].expand(*s.shape[:2], s.shape[2] - 2, dgp.PARAM_DIM)
    forward = dgp.transition(s[:, :, :-2], s[:, :, 1:-1], actions[:, :, 1:], th, PHYSICS)
    assert torch.equal(forward, s[:, :, 2:])
    backward = dgp.previous_state(s[:, :, 2:], s[:, :, 1:-1], actions[:, :, 1:], th, PHYSICS)
    assert torch.allclose(backward, s[:, :, :-2], atol=1e-12)


def test_actions_are_float32_exact() -> None:
    _, _, actions = _trajectory()
    assert torch.equal(actions, actions.float().double())


def test_jacobian_supports_match_declared_graphs() -> None:
    generator = torch.Generator().manual_seed(3)
    previous = torch.randn(2, 2, 2, generator=generator, dtype=torch.float64)
    current = torch.randn(2, 2, 2, generator=generator, dtype=torch.float64)
    action = torch.randn(dgp.ACTION_DIM, generator=generator, dtype=torch.float64)
    theta = dgp.sample_theta(1, PHYSICS, generator)[0]

    def step(p: Tensor, q: Tensor, a: Tensor, t: Tensor) -> Tensor:
        return dgp.flatten_state(dgp.transition(p.view(2, 2, 2), q.view(2, 2, 2), a, t, PHYSICS))

    inputs = (previous.flatten(), current.flatten(), action, theta)
    jac = cast(tuple[Tensor, ...], torch.autograd.functional.jacobian(step, inputs))  # pyright: ignore[reportUnknownMemberType]
    temporal = (jac[0].abs() + jac[1].abs()) > 1e-12
    graphs = dgp.graphs()
    assert torch.equal(temporal, graphs["state"])
    assert torch.equal(jac[2].abs() > 1e-12, graphs["action"])
    assert torch.equal(jac[3].abs() > 1e-12, graphs["parameter"])


def test_physics_rejects_charge_ratios_that_break_temporal_influence() -> None:
    """Equal |beta| for a slot across dimers drops the temporal witness span (Assumption 13)."""
    with pytest.raises(ValueError, match="distinct magnitudes"):
        dgp.Physics(charge_to_mass=((1.0, -0.6), (0.75, 0.6)))
    with pytest.raises(ValueError, match="distinct magnitudes"):
        dgp.Physics(charge_to_mass=((1.0, 0.0), (0.75, -0.4)))
    assert dgp.Physics().charge_to_mass == ((1.0, -0.6), (0.75, -0.4))


def _bead_moved(frames: Tensor, bead: int, delta: float) -> Tensor:
    moved = frames.clone()
    moved[..., 2 * bead : 2 * bead + 2] += delta
    return moved


def test_camera_geometry_keeps_beads_in_frame_and_turns_the_medium_axes() -> None:
    _, states, _ = _trajectory(groups=2)
    frames = states[0, 0, :3]
    # A pure medium-frame u displacement moves both image columns and rows (45 degree camera).
    shifted = frames[:1].clone()
    shifted[:, 0] += 0.5
    delta = CAMERA.pixel_coordinates(shifted.float()) - CAMERA.pixel_coordinates(frames[:1].float())
    assert float(delta[0, 0].abs().min()) > 0.5
    pixels = CAMERA.pixel_coordinates(states.float())
    assert float(pixels.min()) > 0
    assert float(pixels.max()) < 63
    with pytest.raises(ValueError, match="narrower"):
        dgp.Renderer(sigma=1.8, sigma_small=2.0)
    with pytest.raises(ValueError, match="distinguishable"):
        dgp.Renderer(dyes=((1.0, 0.5, 0.1), (1.0, 0.5, 0.1)))
    with pytest.raises(ValueError, match="imaging"):
        dgp.Renderer(imaging="channels")


def test_rgb_camera_shows_dye_colour_and_true_size() -> None:
    _, states, _ = _trajectory(groups=2)
    frames = states[0, 0, :3]
    image = CAMERA(frames)
    assert CAMERA.shape == (3, 64, 64)
    assert image.shape == (3, 3, 64, 64)
    assert float(image.min()) >= 0.0
    for bead, dimer in ((0, 0), (1, 0), (2, 1), (3, 1)):
        change = CAMERA(_bead_moved(frames, bead, 0.7)) - image
        energy = change.square().sum((-1, -2))  # (frames, RGB)
        colour = torch.tensor(CAMERA.dyes[dimer]).square()
        # A bead's light reaches the sensor only through its dimer's dye response.
        ratio = energy / energy.sum(-1, keepdim=True)
        expected = (colour / colour.sum()).expand_as(ratio)
        assert torch.allclose(ratio, expected, atol=0.12)
    # Large beads are drawn wider than small beads: second moment of a bead at the image
    # centre, measured in a window that the others (pushed to the frame edge) cannot reach.
    for bead, width in ((0, CAMERA.sigma), (1, CAMERA.sigma_small)):
        beads = torch.full((4, 2), 40.0, dtype=torch.float64)
        beads[bead] = 0.0
        lone = dgp.flatten_state(beads.reshape(2, 2, 2))
        window = CAMERA(lone)[0, 23:41, 23:41]
        offsets = torch.arange(23, 41, dtype=torch.float32) - (64 - 1) / 2
        rows = window.sum(1)
        spread = (offsets.square() * rows).sum() / rows.sum()
        assert abs(float(spread.sqrt()) - width) < 0.1


def test_rgb_camera_is_injective_when_beads_overlap_exactly() -> None:
    from torch.autograd.functional import jacobian  # pyright: ignore[reportUnknownVariableType]

    _, states, _ = _trajectory(groups=2)
    base = dgp.unflatten_state(states[0, 0, 5]).flatten(-3, -2)  # (4, 2): 1a, 1b, 2a, 2b
    for first, second in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3)):
        beads = base.clone()
        beads[second] = beads[first]
        z = dgp.flatten_state(beads.reshape(2, 2, 2)).float()

        def image(s: Tensor) -> Tensor:
            return CAMERA(s).flatten()

        matrix = cast(Tensor, jacobian(image, z)).double()
        values = cast(Tensor, torch.linalg.svdvals(matrix))  # pyright: ignore[reportUnknownMemberType]
        assert int((values > values[0] * 1e-6).sum()) == dgp.STATE_DIM
        assert float(values[-1] / values[0]) > 0.05


# ---------------------------------------------------------------------------------- data


def test_inputs_are_label_free(tmp_path: Path) -> None:
    _small_cohort(tmp_path, train_groups=4, calibration_groups=4)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["splits"]) == {"train", "calibration", "validation"}
    inputs = vdata.RenderedInputs(tmp_path, "train", torch.device("cpu"))
    batch = inputs.batch(torch.arange(2))
    assert set(batch) == {"observations", "actions"}
    assert batch["observations"].shape == (2, 10, 12, 3, 64, 64)
    with pytest.raises(ValueError, match="TRAIN physical labels"):
        vdata.load_labels(tmp_path, "train")
    assert not (tmp_path / "test.pt").exists()


def test_split_verification_regenerates_stored_splits_and_detects_changes(tmp_path: Path) -> None:
    config = _small_cohort(tmp_path)
    report = vdata.verify_split(tmp_path, "validation")
    assert report["identical"]
    assert report["max_abs_state_difference"] == 0.0
    # The one-time test split is drawn under the stored configuration only.
    with pytest.raises(ValueError, match="different settings"):
        vdata.prepare(tmp_path, dataclasses.replace(config, seed=1), include_test=True)
    assert "test" in vdata.prepare(tmp_path, config, include_test=True)["splits"]
    tensors = torch.load(tmp_path / "validation.pt", weights_only=True)
    tensors["states"][0, 0, 0, 0] += 1.0
    torch.save(tensors, tmp_path / "validation.pt")
    with pytest.raises(ValueError, match="stored validation split changed"):
        vdata.verify_split(tmp_path, "validation")


def test_windows_match_manual_slices(tmp_path: Path) -> None:
    _small_cohort(
        tmp_path, episode_frames=17, train_groups=3, calibration_groups=2, validation_groups=2
    )
    inputs = vdata.RenderedInputs(tmp_path, "train", torch.device("cpu"))
    assert (inputs.episode_frames, inputs.frames) == (17, 12)
    starts = torch.tensor([[0, 5, 3] + [1] * 7, [2] * 10, [5, 4, 3, 2, 1, 0, 5, 4, 3, 2]])
    batch = inputs.batch(torch.arange(3), starts)
    full = inputs.episodes(torch.arange(3))
    for g in range(3):
        for r in range(10):
            s = int(starts[g, r])
            assert torch.equal(batch["observations"][g, r], full["observations"][g, r, s : s + 12])
            assert torch.equal(batch["actions"][g, r], full["actions"][g, r, s : s + 11])
    first = inputs.batch(torch.arange(3))
    assert torch.equal(first["observations"], full["observations"][:, :, :12])


# --------------------------------------------------------------------------------- model


def _cached_model(phase: str = "full") -> SCJEPA:
    model = build_model(0, phase=cast(Any, phase), code_dim=dgp.PARAM_DIM, context=10)
    return assemble(
        CachedEncoder(), cast(Any, model.parameter_encoder), cast(Any, model.predictor), 10
    )


def test_donor_codes_and_first_prediction_stay_attached() -> None:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    model = _cached_model()
    model.train()
    states = torch.randn(4, 3, 12, dgp.STATE_DIM, requires_grad=True)
    actions = torch.randn(4, 3, 11, dgp.ACTION_DIM)
    model.predictor.requires_grad_(False)  # frozen predictor must still pass gradients
    output = cast(ModelOutput, model(states, actions))
    assert output.predictions.shape == (4, 3, 2, dgp.STATE_DIM)
    loss = output.predictions[:, :, 1].sum()
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    grads = [p.grad for p in model.parameter_encoder.parameters()]
    assert all(g is not None for g in grads)
    assert sum(float(cast(Tensor, g).abs().sum()) for g in grads) > 0


def test_second_step_reads_first_prediction() -> None:
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    model = _cached_model("dense")
    model.eval()

    class Constant(torch.nn.Module):
        state_dim, action_dim, param_dim, max_history = 8, 5, 4, 10

        def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
            del actions
            return states.new_zeros(len(states), dgp.PARAM_DIM)

    model.parameter_encoder = cast(Any, Constant())
    states = torch.randn(2, 3, 12, dgp.STATE_DIM, requires_grad=True)
    actions = torch.randn(2, 3, 11, dgp.ACTION_DIM)
    second = cast(ModelOutput, model(states, actions)).predictions[:, :, 1].sum()
    (gradient,) = torch.autograd.grad(second, states)
    # Step 2 sees (z^9, zhat^10); z^8 can only reach it through the attached zhat^10.
    assert float(gradient[:, :, 8].abs().sum()) > 0


def test_geco_relaxes_when_prediction_exceeds_tolerance() -> None:
    geco = SparsityLagrangian(tau=1.0, step_size=0.1, lambda_init=1000.0, momentum=0.99)
    before = float(geco.penalty_weight)
    geco.update(torch.tensor(2.0))  # prediction / tau = 2 > 1
    assert float(geco.penalty_weight) < before
    geco = SparsityLagrangian(tau=1.0, step_size=0.1, lambda_init=1000.0, momentum=0.99)
    geco.update(torch.tensor(0.5))
    assert float(geco.penalty_weight) > before


def test_frozen_cache_windows_equal_direct_encoding(tmp_path: Path) -> None:
    _small_cohort(
        tmp_path, episode_frames=15, train_groups=2, calibration_groups=2, validation_groups=2
    )
    inputs = vdata.RenderedInputs(tmp_path, "train", torch.device("cpu"))
    model = build_model(0, phase="dense", code_dim=8, context=10)
    model.eval()
    cache = vtrain.FrozenCache(model, {"train": inputs})
    starts = torch.tensor([[3] * 10, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]])
    cached = cache.batch("train", torch.arange(2), starts)
    rendered = inputs.batch(torch.arange(2), starts)
    with torch.no_grad():
        flat = rendered["observations"].flatten(0, 1)
        online = model.online_encoder.encode_sequence(flat).reshape(2, 10, 12, dgp.STATE_DIM)
        target = model.target_encoder.encode_sequence(flat[:, 10:]).reshape(2, 10, 2, dgp.STATE_DIM)
    assert torch.allclose(cached["observations"], online, atol=1e-6)
    assert torch.allclose(cached["targets"], target, atol=1e-6)
    assert torch.equal(cached["actions"], rendered["actions"])


def test_frozen_cache_file_round_trip(tmp_path: Path) -> None:
    _small_cohort(
        tmp_path, episode_frames=14, train_groups=2, calibration_groups=2, validation_groups=2
    )
    inputs = {"train": vdata.RenderedInputs(tmp_path, "train", torch.device("cpu"))}
    model = build_model(0, phase="dense", code_dim=8, context=10)
    model.eval()
    path = tmp_path / "cache.pt"
    fresh = vtrain.FrozenCache(model, inputs, path)
    assert path.exists()
    loaded = vtrain.FrozenCache(model, inputs, path)
    assert torch.equal(fresh.online["train"], loaded.online["train"])
    assert torch.equal(fresh.target["train"], loaded.target["train"])
    starts = torch.ones(2, 10, dtype=torch.long)
    for key in ("observations", "actions", "targets"):
        assert torch.equal(
            fresh.batch("train", torch.arange(2), starts)[key],
            loaded.batch("train", torch.arange(2), starts)[key],
        )


# ------------------------------------------------------------------------------ training


def _joint(directory: Path, data: Path, steps: int, **overrides: int) -> vtrain.StageConfig:
    settings: dict[str, Any] = {
        "stage": "joint", "data_dir": str(data), "output_dir": str(directory), "device": "cpu",
        "batch_groups": 2, "code_dim": 8, "log_every": 1, "checkpoint_every": 2,
        "eval_every": 1000, "steps": steps, **overrides,
    }  # fmt: skip
    return vtrain.StageConfig(**settings)


def test_stage_defaults_are_the_reported_settings() -> None:
    config = vtrain.StageConfig(stage="full", data_dir="d", output_dir="o", parent="p", tau=0.1)
    assert (config.learning_rate, config.encoder_learning_rate, config.gate_learning_rate) == (
        1e-3, 1e-4, 1e-3
    )  # fmt: skip
    assert (config.sigreg_weight, config.initial_logit, config.batch_groups) == (0.05, 3.0, 16)
    assert config.tolerance() == 0.1
    with pytest.raises(ValueError, match="positive calibrated tau"):
        vtrain.StageConfig(stage="baseline", data_dir="d", output_dir="o", parent="p")


@pytest.mark.parametrize("episode_frames", [12, 16])
def test_joint_stage_resume_is_exact(tmp_path: Path, episode_frames: int) -> None:
    _small_cohort(tmp_path / "data", episode_frames=episode_frames)
    vtrain.train(_joint(tmp_path / "a", tmp_path / "data", 3))
    interrupted = _joint(tmp_path / "b", tmp_path / "data", 3)
    torch.manual_seed(123)  # pyright: ignore[reportUnknownMemberType]
    vtrain.train(dataclasses.replace(interrupted, steps=2))
    # Restore the full config and resume from the step-2 checkpoint.
    saved = vtrain.load_checkpoint(tmp_path / "b" / "last.pt")
    saved["config"]["steps"] = 3
    torch.save(saved, tmp_path / "b" / "last.pt")
    (tmp_path / "b" / "final.json").unlink()
    vtrain.train(interrupted, resume=True)
    _same_tensors(
        vtrain.load_checkpoint(tmp_path / "a" / "final.pt")["model"],
        vtrain.load_checkpoint(tmp_path / "b" / "final.pt")["model"],
    )


def _sparse_pair(tmp_path: Path) -> tuple[vtrain.StageConfig, Path]:
    """An uninterrupted 4-update full run and a 2-update run of the same condition."""
    _small_cohort(tmp_path / "data", episode_frames=16)
    vtrain.train(_joint(tmp_path / "parent", tmp_path / "data", 2, code_dim=4))
    settings: dict[str, Any] = {
        "stage": "full", "data_dir": str(tmp_path / "data"), "device": "cpu", "batch_groups": 2,
        "code_dim": 4, "log_every": 1, "checkpoint_every": 2, "eval_every": 1000, "tau": 0.05,
        "parent": str(tmp_path / "parent" / "final.pt"),
    }  # fmt: skip
    whole = vtrain.StageConfig(output_dir=str(tmp_path / "whole"), steps=4, **settings)
    vtrain.train(whole)
    vtrain.train(vtrain.StageConfig(output_dir=str(tmp_path / "short"), steps=2, **settings))
    return whole, tmp_path / "short" / "last.pt"


def _assert_continued_exactly(whole: Path, continued: Path) -> None:
    first, second = vtrain.load_checkpoint(whole), vtrain.load_checkpoint(continued)
    assert first["step"] == second["step"] == 4
    _same_tensors(first["model"], second["model"])
    for key in ("log_lambda", "ma_error"):
        assert torch.equal(first["geco"][key], second["geco"][key]), key
    assert torch.equal(first["cpu_rng"], second["cpu_rng"])


def test_fork_extends_the_budget_exactly(tmp_path: Path) -> None:
    whole, source = _sparse_pair(tmp_path)
    with pytest.raises(ValueError, match="may not change"):
        vtrain.fork(
            source,
            dataclasses.replace(whole, output_dir=str(tmp_path / "bad"), learning_rate=2e-3),
        )
    with pytest.raises(ValueError, match="may not change"):
        vtrain.fork(source, dataclasses.replace(whole, output_dir=str(tmp_path / "bad"), tau=0.1))
    with pytest.raises(FileExistsError, match="already holds a run"):
        vtrain.fork(source, dataclasses.replace(whole, output_dir=str(tmp_path / "short")))
    forked = dataclasses.replace(whole, output_dir=str(tmp_path / "forked"))
    (tmp_path / "forked").mkdir()
    (tmp_path / "forked" / "log.txt").write_text("launcher output\n")  # allowed
    record = vtrain.fork(source, forked)
    assert record["source_step"] == 2
    assert set(record["changed"]) == {"output_dir", "steps"}
    vtrain.train(forked, resume=True)
    _assert_continued_exactly(tmp_path / "whole" / "last.pt", tmp_path / "forked" / "last.pt")


def _as_development_checkpoint(saved: dict[str, Any], head: int = 6) -> dict[str, Any]:
    """The layout written during development: legacy config fields and an unused SIGReg head."""
    legacy = copy.deepcopy(saved)
    legacy["config"].update(
        tau_relax=1.0, tau_relax_steps=0, tau_anneal_steps=0, sigreg_target="raw"
    )
    legacy["head"] = {f"net.{k}": torch.zeros(2) for k in range(head)}
    optimizer = legacy["optimizer"]
    groups = optimizer["param_groups"]
    end = max(groups[1]["params"]) + 1  # the head closed the encoder group

    def shifted(index: int) -> int:
        return index + head if index >= end else index

    optimizer["state"] = {shifted(i): v for i, v in optimizer["state"].items()}
    for group in groups[2:]:
        group["params"] = [shifted(i) for i in group["params"]]
    groups[1]["params"] = [*groups[1]["params"], *range(end, end + head)]
    return legacy


def test_development_checkpoints_continue_exactly(tmp_path: Path) -> None:
    whole, source = _sparse_pair(tmp_path)
    saved = vtrain.load_checkpoint(source)
    legacy = _as_development_checkpoint(saved)
    upgraded = vtrain.upgrade_checkpoint(legacy)
    assert upgraded["config"] == saved["config"]
    assert "head" not in upgraded
    assert [len(g["params"]) for g in upgraded["optimizer"]["param_groups"]] == [
        len(g["params"]) for g in saved["optimizer"]["param_groups"]
    ]
    torch.save(legacy, tmp_path / "legacy.pt")
    forked = dataclasses.replace(whole, output_dir=str(tmp_path / "forked"))
    vtrain.fork(tmp_path / "legacy.pt", forked)
    vtrain.train(forked, resume=True)
    _assert_continued_exactly(tmp_path / "whole" / "last.pt", tmp_path / "forked" / "last.pt")
    trained = copy.deepcopy(legacy)
    trained["optimizer"]["state"][trained["optimizer"]["param_groups"][1]["params"][-1]] = {}
    with pytest.raises(ValueError, match="auxiliary head was trained"):
        vtrain.upgrade_checkpoint(trained)
    scheduled = copy.deepcopy(legacy)
    scheduled["config"]["tau_relax"] = 2.0
    with pytest.raises(ValueError, match="tau_relax"):
        vtrain.upgrade_checkpoint(scheduled)


def test_evaluate_checkpoint_scores_the_requested_split(tmp_path: Path) -> None:
    """Calibration-fitted scores on validation or (explicitly generated) test."""
    config = _small_cohort(tmp_path / "data", episode_frames=14)
    vtrain.train(_joint(tmp_path / "joint", tmp_path / "data", 1, code_dim=6))
    checkpoint = tmp_path / "joint" / "final.pt"
    with pytest.raises(FileNotFoundError):  # test is never generated implicitly
        vtrain.evaluate_checkpoint(
            checkpoint, tmp_path / "data", tmp_path / "t.json", "cpu", "test"
        )
    vdata.prepare(tmp_path / "data", config, include_test=True)
    valid = vtrain.evaluate_checkpoint(checkpoint, tmp_path / "data", tmp_path / "v.json", "cpu")
    test = vtrain.evaluate_checkpoint(
        checkpoint, tmp_path / "data", tmp_path / "t.json", "cpu", "test"
    )
    assert (valid["scored_split"], test["scored_split"]) == ("validation", "test")
    assert valid["calibration_prediction_mse"] == test["calibration_prediction_mse"]
    assert valid["prediction_mse"] != test["prediction_mse"]
    assert valid["checkpoint_sha256"] == test["checkpoint_sha256"]
    # The figure export recomputes the stored evaluation from the same collection.
    record = analysis.figure_data(checkpoint, tmp_path / "data", tmp_path / "t.json", "cpu")
    assert record["matches_stored"]
    assert record["max_metric_difference"] == 0.0
    pearson = torch.tensor(record["pearson"])
    assert torch.allclose(pearson.diagonal(), torch.tensor(test["per_coordinate_correlation"]))


# ---------------------------------------------------------------------------- evaluation


def _validation_like(groups: int = 64, seed: int = 5) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    theta = dgp.sample_theta(groups, PHYSICS, generator)
    states, _ = dgp.simulate(theta, 4, 12, PHYSICS, generator)
    return states, theta


def _orthogonal(size: int, seed: int) -> Tensor:
    gaussian = torch.randn(size, size, generator=torch.Generator().manual_seed(seed))
    q, _ = cast(tuple[Tensor, Tensor], torch.linalg.qr(gaussian.double()))  # pyright: ignore[reportUnknownMemberType]
    return q


def test_state_metrics_on_known_charts() -> None:
    from diagnostics.vision_dimer.evaluate import state_metrics

    cal, _ = _validation_like(seed=5)
    val, _ = _validation_like(seed=6)
    exact = state_metrics(cal, cal, val, val)
    assert exact["mcc"] > 0.999
    assert exact["block_purity_min"] > 0.999
    assert exact["state_r2_min"] > 0.999
    rotation = _orthogonal(4, 1)
    mix = cast(Tensor, torch.block_diag(rotation, rotation))  # pyright: ignore[reportUnknownMemberType]
    rotated = state_metrics(cal @ mix.T, cal, val @ mix.T, val)
    assert rotated["block_purity_min"] > 0.999  # within-block mixing keeps purity
    assert rotated["block_purity_knn_min"] > 0.9  # small-sample 4D kNN bias ~.05
    assert rotated["state_r2_min"] > 0.999  # and all information
    assert rotated["mcc"] < 0.97
    assert rotated["spearman_mcc"] < 0.97  # rank scores still see the rotation
    dense = _orthogonal(8, 2)
    across = state_metrics(cal @ dense.T, cal, val @ dense.T, val)
    assert across["block_purity_mean"] < 0.9
    assert across["block_purity_knn_mean"] < 0.7


def test_theorem_compatible_scores_accept_elementwise_monotone_charts() -> None:
    """A permuted, elementwise saturating chart is identified up to elementwise maps."""
    from diagnostics.vision_dimer.evaluate import state_metrics

    cal, _ = _validation_like(seed=5)
    val, _ = _validation_like(seed=6)
    centre, scale = cal.reshape(-1, 8).mean(0), 1.5 * cal.reshape(-1, 8).std(0)
    learned_to_true = torch.tensor([4, 0, 5, 1, 6, 2, 7, 3])

    def chart(z: Tensor) -> Tensor:  # centred, so no coordinate saturates into a flat map
        return torch.tanh((z - centre) / scale)[..., learned_to_true]

    metrics = state_metrics(chart(cal), cal, chart(val), val)
    assert metrics["spearman_mcc"] > 0.999
    assert min(metrics["per_coordinate_spearman"]) > 0.999
    assert abs(sum(metrics["per_coordinate_spearman"]) / 8 - metrics["spearman_mcc"]) < 1e-9
    assert metrics["coordinatewise_r2_monotone_min"] > 0.99
    assert metrics["block_purity_knn_min"] > 0.9  # small-sample 4D kNN bias
    assert metrics["block_purity_knn_mean"] > metrics["block_purity_mean"]
    assert [int(learned_to_true[c]) for c in metrics["spearman_assignment"]] == list(range(8))


def test_graph_shd_uses_the_state_metric_assignment_under_a_permuted_chart() -> None:
    """End to end: the stored ``assignment`` (true -> learned) aligns the gate graphs.

    The permutation crosses dimer blocks and differs from its inverse; reading the
    assignment in the wrong direction would give nonzero SHD and wrong correlations.
    """
    from diagnostics.vision_dimer.evaluate import graph_metrics, state_metrics

    learned_to_true = [4, 0, 5, 1, 6, 2, 7, 3]  # learned coordinate i is true coordinate [i]
    perm = torch.tensor(learned_to_true)
    cal, _ = _validation_like(seed=5)
    val, _ = _validation_like(seed=6)
    scales = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.7, 1.5, -0.4], dtype=cal.dtype)
    metrics = state_metrics(cal[..., perm] * scales, cal, val[..., perm] * scales, val)
    assert metrics["mcc"] > 0.999
    assert min(metrics["per_coordinate_correlation"]) > 0.999
    assert [learned_to_true[j] for j in metrics["assignment"]] == list(range(8))

    model = build_model(0, phase="full", code_dim=dgp.PARAM_DIM, context=10)
    predictor = cast(Any, model.predictor)
    code_parameters = [2, 0, 3, 1]  # learned code j encodes parameter code_parameters[j]
    true = dgp.graphs()
    with torch.no_grad():
        predictor.state_logits.copy_(10 * true["state"][perm][:, perm].float() - 5)
        predictor.action_logits.copy_(10 * true["action"][perm].float() - 5)
        parameter = true["parameter"][perm][:, torch.tensor(code_parameters)].float()
        predictor.parameter_logits.copy_(10 * parameter - 5)
    graphs = graph_metrics(model, metrics["assignment"], code_parameters)
    assert (graphs["state_shd"], graphs["action_shd"], graphs["parameter_shd"]) == (0, 0, 0)
    assert (graphs["state_edges"], graphs["action_edges"], graphs["parameter_edges"]) == (32, 24, 8)


def test_summary_contrasts_criteria_and_rotation_reference() -> None:
    from diagnostics.vision_dimer.summarize import rotation_reference, sign_flip_p, summarize

    def run(spearman: float, pearson: float, knn: float) -> dict[str, Any]:
        return {
            "mcc": pearson, "spearman_mcc": spearman, "block_purity_mean": knn - 0.03,
            "block_purity_knn_mean": knn, "state_r2_min": 0.995, "state_r2_cubic": [0.995] * 6,
            "parameter_r2_min": 0.97, "parameter_r2": [0.97] * 6,
            "coordinatewise_r2_monotone": [0.99] * 6, "coordinatewise_r2_monotone_min": 0.99,
            "nmse": 0.008, "code_shuffle_ratio": 30.0, "state_edges": 18, "action_edges": 36,
            "parameter_edges": 12, "state_shd": 0, "action_shd": 0, "parameter_shd": 0,
            "per_coordinate_spearman": [spearman] * 6,
        }  # fmt: skip

    runs: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in range(5):
        runs[(seed, "full")] = run(0.999 - 0.001 * seed, 0.978, 0.998)
        runs[(seed, "baseline")] = run(0.92 + 0.01 * seed, 0.91, 0.998)
    summary = summarize(runs)
    contrast = summary["contrasts"]["spearman_mcc"]
    assert contrast["positive"] == 5
    assert contrast["sign_flip_p_one_sided"] == 1 / 32
    assert summary["runs"]["seed0/full"]["criteria"]["spearman_mcc"]
    assert not summary["runs"]["seed0/full"]["criteria"]["pearson_mcc"]  # reported, not required
    assert summary["runs"]["seed4/baseline"]["criteria"]["below_full_spearman"]
    assert abs(summary["arms"]["full"]["spearman_mcc"]["mean"] - 0.997) < 1e-12
    assert sign_flip_p([1.0, -1.0]) == 0.75
    states, _ = _validation_like(groups=16, seed=3)
    reference = rotation_reference(states, draws=50)
    within = reference["within_block_spearman"]["mean"]
    assert reference["unrestricted_spearman"]["mean"] < within < 0.99
    assert abs(reference["within_block_pearson"]["mean"] - within) < 0.05


# ----------------------------------------------------------------------- paper analyses


def test_chart_transforms_are_orthogonal_and_rotate_each_bead() -> None:
    states, _ = _validation_like(groups=4)
    z = states.reshape(-1, dgp.STATE_DIM)
    assert torch.allclose(analysis.rotate_beads(z, 0.0), z)
    quarter = analysis.rotate_beads(z, 90.0).reshape(-1, 4, 2)
    beads = z.reshape(-1, 4, 2)
    assert torch.allclose(quarter[..., 0], -beads[..., 1], atol=1e-12)
    assert torch.allclose(quarter[..., 1], beads[..., 0], atol=1e-12)
    mixed = analysis.centre_separation(z)
    assert torch.allclose(mixed.square().sum(1), z.square().sum(1))
    # Dimer 1, u axis: centre (u1a + u1b) / sqrt 2 and separation (u1a - u1b) / sqrt 2.
    assert torch.allclose(mixed[:, 0], (z[:, 0] + z[:, 2]) / math.sqrt(2))
    assert torch.allclose(mixed[:, 2], (z[:, 0] - z[:, 2]) / math.sqrt(2))


def test_even_mixing_reference_matches_its_covariance_formula() -> None:
    pytest.importorskip("matplotlib")
    from diagnostics.vision_dimer.paper_figures import even_mixing_mcc

    states, _ = _validation_like(groups=32)
    z = states.reshape(-1, dgp.STATE_DIM)
    split: dict[str, Any] = {
        "state_std": cast(list[float], z.std(0).tolist()),  # pyright: ignore[reportUnknownMemberType]
        "correlation": cast(list[list[float]], torch.corrcoef(z.T).tolist()),  # pyright: ignore[reportUnknownMemberType]
    }
    manifest: dict[str, Any] = {"splits": {"x": split}}
    references = analysis.chart_references(z, CAMERA, draws=4)
    assert abs(references["medium_centre_separation_chart"] - even_mixing_mcc(manifest, "x")) < 1e-9
    assert references["within_block"]["p05"] <= references["within_block"]["mean"]
    assert len(references["rotation_curve"]) == 90
    assert references["rotation_curve"][0] > 0.999  # 0 degrees is the medium chart itself


def test_code_spread_separates_within_and_between_system_variance() -> None:
    generator = torch.Generator().manual_seed(0)
    means = torch.randn(400, 1, 3, generator=generator, dtype=torch.float64)
    constant = analysis.spread(means.expand(-1, 10, -1))
    assert constant["within_share"] < 1e-12
    assert constant["worst_direction_share"] < 1e-12
    noise = 0.5 * torch.randn(400, 10, 3, generator=generator, dtype=torch.float64)
    noisy = analysis.spread(means + noise)
    assert abs(noisy["within_share"] - 0.25 / 1.25) < 0.03
    assert abs(noisy["sd_ratio"] - 0.5) < 0.05
    assert noisy["worst_direction_share"] >= noisy["within_share_max_coordinate"] - 1e-9
    assert (noisy["groups"], noisy["contexts_per_group"], noisy["code_dim"]) == (400, 10, 3)


def test_curves_join_the_continuation_and_start_from_the_dense_initializer(
    tmp_path: Path,
) -> None:
    for s in analysis.SEEDS:
        seed = tmp_path / f"seed{s}"
        (seed / "adapt").mkdir(parents=True)
        (seed / "adapt" / "final.json").write_text(json.dumps({"mcc": 0.5, "nmse": 1.0}))
        for arm in ("baseline", "full"):
            for sub, steps in (("sparse-300k", (100, 300)), ("sparse-500k", (300, 500))):
                (seed / sub / arm).mkdir(parents=True)
                lines = [
                    json.dumps({"step": t, "mcc": t / 1000 + (sub == "sparse-500k")}) for t in steps
                ]
                (seed / sub / arm / "evaluations.jsonl").write_text("\n".join(lines) + "\n")
    curves = analysis.curves(tmp_path)
    rows = curves["seed3/full"]
    assert [r["step"] for r in rows] == [0, 100, 300, 500]
    assert [r["mcc"] for r in rows] == [0.5, 0.1, 0.3, 1.5]  # step 300 from the 300k run
    assert rows[0]["nmse"] == 1.0
    assert rows[1]["nmse"] is None


# ---------------------------------------------------------------------------- downstream


def test_downstream_truth_reproduces_the_simulator_and_holds_stay_within_a_dimer() -> None:
    theta, states, actions = _trajectory(groups=16)
    contexts = downstream.truth_contexts(
        {"states": states, "actions": actions, "theta": theta}, (0,), groups=16
    )
    truth = downstream.Truth(PHYSICS, contexts)
    free = truth.rollout({}, None)
    assert torch.equal(free, states[:, :, downstream.T + 2].flatten(0, 1))
    held = truth.rollout(truth.hold_value(0, 0.3), None)  # hold u1a at 0.3
    assert bool((held[:, 0] == 0.3).all())
    assert torch.allclose(held[:, 4:], free[:, 4:], atol=1e-12)  # dimer 2 is untouched
    assert not torch.allclose(held[:, 2:4], free[:, 2:4])  # its partner bead moves
    still = truth.rollout(truth.current(2), downstream.kicks(1)[0])  # bead 2a held where it is
    assert torch.equal(still[:, 4:6], contexts["cur"][:, 4:6])
    assert [int(k.nonzero()) for k in downstream.kicks(1)] == [3, 4]
    grid = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    values = torch.stack((2.0 * grid + 1.0, -0.5 * grid), 1)
    assert torch.allclose(downstream.slope(grid, values), torch.tensor([2.0, -0.5]).double())


def test_downstream_calibration_maps_follow_the_rank_direction() -> None:
    pytest.importorskip("scipy")
    x = torch.linspace(-2.0, 2.0, 41, dtype=torch.float64)
    increasing = downstream.Monotone(x, torch.tanh(x))
    assert torch.allclose(increasing(x), torch.tanh(x), atol=1e-12)
    decreasing = downstream.Monotone(x, -(x**3))
    assert torch.allclose(
        decreasing(torch.tensor([-2.0, 0.0, 2.0]).double()), torch.tensor([8.0, 0.0, -8.0]).double()
    )
    assert float(decreasing(torch.tensor([5.0]).double())) == -8.0  # constant extension
