"""Protect the physical-query support ordering and reproduce both native query interfaces."""

from pathlib import Path

import pytest
import torch

from diagnostics.vision_dimer import data, downstream
from diagnostics.vision_dimer.model import build_model

from . import kicks, system_effects
from .support_queries import recipients, responses


def test_external_donors_are_distinct_and_do_not_cross_windows_or_systems() -> None:
    """Every output keeps its recipient history/actions and excludes the recipient code."""
    starts = (0, 14, 28)
    groups, views, width = 4, 10, 2
    codes = torch.arange(len(starts) * groups * views * width, dtype=torch.float64).reshape(
        -1, width
    )
    panel = codes.reshape(len(starts), groups, views, width)
    state = codes + 0.125
    actions = codes + 0.375
    latent = {"code": codes, "online": state, "actions": actions}
    assert recipients(latent, "own", groups=groups, views=views, starts=starts) is latent
    for mode in ("single", "pair"):
        result = recipients(latent, mode, groups=groups, views=views, starts=starts)
        picked = result["code"].reshape_as(panel)
        assert result["online"] is state
        assert result["actions"] is actions
        for window in range(len(starts)):
            for system in range(groups):
                for index in range(views):
                    ref = panel[window, system, (index - 1) % views]
                    if mode == "pair":
                        ref = (ref + panel[window, system, (index - 2) % views]) / 2
                    torch.testing.assert_close(picked[window, system, index], ref)
        changed = codes.clone()
        changed[10] += 100000
        disturbed = recipients(
            latent | {"code": changed}, mode, groups=groups, views=views, starts=starts
        )["code"]
        affected = torch.nonzero((disturbed != result["code"]).any(-1)).flatten()
        torch.testing.assert_close(affected, torch.tensor([11] if mode == "single" else [11, 12]))
    with pytest.raises(ValueError, match="geometry"):
        recipients(latent, "pair", groups=groups, views=2, starts=starts)
    with pytest.raises(ValueError, match="rows"):
        recipients(latent, "single", groups=groups + 1, views=views, starts=starts)


@torch.inference_mode()
def test_rendered_cpu_own_single_and_pair_use_same_physical_query(tmp_path: Path) -> None:
    """Compare the native own physical slope/matrices and propagate both disjoint support modes."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = data.DataConfig(
            train_groups=5,
            calibration_groups=6,
            validation_groups=4,
            test_groups=2,
            episode_frames=40,
        )
        data.prepare(tmp_path, cfg)
        cpu = torch.device("cpu")
        c, v = (data.RenderedInputs(tmp_path, s, cpu) for s in ("calibration", "validation"))
        model = build_model(0, phase="full", code_dim=4, context=cfg.context).eval()
        f = downstream.encode(model, c, (0,), len(c))["online"].reshape(-1, 8)
        phys = (
            data.load_labels(tmp_path, "calibration")["states"][:, :, : cfg.frames]
            .reshape(-1, 8)
            .double()
        )
        grids = torch.quantile(phys, phys.new_tensor(downstream.QUANTILES), dim=0).T
        lat = downstream.encode(model, v, downstream.STARTS, len(v))
        reference = downstream.Model(model, lat, f, phys)
        scalar = downstream.experiments(reference, list(grids))
        previous = kicks.responses(reference)
        native_mse = None
        for mode in ("own", "single", "pair"):
            selected = recipients(lat, mode, groups=len(v), views=v.views, starts=downstream.STARTS)
            values, names, mse = responses(model, f, phys, selected, grids[4], len(v), v.views)
            assert names == reference.match
            assert mse.shape == (len(v),)
            for physical in values.values():
                assert physical.q1.shape == (len(v),)
                assert physical.q2.shape == (len(v), 4, 2, 2)
                assert torch.isfinite(physical.q2).all()
            if mode == "own":
                native_mse = mse
                torch.testing.assert_close(
                    values["scalar"].q1.mean(), phys.new_tensor(scalar["q1"]["u2a"]["slopes"][6])
                )
                for i, bead in enumerate(system_effects.BEADS):
                    torch.testing.assert_close(
                        values["scalar"].q2.mean(0)[i], previous[bead]["held"]
                    )
            else:
                assert native_mse is not None
                assert not torch.equal(native_mse, mse)
    finally:
        torch.set_num_threads(before)
