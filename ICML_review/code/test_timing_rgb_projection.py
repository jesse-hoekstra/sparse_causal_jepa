"""Check exact projection diagnostics on native first-window RGB frames without optimization."""

from pathlib import Path
from unittest.mock import patch

import torch

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train
from diagnostics.vision_dimer.model import build_model

from .timing_rgb_projection import replay
from .vision_acquisition import sha


def test_native_inverse_and_constant_code(tmp_path: Path) -> None:
    """Match the native spectrum for a varying code and record failure for a constant code."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = tmp_path / "data"
        vdata.prepare(
            data, vdata.DataConfig(train_groups=8, calibration_groups=2, validation_groups=2)
        )
        model = build_model(3, phase="dense", code_dim=8, context=10)
        meta = {"seed": 3, "code_dim": 8, "context": 10, "shape": [3, 64, 64], "stage": "acquire"}

        class TrainOnly(vdata.RenderedInputs):
            def __init__(self, directory: Path, split: str, device: torch.device) -> None:
                assert split == "train"
                super().__init__(directory, split, device)

        with (
            patch.object(vdata, "RenderedInputs", TrainOnly),
            patch.object(vdata, "load_labels", side_effect=AssertionError("unexpected labels")),
        ):
            for constant in (False, True):
                stage = tmp_path / str(constant)
                stage.mkdir()
                weights = model.state_dict()
                if constant:
                    weights["parameter_encoder.head.4.weight"].zero_()
                    weights["parameter_encoder.head.4.bias"].fill_(0.25)
                checkpoint = stage / "final.pt"
                torch.save({"meta": meta, "model": weights}, checkpoint)
                identity = sha(checkpoint)
                result = replay(checkpoint, data, tmp_path, "cpu")
                assert sha(checkpoint) == identity
                assert result["all_train_rows_identical"] == constant
                assert result["code_shape"] == [8, 10, 8]
                assert result["inverse_succeeded"] != constant
                assert not (tmp_path / "unused").exists()
                if constant:
                    assert result["coordinate_span"] == [0.0] * 8
                    assert result["projected_total"]["positive_resolved_rank"] == 0
                    assert result["total"]["numeric_matrix_rank"] == 0
                    assert "singular" in result["inverse_error"]
                else:
                    assert result["inverse_finite"]
                    assert result["projected_total"]["positive_resolved_rank"] == 4
                    official = train.project(stage, data, tmp_path / "official", device="cpu")
                    torch.testing.assert_close(
                        result["shared"]["eigenvalues"],
                        official["shared_eigenvalues"],
                        rtol=1e-8,
                        atol=1e-15,
                    )
    finally:
        torch.set_num_threads(threads)
