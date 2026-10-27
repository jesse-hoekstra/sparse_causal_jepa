import hashlib
import json
from pathlib import Path

import pytest

from diagnostics.joint_acquisition.backfill import verified_results


def test_backfill_verifies_saved_results_before_publishing(tmp_path: Path) -> None:
    tracking = {"id": "original-run", "mode": "online"}
    result = {"identity": "original-diagnostic", "tracking": tracking}
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(result))
    (tmp_path / "tracking.json").write_text(json.dumps(tracking))
    (tmp_path / "completed.json").write_text(
        json.dumps(
            {
                "identity": result["identity"],
                "outputs": {"results.json": hashlib.sha256(result_path.read_bytes()).hexdigest()},
            }
        )
    )
    assert verified_results(tmp_path) == result
    (tmp_path / "tracking.json").write_text(json.dumps({**tracking, "id": "wrong-run"}))
    with pytest.raises(ValueError, match="tracking identity changed"):
        verified_results(tmp_path)
    (tmp_path / "tracking.json").write_text(json.dumps(tracking))
    result_path.write_text("tampered")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        verified_results(tmp_path)
