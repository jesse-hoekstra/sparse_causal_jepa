import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "diagnostics/masked_mlp_learned/acquisition_batch.sh"

# Exercise the real shell scheduler with process-backed tmux/SSH substitutes.
# Workers cannot finish until all forty new attempts have started, so a silently
# serial scheduler fails this test instead of passing through instant fake jobs.
DRIVER = r"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ["BATCH_TEST_ROOT"])
name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "hostname":
    print("omi-rapid-22")
elif name == "nvidia-smi":
    print("0, 0")
elif name == "sleep":
    time.sleep(0.025)
elif name == "ssh":
    os.execv("/bin/bash", ["bash", "-c", args[-1]])
elif name == "tmux":
    if args[0] == "new-session":
        session = args[args.index("-s") + 1]
        child = subprocess.Popen(
            ["bash", "-c", args[-1]], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        (root / "sessions" / session).write_text(str(child.pid))
    elif args[0] == "kill-session":
        session = args[args.index("-t") + 1]
        with (root / "killed.txt").open("a") as stream:
            stream.write(session + "\n")
        try:
            os.killpg(int((root / "sessions" / session).read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        raise RuntimeError(args)
elif name == "python":
    if "train" in args:
        output = Path(args[args.index("--output-dir") + 1])
        output.mkdir(parents=True)
        assert args[args.index("--data-dir") + 1] == str(root / "custom data")
        seed = int(args[args.index("--acquisition-seed") + 1])
        (root / "started" / str(seed)).write_text(json.dumps(args))
        deadline = time.monotonic() + 20
        while len(list((root / "started").iterdir())) != 40:
            if time.monotonic() > deadline:
                raise RuntimeError("All eight waves were not launched concurrently")
            time.sleep(0.01)
        if os.environ["BATCH_TEST_FAILURE"] == "1":
            if seed == 1802:
                raise SystemExit(7)
            time.sleep(25)
            raise RuntimeError("Coordinator failed to stop a remaining worker")
        (output / "completion.json").write_text("{}\n")
    elif "diagnostics.masked_mlp_learned.selection" in args:
        candidates = args[args.index("--runs") + 1:args.index("--output")]
        assert len(candidates) == 16
        assert all((Path(p) / "completion.json").is_file() for p in candidates)
        output = Path(args[args.index("--output") + 1])
        output.write_text(json.dumps({"candidates": candidates, "admitted": True}))
    else:
        raise RuntimeError(args)
else:
    raise RuntimeError(name)
"""


@pytest.mark.parametrize("failure", [False, True])
def test_parallel_extension_preserves_prior_attempts_and_stops_owned_jobs(
    tmp_path: Path, failure: bool
) -> None:
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("sessions", "started"):
        (tmp_path / name).mkdir()
    driver = tools / "driver.py"
    driver.write_text(f"#!{sys.executable}\n" + textwrap.dedent(DRIVER))
    driver.chmod(0o755)
    for name in ("hostname", "nvidia-smi", "sleep", "ssh", "tmux"):
        (tools / name).symlink_to(driver)
    repository = tmp_path / "repository"
    (repository / ".venv/bin").mkdir(parents=True)
    (repository / ".venv/bin/python").symlink_to(driver)
    previous = tmp_path / "previous"
    for attempt in range(8):
        for seed in range(5):
            job = previous / f"attempt-{attempt}-s{seed}"
            (job / "results").mkdir(parents=True)
            (job / "exit-code.txt").write_text("0\n")
            (job / "results/completion.json").write_text("{}\n")
    environment = os.environ | {
        "PATH": f"{tools}:{os.environ['PATH']}",
        "REPO_ROOT": str(repository),
        "CODE_DIR": str(ROOT),
        "DATA_DIR": str(tmp_path / "custom data"),
        "FIRST_ATTEMPT": "8",
        "PREVIOUS_BATCH": str(previous),
        "POOL_ATTEMPTS": "16",
        "CONCURRENT_WAVES": "8",
        "BATCH_TEST_ROOT": str(tmp_path),
        "BATCH_TEST_FAILURE": str(int(failure)),
    }
    try:
        result = subprocess.run(
            [
                "bash",
                str(COORDINATOR),
                "test",
                str(tmp_path / "states"),
                "22",
                "22",
                "22",
                "14",
                "14",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        batch = repository / "outputs/diagnostics/masked-mlp-restarts-test"
        expected_seeds = {
            1000 + 100 * attempt + seed for attempt in range(8, 16) for seed in range(5)
        }
        assert {int(path.name) for path in (tmp_path / "started").iterdir()} == expected_seeds
        for attempt in range(8):
            for seed in range(5):
                prior = previous / f"attempt-{attempt}-s{seed}"
                assert (batch / prior.name).is_symlink()
                assert (batch / prior.name).resolve() == prior
                assert (prior / "exit-code.txt").read_text() == "0\n"
                assert (prior / "results/completion.json").read_text() == "{}\n"
        if failure:
            assert result.returncode == 1, result.stdout + result.stderr
            assert not list(batch.glob("selection-s*.json"))
            killed = set((tmp_path / "killed.txt").read_text().splitlines())
            assert killed == {
                f"scj-acquire-test-a{attempt}-s{seed}"
                for attempt in range(8, 16)
                for seed in range(5)
            }
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            assert len(list(batch.glob("selection-s*.json"))) == 5
            assert not (tmp_path / "killed.txt").exists()
            for seed in range(5):
                receipt = json.loads((batch / f"selection-s{seed}.json").read_text())
                assert len(receipt["candidates"]) == 16
    finally:
        for session in (tmp_path / "sessions").iterdir():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(int(session.read_text()), signal.SIGTERM)
