"""Repeat the unmodified RGB continuation twice on the failed replay's original physical GPU."""

import argparse
import json
import os
import shlex
import socket
import subprocess
from pathlib import Path
from typing import Any

import torch

from diagnostics.vision_dimer.train import FORKABLE, load_checkpoint
from scjepa.experiments.storage import file_lock, write_json

from .changed_queries import Inputs, require
from .single_acquisition import _head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import DATA, MANIFEST, OUTPUT, PYTHON, checkpoint_hashes, declared, read
from .vision_loss import REPLAY_ABSOLUTE, REPLAY_RELATIVE, compare_replay

FOUR = OUTPUT / "rgb-four-s3"
FAILED = OUTPUT / "vision-combined-s3"
NATIVE = FOUR / "sparse-500k/full"
PARENT = FOUR / "sparse-300k/full/last.pt"
DEST = OUTPUT / "vision-native-control-s3"
START, END = 300_000, 300_500
FIELDS = {
    "prediction",
    "state_std_min",
    "state_std_max",
    "sigreg",
    "open_state",
    "open_action",
    "open_parameter",
    "log_lambda",
    "tolerance",
}
POLICY = """Hypothesis: the unmodified released fork reproduces its archived first logging
interval from the same optimizer, model, sampler and random states on the original physical GPU.
Remove the observational branch used by the failed replay. Launch two independent fresh
native Python processes from the same checkpoint and score every scientific TRAIN log field.
The first repeat alone determines whether native replay clears the screen. The second repeat
measures repeatability regardless of the first outcome. Two archive matches support reproduction
of this interval. Disagreement between repeats measures run-to-run variation. Matching repeats
that both miss the archive indicate a repeatable drift from the older continuation; they do
not isolate its cause. The prior combined seed disposition is retained in every case. Sampler
and RNG state equality among the two earlier failed probes is a separate forensic measurement.
Training uses only the native label-free objective with TF32 off; the native endpoint routine
evaluates calibration and validation. This control uses no endpoint labels for admission."""


def _reading(path: Path, files: Inputs) -> dict[str, Any]:
    files.bind(path)
    rows = [r for r in map(json.loads, path.read_text().splitlines()) if r["step"] == END]
    require(
        len(rows) == 1 and rows[0].keys() == FIELDS | {"seconds_per_step", "step"},
        f"Expected exactly one complete native logging interval at {path}",
    )
    return rows[0]


def _hardware(stage: Path, files: Inputs, *, recorded_root: Path | None = None) -> dict[str, str]:
    paths = list(stage.glob("wandb/run-*/files/wandb-metadata.json"))
    require(len(paths) == 1, f"Ambiguous W&B run metadata at {stage}")
    meta = files.record(paths[0])
    gpus = meta["gpu_nvidia"]
    require(
        len(gpus) == 1
        and gpus[0]["name"] == meta["gpu"]
        and gpus[0]["uuid"].startswith("GPU-")
        and Path(meta["root"]).resolve() == (recorded_root or stage).resolve(),
        f"Ambiguous recorded GPU at {stage}",
    )
    return {"host": meta["host"], "uuid": gpus[0]["uuid"], "name": meta["gpu"]}


def _available(hw: dict[str, str]) -> None:
    require(socket.gethostname().split(".")[0] == hw["host"].split(".")[0], "Wrong native worker")
    base = ["nvidia-smi", "--id=" + hw["uuid"]]
    device = subprocess.check_output(
        [*base, "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True
    ).strip()
    uuid, name, free = (v.strip() for v in device.split(","))
    processes = subprocess.check_output(
        [*base, "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    require(
        uuid == hw["uuid"] and name == hw["name"] and not processes.strip() and int(free) >= 6144,
        "Native physical GPU is occupied, unavailable or below the memory threshold",
    )


def snapshot_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare stored streams and resolved Adam step counters without consuming randomness."""
    require(left["step"] == right["step"] == END, "Wrong failed replay snapshot step")

    def equal(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.dtype == b.dtype and torch.equal(a, b)

    def counters(cp: dict[str, Any]) -> dict[str, int]:
        return {str(k): int(v["step"].item()) for k, v in cp["optimizer"]["state"].items()}

    a, b = left["sampler"], right["sampler"]
    require(a.keys() == b.keys() == {"rng", "order", "offset"}, "Unrecognized sampler layout")
    ca, cb = counters(left), counters(right)
    require(bool(ca) and bool(cb), "Missing optimizer step counters")
    return {
        "step": END,
        "sampler": {
            "rng": equal(a["rng"], b["rng"]),
            "order": equal(a["order"], b["order"]),
            "offset": a["offset"] == b["offset"],
        },
        **{k: equal(left[k], right[k]) for k in ("windows", "cpu_rng", "cuda_rng")},
        "optimizer_steps": {"equal": ca == cb, "original_worker": ca, "gateway": cb},
    }


def _failures(
    files: Inputs, archived: dict[str, Any], hw: dict[str, str], parent: str, four_commit: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    earlier = FAILED / "prior-attempts/gateway-replay"
    records: dict[str, Any] = {}
    snapshots: list[dict[str, Any]] = []
    rev = ""
    for key, root in (("original_worker", FAILED), ("gateway", earlier)):
        m = files.record(root / "metrics.json")
        pre = files.record(root / "prepare.json")
        rep = root / "control-replay"
        probe = files.record(rep / "replay.json")
        forks = files.record(rep / "fork.json")
        require(
            m["seed"] == pre["seed"] == 3
            and m["reason"] == "preparation"
            and m["checks"] == pre["checks"] == {"native_replay": False}
            and m["passed"] is False
            and m["verified"] is False
            and pre["ready"] is False
            and m["new_sibling_updates"] == 0
            and m["route"] == "fresh_changed_winner"
            and m["commit"] == pre["commit"] == probe["commit"]
            and m["native_parent"] == forks["source"] == str(PARENT)
            and m["parent_sha256"]
            == pre["source_sha256"]
            == probe["source_sha256"]
            == forks["source_sha256"]
            == parent
            and forks["source_step"] == probe["start"] == START
            and m["prepare_sha256"] == files[str(root / "prepare.json")],
            f"Wrong trigger at {root}",
        )
        files.bind(root / "NOTE.md", m["note_sha256"])
        for name, digest in pre["files"].items():
            files.bind(root / Path(name).relative_to(FAILED), digest)
        files.bind(rep / "last.pt", probe["resumable_sha256"])
        files.bind(rep / "final.pt", probe["checkpoint_sha256"])
        require(
            m["input_sha256"][str(FOUR / "metrics.json")] == files[str(FOUR / "metrics.json")]
            and m["input_sha256"][str(PARENT)] == parent,
            "Wrong four-head provenance",
        )
        stream = _reading(rep / "history.jsonl", files)
        replay = compare_replay(stream, archived)
        require(
            not replay["numerically_matched"] and all(probe[k] == v for k, v in replay.items()),
            f"Archived replay classification differs at {rep}",
        )
        machine = _hardware(rep, files, recorded_root=FAILED / "control-replay")
        require((machine == hw) == (key == "original_worker"), "Wrong archived worker identity")
        if rev:
            require(m["commit"] == rev, "Original failed attempts used different revisions")
        rev = m["commit"]
        records[key] = {
            "hardware": machine,
            "metrics_sha256": files[str(root / "metrics.json")],
            "source_commit": rev,
            "measured": stream,
            "versus_archive": replay,
            "snapshot_sha256": files[str(rep / "last.pt")],
        }
        snapshots.append(load_checkpoint(rep / "last.pt"))
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "--quiet",
            four_commit,
            "--",
            "diagnostics/vision_dimer",
            "diagnostics/shared",
            "src",
        ],
        check=True,
    )
    return records, snapshot_comparison(snapshots[0], snapshots[1]), rev


def _argv(stage: Path) -> list[str]:
    return [
        PYTHON,
        "-m",
        "diagnostics.vision_dimer.train",
        "fork",
        "--source",
        str(PARENT),
        "--output-dir",
        str(stage),
        "--steps",
        str(END),
        "--wandb-mode",
        "online",
        "--wandb-group",
        f"icml-review-native-control-s3-{stage.name}",
    ]


def _command(stage: Path, hw: dict[str, str]) -> str:
    return (
        f"OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={shlex.quote(hw['uuid'])} "
        "PYTHONPATH=$PWD:$PWD/src "
        + shlex.join(_argv(stage))
        + f" > {shlex.quote(str(stage / 'run.log'))} 2>&1"
    )


def _stage(
    rev: str,
    stage: Path,
    hw: dict[str, str],
    config: dict[str, Any],
    inputs: Inputs,
    note: str,
    archived: dict[str, Any],
) -> dict[str, Any]:
    declared(stage / "NOTE.md", f"# {stage.name}\n\n{note}\n    {_command(stage, hw)}\n")
    _head(rev)
    result = stage / "stage.json"
    log, code = stage / "run.log", stage / "exit_code"
    if not log.exists():
        require(
            not result.exists()
            and not any(
                (stage / k).exists()
                for k in ("fork.json", "history.jsonl", "final.pt", "last.pt", "exit_code")
            ),
            "Refusing to overwrite existing native evidence",
        )
        _available(hw)
        require(
            all(
                os.environ.get(k) != "1"
                for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
            ),
            "Environment would force TF32 on",
        )
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": hw["uuid"],
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(map(str, (Path.cwd(), Path.cwd() / "src"))),
        }
        with log.open("x") as out:
            process = subprocess.run(_argv(stage), env=env, stdout=out, stderr=subprocess.STDOUT)
        code.write_text(f"{process.returncode}\n")
        process.check_returncode()
    require(
        code.is_file() and code.read_text().strip() == "0",
        f"Interrupted or failed process at {stage}; preserve its evidence",
    )
    _head(rev)
    files = Inputs()
    measured = _reading(stage / "history.jsonl", files)
    fork = files.record(stage / "fork.json")
    cfg = files.record(stage / "config.json")["config"]
    require(
        {k: v for k, v in cfg.items() if k not in FORKABLE}
        == {k: v for k, v in config.items() if k not in FORKABLE}
        and cfg["steps"] == END
        and cfg["output_dir"] == str(stage)
        and cfg["wandb_mode"] == "online"
        and cfg["wandb_group"] == _argv(stage)[-1]
        and fork["source"] == str(PARENT)
        and fork["source_sha256"] == inputs[str(PARENT)]
        and fork["source_step"] == START
        and files.record(stage / "final.json")["step"] == END
        and _hardware(stage, files) == hw,
        f"Native process configuration, hardware or budget differs at {stage}",
    )
    for name in ("NOTE.md", "run.log", "exit_code", "evaluations.jsonl", "final.pt", "last.pt"):
        files.bind(stage / name)
    record = {
        "commit": rev,
        "command": _command(stage, hw),
        "parent_sha256": inputs[str(PARENT)],
        "hardware": hw,
        "measured": measured,
        "versus_archive": compare_replay(measured, archived),
        "output_sha256": dict(files),
    }
    _head(rev)
    if result.exists():
        require(read(result) == record, "Native stage receipt differs")
    else:
        write_json(result, record)
    inputs.bind(result)
    inputs.update(files)
    return record


def run(rev: str) -> dict[str, Any]:
    """Run both fixed native replicates, retaining the first replicate as the only screen."""
    _head(rev)
    with file_lock(DEST / ".lock"):
        files = Inputs()
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        files.bind(MANIFEST)
        four = files.record(FOUR / "metrics.json")
        cfg = files.record(NATIVE / "config.json")["config"]
        comp = files.record(NATIVE / "review-completion.json")
        source = files.bind(PARENT, comp["parent_sha256"])
        require(
            four["seed"] == cfg["seed"] == 3
            and four["route"] == "fresh"
            and cfg["stage"] == "full"
            and cfg["device"] == "cuda"
            and cfg["log_every"] == END - START
            and cfg["steps"] == 500000
            and Path(cfg["data_dir"]) == DATA
            and four["checkpoint_sha256"]["500k/full"] == comp["checkpoint_sha256"]
            and four["commit"] == comp["commit"],
            "Wrong four-head native full endpoint",
        )
        fork = files.record(NATIVE / "fork.json")
        require(
            fork["source"] == str(PARENT)
            and fork["source_step"] == START
            and fork["source_sha256"] == source,
            "Wrong original fork",
        )
        files.bind(NATIVE / "final.pt", comp["checkpoint_sha256"])
        files.bind(FOUR / "NOTE.md", four["note_sha256"])
        manifest = DATA / "manifest.json"
        data = files.record(manifest)
        files.bind(manifest, four["data_manifest_sha256"])
        registry = checkpoint_hashes()
        splits = {k: data["splits"][k]["sha256"] for k in ("train", "calibration", "validation")}
        for k, h in splits.items():
            path = DATA / f"{k}.pt"
            require(
                registry[str(path)] == files.bind(path, h), f"Data registration differs at {path}"
            )
        hw = _hardware(NATIVE, files)
        worker = files.record(OUTPUT / "four-queue/rgb-four-s3/full.ok.json")
        require(
            worker["commit"] == four["commit"]
            and worker["seed"] == 3
            and worker["phase"] == "full"
            and worker["exit_code"] == 0
            and worker["host"] == hw["host"]
            and worker["gpu"] == hw["name"],
            "Native worker receipt differs",
        )
        archived = _reading(NATIVE / "history.jsonl", files)
        failures, streams, failed_rev = _failures(files, archived, hw, source, four["commit"])
        require(
            read(FAILED / "metrics.json")["splits"] == splits
            and cfg["tau"] == archived["tolerance"] == read(FAILED / "metrics.json")["tolerance"],
            "Cohort or native tolerance changed",
        )
        for path in (
            "diagnostics/vision_dimer/train.py",
            "ICML_review/code/vision_loss.py",
            __file__,
        ):
            files.bind(Path(path).resolve().relative_to(Path.cwd()))
        stages = [DEST / name for name in ("first", "second")]
        command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
            [PYTHON, "-m", "ICML_review.code.vision_native_control", rev]
        )
        failure_hashes = {k: v["metrics_sha256"] for k, v in failures.items()}
        note = (
            f"{POLICY}\n\nCommit {rev}. Original four-head {four['commit']}. "
            f"Failed replay {failed_rev}.\n"
            f"Data {DATA}, manifest {files[str(manifest)]}, registered splits {splits}.\n"
            f"Parent {PARENT}, SHA-256 {source}. Native archive {NATIVE}, tau {cfg['tau']}.\n"
            f"Registered failure digests {failure_hashes}.\n"
            f"Native worker and device {hw}. TRAIN log step {END}. "
            f"Relative tolerance {REPLAY_RELATIVE}, absolute tolerance {REPLAY_ABSOLUTE}.\n"
            f"\n    {command}\n" + "".join(f"\n    {_command(s, hw)}\n" for s in stages)
        )
        declared(DEST / "NOTE.md", "# Two independent native controls\n\n" + note)
        first, second = [_stage(rev, s, hw, cfg, files, note, archived) for s in stages]
        comparison = compare_replay(second["measured"], first["measured"])
        result = {
            "seed": 3,
            "commit": rev,
            "command": command,
            "verified": True,
            "screen_matched": first["versus_archive"]["numerically_matched"],
            "comparisons": {
                "first_vs_archive": first["versus_archive"],
                "second_vs_archive": second["versus_archive"],
                "second_vs_first": comparison,
            },
            "prior_failures": failures,
            "prior_failed_snapshot_comparison": streams,
            "archived": archived,
            "hardware": hw,
            "source": str(PARENT),
            "source_sha256": source,
            "splits": splits,
            "tau": cfg["tau"],
            "stages": {"first": first, "second": second},
            "note_sha256": files.bind(DEST / "NOTE.md"),
            "input_sha256": dict(files),
        }
        _head(rev)
        end = DEST / "metrics.json"
        if end.exists():
            require(read(end) == result, "Terminal control receipt differs")
        else:
            write_json(end, result)
        return result


def main() -> None:
    """Launch the two predeclared native control processes on the original RGB worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    print(json.dumps({"screen_matched": run(parser.parse_args().revision)["screen_matched"]}))


if __name__ == "__main__":
    main()
