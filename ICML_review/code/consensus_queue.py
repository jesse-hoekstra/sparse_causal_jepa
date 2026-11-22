"""Run independent, pilot-gated stages on idle GPUs in a shared, pinned checkout."""

import argparse
import contextlib
import fcntl
import os
import shlex
import socket
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scjepa.experiments.storage import write_json

from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import OUTPUT, PYTHON, read, sha

QUEUE = OUTPUT / "consensus-queue"
LINEAR = ("linear-consensus-synthetic", "linear-consensus-vision")
RESTARTS = ("rgb-predictive", "synthetic-warm")
TEACHERS = ("vision-invariant", "synthetic-consensus")
# Prefer independent short controls when an appropriate device becomes free.
METHODS = (*LINEAR, *TEACHERS, *RESTARTS)
RGB = ("linear-consensus-vision", "vision-invariant", "rgb-predictive", "rgb-four")
PHASES = {
    "vision-invariant": ("prepare", "own", "sibling", "assess"),
    "synthetic-consensus": ("same", "other", "assess"),
    **{m: ("prepare", "full", "baseline", "assess") for m in (*RESTARTS, "rgb-four")},
    **{m: ("fit",) for m in LINEAR},
}


@dataclass(frozen=True)
class Task:
    """A uniquely identified phase with its command, compute constraint and predecessors."""

    method: str
    seed: int
    phase: str

    @property
    def directory(self) -> Path:
        """Keep scheduling receipts apart from scientific endpoints."""
        parent = OUTPUT / "four-queue" if self.method == "rgb-four" else QUEUE
        return parent / f"{self.method}-s{self.seed}"

    def command(self, rev: str) -> list[str]:
        """Call the same public module documented in each scientific note."""
        if self.phase == "queries":
            module, args = "consensus_queries", [rev, self.method, str(self.seed)]
        elif self.method in LINEAR:
            module, args = "linear_consensus", [rev, self.method.split("-")[-1], str(self.seed)]
        else:
            module = self.method.replace("-", "_")
            args = [rev, str(self.seed), self.phase]
        return [PYTHON, "-m", "ICML_review.code." + module, *args]

    def finished(self, rev: str) -> bool:
        """Accept a phase receipt only for this command and checkout."""
        path = self.directory / f"{self.phase}.ok.json"
        if not path.exists():
            return False
        value = read(path)
        if (
            value.get("commit") != rev
            or value.get("phase") != self.phase
            or value.get("seed") != self.seed
            or value.get("exit_code") != 0
            or value.get("command") != shlex.join(self.command(rev))
        ):
            raise ValueError(f"Invalid phase receipt {path}")
        return True

    def matches_gpu(self, model: str) -> bool:
        """Replay native stochastic continuations on the original seed's GPU family."""
        if self.phase in ("fit", "queries", "assess"):
            return True
        if self.method in TEACHERS:
            originals = 3 if self.method == "vision-invariant" else 2
            return ("3090" in model) == (self.seed < originals)
        return True

    def predecessors(self) -> tuple[str, ...]:
        """Determine which same-seed training phases must have completed."""
        phases = PHASES[self.method]
        if self.phase == "assess":
            return phases[:-1]
        if "prepare" in phases and self.phase != "prepare":
            return ("prepare",)
        return ()


def receipt(method: str, seed: int, rev: str) -> dict[str, Any] | None:
    """Check the seed and revision of an atomic terminal result."""
    path = OUTPUT / f"{method}-s{seed}/metrics.json"
    if not path.exists():
        return None
    value = read(path)
    if value["commit"] != rev or value["seed"] != seed or type(value["passed"]) is not bool:
        raise ValueError(f"Invalid terminal result {path}")
    return value


def pending(rev: str, methods: tuple[str, ...] = METHODS) -> list[Task]:
    """Enumerate ready phases and respect every dataset-specific seed-zero decision."""
    tasks: list[Task] = []
    for method in methods:
        pilot = receipt(method, 0, rev)
        for seed in range(5 if pilot and pilot["passed"] else 1):
            completed = receipt(method, seed, rev)
            if completed:
                q = OUTPUT / f"consensus-queries-{method}-s{seed}/metrics.json"
                if method in RGB:
                    if not q.exists():
                        tasks.append(Task(method, seed, "queries"))
                    else:
                        measurement = read(q)
                        source = OUTPUT / f"{method}-s{seed}/metrics.json"
                        if (
                            measurement["commit"] != rev
                            or measurement["method"] != method
                            or measurement["seed"] != seed
                            or measurement["input_sha256"].get(str(source)) != sha(source)
                        ):
                            raise ValueError(f"Invalid intervention result {q}")
                continue
            for phase in PHASES[method]:
                task = Task(method, seed, phase)
                if task.finished(rev):
                    if phase in ("assess", "fit") and receipt(method, seed, rev) is None:
                        marker = task.directory / f"{phase}.ok.json"
                        if time.time() - marker.stat().st_mtime > 120:
                            raise ValueError(f"Missing {method} seed {seed} terminal receipt")
                        tasks.append(task)
                    continue
                if all(Task(method, seed, p).finished(rev) for p in task.predecessors()):
                    tasks.append(task)
    return tasks


@contextlib.contextmanager
def try_lock(path: Path) -> Generator[bool, None, None]:
    """Try a shared-filesystem advisory lock that disappears when its process dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def device() -> tuple[str, bool]:
    """Read physical occupancy without changing any GPU process."""

    def query(*args: str) -> str:
        return subprocess.check_output(["nvidia-smi", "--id=0", *args], text=True).strip()

    apps = query("--query-compute-apps=pid", "--format=csv,noheader")
    name, free = query("--query-gpu=name,memory.free", "--format=csv,noheader,nounits").split(",")
    return name, not apps and int(free) >= 8192


def attempt(task: Task, rev: str, gpu: str, methods: tuple[str, ...] = METHODS) -> bool:
    """Claim a ready phase, execute with a pinned checkout, and preserve all diagnostics."""
    root = task.directory
    phase = task.phase
    failed = root / f"{phase}.failure.json"
    if failed.exists() or not task.matches_gpu(gpu):
        return False
    with try_lock(root / f".{phase}.lock") as acquired:
        if (
            not acquired
            or task not in pending(rev, methods)
            or failed.exists()
            or (phase != "queries" and task.finished(rev))
        ):
            return False
        if not device()[1]:
            return False
        check_head(rev)
        command = task.command(rev)
        info = {
            "commit": rev,
            "seed": task.seed,
            "phase": phase,
            "host": socket.gethostname(),
            "gpu": gpu,
            "command": shlex.join(command),
            "start_unix": time.time(),
        }
        write_json(root / f"{phase}.start.json", info)
        with (root / f"{phase}.log").open("a", buffering=1) as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        check_head(rev)
        info.update(exit_code=result.returncode, end_unix=time.time())
        write_json(failed if result.returncode else root / f"{phase}.ok.json", info)
        return True


def main() -> None:
    """Work until all qualified training and causal evaluations have terminal receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("--four", action="store_true", help="run the four-head campaign only")
    args = parser.parse_args()
    rev: str = args.commit
    methods = ("rgb-four",) if args.four else METHODS
    jobs = OUTPUT / "four-queue" if args.four else QUEUE
    os.environ.update(
        PYTHONPATH=f"{Path.cwd()}:{Path.cwd() / 'src'}",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="0",
        PYTHONUNBUFFERED="1",
    )
    worker = socket.gethostname().split(".")[0]
    with try_lock(jobs / f".worker-{worker}.lock") as acquired:
        if not acquired:
            raise RuntimeError("A campaign worker is already active on this physical node")
        while True:
            check_head(rev)
            tasks = pending(rev, methods)
            if not tasks:
                return
            gpu, free = device()
            if free and any(attempt(t, rev, gpu, methods) for t in tasks):
                continue
            if all((t.directory / f"{t.phase}.failure.json").exists() for t in tasks):
                raise RuntimeError("Remaining eligible phases have a recorded technical failure")
            time.sleep(40)


if __name__ == "__main__":
    main()
