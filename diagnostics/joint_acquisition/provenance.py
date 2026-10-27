"""Verify archived training bytes separately from the new diagnostic identity."""

import ast
import hashlib
from pathlib import Path


def historical_training_identity(root: Path) -> str:
    """Reproduce the original prediction-control identity from an actual source tree."""
    production = hashlib.sha256()
    files = sorted((root / "src/scjepa").rglob("*.py"))
    files += sorted((root / "scripts").glob("*.py"))
    if not files or not (root / "scripts").is_dir():
        raise ValueError("Historical source root must contain src/scjepa and scripts")
    for path in files:
        production.update(str(path.relative_to(root)).encode())
        production.update(path.read_bytes())
    digest = hashlib.sha256(production.hexdigest().encode())
    diagnostics = root / "diagnostics"
    digest.update((diagnostics / "evaluate_context_variance.py").read_bytes())
    for name in (
        "predictive_supervision_control",
        "multiquery_control",
        "parameter_sigreg_control",
    ):
        for path in sorted((diagnostics / name).glob("*.py")):
            digest.update(str(path.relative_to(diagnostics)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _semantics(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
        ):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def validate_training_semantics(archive: Path, runtime: Path) -> None:
    """Require unchanged model, DGP, objective and optimizer mapping semantics.

    Only documentation/format differences are ignored in training dependencies.
    Experiment evaluation/reporting code may evolve because it is not used to
    restore or update the frozen-state prediction-control checkpoint.
    """
    directories = (
        "src/scjepa/models",
        "src/scjepa/data",
        "src/scjepa/losses",
        "src/scjepa/third_party",
        "src/scjepa/training",
    )
    paths: set[Path] = set()
    for directory in directories:
        archived = {path.relative_to(archive) for path in (archive / directory).rglob("*.py")}
        current = {path.relative_to(runtime) for path in (runtime / directory).rglob("*.py")}
        if not archived or archived != current:
            raise ValueError(f"Historical training dependency file set changed: {directory}")
        paths.update(archived)
    paths.update(
        Path(name)
        for name in (
            "diagnostics/multiquery_control/forward.py",
            "diagnostics/multiquery_control/objective.py",
            "diagnostics/predictive_supervision_control/objective.py",
            "diagnostics/parameter_sigreg_control/run.py",
        )
    )
    for path in sorted(paths):
        if not (archive / path).is_file() or not (runtime / path).is_file():
            raise ValueError(f"Missing historical training dependency: {path}")
        if _semantics(archive / path) != _semantics(runtime / path):
            raise ValueError(f"Runtime training semantics differ from verified archive: {path}")
