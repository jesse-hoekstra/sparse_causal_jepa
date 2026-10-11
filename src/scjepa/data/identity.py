"""Deterministic dataset identity for generic checkpoint/resume validation."""

import hashlib
import json
from collections.abc import Mapping, Sized
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset


def tensor_digest(value: Tensor) -> str:
    """Hash dtype, shape, and contents with bounded byte-copy working memory."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
    flat = value.detach().reshape(-1)
    # Copies at most one MiB at a time, including tensors resident on accelerators.
    elements = max(1, 1024 * 1024 // value.element_size())
    for offset in range(0, flat.numel(), elements):
        chunk = flat[offset : offset + elements].contiguous().cpu().view(torch.uint8)
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def training_tensors_fingerprint(tensors: Mapping[str, Tensor]) -> str:
    """Bind every training tensor, including original episode IDs, to one identity."""
    description = {key: tensor_digest(value) for key, value in sorted(tensors.items())}
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def dataset_fingerprint(dataset: Dataset[dict[str, Tensor]]) -> str:
    """Use declared provenance, otherwise stream-hash a finite deterministic dataset.

    Built-in generated/bundle adapters expose a versioned descriptor identity and
    need no rendering here. Custom datasets should expose an immutable fingerprint
    for efficiency; the fallback checks their training samples once at setup.
    """
    identity: object = getattr(dataset, "fingerprint", None)
    if isinstance(identity, str) and identity:
        content: object = getattr(dataset, "content_fingerprint", None)
        if isinstance(content, str) and content:
            return hashlib.sha256(json.dumps([identity, content]).encode()).hexdigest()
        return identity
    digest = hashlib.sha256(b"scjepa-generic-training-dataset-v1")
    size = len(cast(Sized, dataset))
    digest.update(str(size).encode())
    for index in range(size):
        digest.update(training_tensors_fingerprint(dataset[index]).encode())
    return digest.hexdigest()
