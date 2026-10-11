"""Independent addressable CPU random streams, unaffected by global RNG or access order."""

import hashlib
import json

import torch

SPLITS = ("train", "validation", "evaluation_calibration", "test")
STREAM_VERSION = "scjepa-dgp-streams-v1"


def episode_id(root_seed: int, split: str, index: int) -> int:
    """Injectively encode a 31-bit root, two-bit split, and 30-bit episode index."""
    if not 0 <= root_seed < 2**31 or split not in SPLITS or not 0 <= index < 2**30:
        raise ValueError("invalid root seed, split, or episode address")
    return (root_seed << 32) | (SPLITS.index(split) << 30) | index


def stream_seed(root_seed: int, split: str, index: int, stream: str) -> int:
    """Derive a stable seed for one scientific sampling responsibility."""
    identity = episode_id(root_seed, split, index)
    if not stream:
        raise ValueError("stream label must be nonempty")
    payload = json.dumps([STREAM_VERSION, identity, stream], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def generator(root_seed: int, split: str, index: int, stream: str) -> torch.Generator:
    """Create a fresh isolated generator for deterministic episode retrieval."""
    return torch.Generator(device="cpu").manual_seed(stream_seed(root_seed, split, index, stream))
