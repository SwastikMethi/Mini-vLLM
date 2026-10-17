"""Run configuration. One frozen object per run; changing block size needs a new run."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"  # pinned 2026-10-17
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"


def default_device() -> str:
    return os.environ.get("MINI_VLLM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")


def dtype_for(device: str) -> torch.dtype:
    # CPU is the fp32 parity path; MPS runs the checkpoint's native bf16.
    return torch.float32 if device == "cpu" else torch.bfloat16


@dataclass(frozen=True)
class Config:
    device: str = field(default_factory=default_device)
    block_size: int = 16
    pool_blocks: int = 256
    max_active: int = 8
    queue_cap: int = 64
    max_request_tokens: int = 2048
    default_max_new_tokens: int = 128
    deadline_ms: int = 120_000
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"

    @property
    def dtype(self) -> torch.dtype:
        return dtype_for(self.device)
