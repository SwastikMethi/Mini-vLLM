from dataclasses import dataclass, field

import torch


@dataclass
class Request:
    rid: str
    client_request_id: str
    prompt: str
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float
    seed: int
    accepted_ns: int
    deadline_ns: int
    status: str = "queued"  # queued | prefill | decoding | completed | cancelled | failed
    tokens: list[int] = field(default_factory=list)
    text: str = ""  # streamed text so far (complete characters only)
    cached: int = 0  # committed K/V positions
    decode_step: int = 0
    reservation: int = 0
    cancel: str | None = None  # "user" | "deadline" once requested
    wait: tuple[str, int] | None = None  # last emitted (reason, position)
    generator: torch.Generator | None = None

    def __post_init__(self):
        if self.temperature > 0:
            self.generator = torch.Generator().manual_seed(self.seed)

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "cancelled", "failed")
