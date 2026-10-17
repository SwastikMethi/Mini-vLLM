"""Fixed paged KV pool: tensors, free list, ownership, block tables and admission credits.

Only the engine thread touches this object. R0 invariant: a block is free or owned by exactly one request.
"""

import heapq
import math

import torch


class CapacityError(RuntimeError):
    pass


class KVPool:
    def __init__(self, *, layers, blocks, block_size, kv_heads, head_dim, dtype, device):
        self.num_blocks, self.block_size = blocks, block_size
        shape = (layers, blocks, block_size, kv_heads, head_dim)
        # Allocated once; ownership changes never allocate device memory.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self._free = list(range(blocks))  # min-heap: lowest free ID first, reuse scatters naturally
        self.owner: dict[int, str] = {}
        self.tables: dict[str, list[int]] = {}
        self.credits: dict[str, int] = {}

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    def blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_size)

    @property
    def credits_free(self) -> int:
        return self.num_blocks - sum(self.credits.values())

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def reserve(self, rid: str, n: int) -> None:
        assert rid not in self.credits, f"{rid} already holds credits"
        if n > self.credits_free:
            raise CapacityError(f"need {n} credits, {self.credits_free} free")
        self.credits[rid] = n
        self.tables[rid] = []

    def ensure(self, rid: str, tokens: int) -> list[int]:
        """Grow rid's table to cover `tokens` positions. All-or-nothing; returns newly owned block IDs."""
        table = self.tables[rid]
        need = self.blocks_for(tokens) - len(table)
        if need <= 0:
            return []
        if len(table) + need > self.credits[rid]:
            raise CapacityError(f"{rid} would exceed its {self.credits[rid]} reserved blocks")
        if need > len(self._free):  # unreachable while credits are respected; kept as the invariant check
            raise CapacityError(f"need {need} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in range(need)]
        for b in new:
            self.owner[b] = rid
        table.extend(new)
        return new

    def release(self, rid: str) -> list[int]:
        """Return rid's blocks and credits. Exactly-once: a second call returns [] and changes nothing."""
        self.credits.pop(rid, None)
        blocks = self.tables.pop(rid, [])
        for b in blocks:
            assert self.owner.pop(b) == rid
            heapq.heappush(self._free, b)
        return blocks

    def slots(self, rid: str, positions: torch.Tensor) -> torch.Tensor:
        """Flat slot index (block * block_size + offset) for logical positions of rid."""
        table = torch.tensor(self.tables[rid], device=positions.device)
        return table[positions // self.block_size] * self.block_size + positions % self.block_size

    def layer_views(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.k[layer], self.v[layer]
        return k.view(-1, *k.shape[2:]), v.view(-1, *v.shape[2:])
