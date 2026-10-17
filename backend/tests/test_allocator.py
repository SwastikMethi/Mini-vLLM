import pytest
import torch

from mini_vllm.kv_cache import CapacityError, KVPool


def pool(blocks=4, bs=16):
    return KVPool(layers=1, blocks=blocks, block_size=bs, kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu")


def test_credits_bound_reservations():
    p = pool(blocks=4)
    p.reserve("A", 3)
    with pytest.raises(CapacityError):
        p.reserve("B", 2)  # free physical blocks exist, but growth credits do not
    assert p.free_blocks == 4 and p.credits_free == 1
    p.reserve("B", 1)
    assert p.credits_free == 0


def test_ensure_is_all_or_nothing_and_within_credits():
    p = pool(blocks=4)
    p.reserve("A", 2)
    assert p.ensure("A", 17) == [0, 1]
    with pytest.raises(CapacityError):
        p.ensure("A", 33)  # would need a third block
    assert p.tables["A"] == [0, 1] and p.free_blocks == 2
    assert p.ensure("A", 32) == []  # boundary: 32 tokens still fit two blocks


def test_release_exactly_once_and_reuse_scatters():
    p = pool(blocks=4)
    p.reserve("A", 2), p.reserve("B", 2)
    p.ensure("A", 16), p.ensure("B", 16), p.ensure("A", 17)  # A=[0,2], B=[1]
    assert p.tables["A"] == [0, 2]
    assert p.release("A") == [0, 2]
    assert p.release("A") == []  # second release is a no-op
    assert p.credits == {"B": 2} and set(p.owner) == {1}
    p.reserve("C", 2)
    assert p.ensure("C", 20) == [0, 2]  # reused, non-contiguous
    assert all(p.owner[b] == "C" for b in (0, 2))


def test_slot_addressing_four_token_blocks():
    p = pool(blocks=4, bs=4)
    p.reserve("A", 2)
    p.ensure("A", 4)
    assert p.ensure("A", 5) == [1]  # position 4 crosses into logical block 1
    p.tables["A"] = [3, 1]  # scattered mapping
    got = p.slots("A", torch.tensor([3, 4, 5])).tolist()
    assert got == [3 * 4 + 3, 1 * 4 + 0, 1 * 4 + 1]
