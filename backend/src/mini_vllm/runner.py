"""Controlled Qwen3 forward over the paged pool: one-prompt prefill and batched single-token decode.

Reuses the HF layer modules (projections, Q/K RMSNorm, RoPE, MLP, norms, lm_head) but owns the attention read
path. The runner never admits, allocates or releases; callers ensure blocks exist before calling.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from .kv_cache import KVPool


@dataclass(frozen=True)
class DecodeRow:
    request_id: str
    token: int  # previously sampled token, uncached until this forward
    position: int  # its absolute position == tokens already cached for this request


class Runner:
    def __init__(self, model, pool: KVPool):
        self.model, self.pool = model, pool
        self.device = next(model.parameters()).device
        cfg = model.config
        self.head_dim = cfg.head_dim
        self.groups = cfg.num_attention_heads // cfg.num_key_value_heads

    @torch.inference_mode()
    def prefill(self, rid: str, ids: list[int]) -> torch.Tensor:
        """Process a whole prompt from position 0; returns last-position logits [vocab]."""
        pos = torch.arange(len(ids), device=self.device)
        slots = self.pool.slots(rid, pos)
        return self._forward(torch.tensor([ids], device=self.device), pos[None], slots, None)[0]

    @torch.inference_mode()
    def decode(self, rows: list[DecodeRow]) -> torch.Tensor:
        """One token per row; returns logits [B, vocab] in row order."""
        ids = torch.tensor([[r.token] for r in rows], device=self.device)
        pos = torch.tensor([[r.position] for r in rows], device=self.device)
        new_slots = torch.cat([self.pool.slots(r.request_id, pos[i]) for i, r in enumerate(rows)])
        # Gather index over each row's history including the new token, padded to the batch maximum.
        lengths = torch.tensor([r.position + 1 for r in rows], device=self.device)
        span = torch.arange(int(lengths.max()), device=self.device)
        valid = span[None] < lengths[:, None]  # [B, S]; True = participates
        hist = torch.stack(
            [self.pool.slots(r.request_id, span.clamp(max=r.position)) for r in rows]
        )  # pad positions point at the row's own last slot, then get zeroed and masked
        return self._forward(ids, pos, new_slots, (hist, valid))

    def _forward(self, ids, pos, write_slots, gather) -> torch.Tensor:
        m = self.model.model
        b, t = ids.shape
        h = m.embed_tokens(ids)
        cos, sin = m.rotary_emb(h, pos)
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            shape = (b, t, -1, self.head_dim)
            q = a.q_norm(a.q_proj(x).view(shape)).transpose(1, 2)
            k = a.k_norm(a.k_proj(x).view(shape)).transpose(1, 2)
            v = a.v_proj(x).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # K is cached post-norm, post-RoPE
            kf, vf = self.pool.layer_views(li)
            kf[write_slots] = k.transpose(1, 2).reshape(-1, *kf.shape[1:])
            vf[write_slots] = v.transpose(1, 2).reshape(-1, *vf.shape[1:])
            if gather is None:  # prefill from position 0: history is exactly this prompt
                o = F.scaled_dot_product_attention(q, repeat_kv(k, self.groups), repeat_kv(v, self.groups), is_causal=True)
            else:
                hist, valid = gather
                keep = valid[:, :, None, None]
                kh = (kf[hist] * keep).transpose(1, 2)  # [B, H_kv, S, D]
                vh = (vf[hist] * keep).transpose(1, 2)
                # Non-square decode: explicit length mask, never is_causal (upper-left alignment is wrong here).
                o = F.scaled_dot_product_attention(
                    q, repeat_kv(kh, self.groups), repeat_kv(vh, self.groups),
                    attn_mask=valid[:, None, None, :], is_causal=False,
                )
            h = h + a.o_proj(o.transpose(1, 2).reshape(b, t, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.model.lm_head(m.norm(h[:, -1])).float()


def sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> int:
    if temperature == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.cpu() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
