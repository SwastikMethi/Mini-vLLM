"""Load the pinned Qwen3 model and tokenizer once; shared by the reference harness and the runner."""

import platform

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID, MODEL_REVISION, dtype_for


def load(device: str):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype_for(device))
    model.to(device).eval()
    cfg = model.config
    # The runner reimplements attention against these internals; fail loudly if they drift.
    assert type(model).__name__ == "Qwen3ForCausalLM"
    assert not any(t == "sliding_attention" for t in getattr(cfg, "layer_types", []) or [])
    attn = model.model.layers[0].self_attn
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
    return model, tok


def encode_chat(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def manifest(model, device: str) -> dict:
    cfg = model.config
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": device,
        "device_name": platform.processor() or platform.machine(),
        "dtype": str(dtype_for(device)).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "num_layers": cfg.num_hidden_layers,
        "num_kv_heads": cfg.num_key_value_heads,
        "num_q_heads": cfg.num_attention_heads,
        "head_dim": cfg.head_dim,
    }
