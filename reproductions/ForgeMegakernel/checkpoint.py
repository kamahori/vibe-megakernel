"""Read-only Qwen3 checkpoint loader for the independent decode oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch

from .config import ModelConfig


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qwen3_checkpoint(snapshot: Path, max_seq: int) -> tuple[ModelConfig, dict[str, torch.Tensor]]:
    """Load exactly the canonical bf16 bits, with explicit weight-name mapping.

    The cache length is intentionally caller-bounded; the checkpoint's advertised
    maximum context would allocate a large mostly empty CPU KV arena.
    """
    snapshot = snapshot.resolve()
    metadata = json.loads((snapshot / "config.json").read_text())
    if metadata.get("model_type") != "qwen3" or max_seq < 1:
        raise ValueError("expected a Qwen3 model and positive cache length")
    if metadata.get("torch_dtype") not in {"bfloat16", "torch.bfloat16"}:
        raise ValueError("checkpoint must declare bf16 weights")
    if max_seq > metadata["max_position_embeddings"]:
        raise ValueError("requested cache exceeds checkpoint context limit")
    cfg = ModelConfig(
        hidden=metadata["hidden_size"], layers=metadata["num_hidden_layers"],
        q_heads=metadata["num_attention_heads"], kv_heads=metadata["num_key_value_heads"],
        head_dim=metadata["head_dim"], intermediate=metadata["intermediate_size"],
        vocab=metadata["vocab_size"], max_seq=max_seq,
        eps=metadata["rms_norm_eps"], rope_theta=metadata["rope_theta"],
    )
    path = snapshot / "model.safetensors"
    names = {
        "ln1": "input_layernorm.weight",
        "wq": "self_attn.q_proj.weight",
        "wk": "self_attn.k_proj.weight",
        "wv": "self_attn.v_proj.weight",
        "qn": "self_attn.q_norm.weight",
        "kn": "self_attn.k_norm.weight",
        "wo": "self_attn.o_proj.weight",
        "ln2": "post_attention_layernorm.weight",
        "wg": "mlp.gate_proj.weight",
        "wu": "mlp.up_proj.weight",
        "wd": "mlp.down_proj.weight",
    }
    with safe_open(str(path), framework="pt", device="cpu") as source:
        weights = {
            "embed": source.get_tensor("model.embed_tokens.weight"),
            "final_norm": source.get_tensor("model.norm.weight"),
        }
        lm_head = source.get_tensor("lm_head.weight")
        if metadata.get("tie_word_embeddings") is not True or not torch.equal(lm_head, weights["embed"]):
            raise ValueError("reference expects a tied Qwen3 LM head")
        for local_name, suffix in names.items():
            weights[local_name] = torch.stack([
                source.get_tensor(f"model.layers.{layer}.{suffix}")
                for layer in range(cfg.layers)
            ])
    if any(tensor.dtype != torch.bfloat16 for tensor in weights.values()):
        raise ValueError("all checkpoint tensors must be bf16")
    expected = {
        "embed": (cfg.vocab, cfg.hidden),
        "final_norm": (cfg.hidden,),
        "ln1": (cfg.layers, cfg.hidden),
        "wq": (cfg.layers, cfg.q_heads * cfg.head_dim, cfg.hidden),
        "wk": (cfg.layers, cfg.kv_heads * cfg.head_dim, cfg.hidden),
        "wv": (cfg.layers, cfg.kv_heads * cfg.head_dim, cfg.hidden),
        "qn": (cfg.layers, cfg.head_dim),
        "kn": (cfg.layers, cfg.head_dim),
        "wo": (cfg.layers, cfg.hidden, cfg.q_heads * cfg.head_dim),
        "ln2": (cfg.layers, cfg.hidden),
        "wg": (cfg.layers, cfg.intermediate, cfg.hidden),
        "wu": (cfg.layers, cfg.intermediate, cfg.hidden),
        "wd": (cfg.layers, cfg.hidden, cfg.intermediate),
    }
    for name, shape in expected.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(weights[name].shape)} != {shape}")
    return cfg, weights
