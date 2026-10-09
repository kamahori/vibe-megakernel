"""Capture-safe PyTorch baselines for timing P0 candidates.

The grading references in ``megabench.tasks`` synchronize with the host: they
read token ids and routing or acceptance indices with ``.item()`` and build
small device tensors from Python values. Stream capture rejects those calls,
so the harness's CUDA/HIP graph baseline fails, and ``torch.compile`` breaks
its graph at the same points.

``build`` returns a step function that runs the same operators with every
index kept on the device. Host-side constants such as RoPE rows are prepared
before capture. The step functions are timing baselines, not oracles:
``timing_baselines`` checks each variant against the BF16 reference and the
FP32 oracle band before reporting its time.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F

from .cases import Case
from .harness.benchmark import _measure
from .harness.correctness import _compare
from .tasks.common import attention, rms, rope
from .tasks.references.qwen3 import _rms, _rot_half, rope_tables

Inputs = dict[str, torch.Tensor]
Step = Callable[[Inputs], dict[str, torch.Tensor]]

SUPPORTED_FAMILIES = ("dense_step", "moe_step", "quant_step", "spec_target_step")
# Inductor autotunes matmul and reduction configs; graph replay is applied
# separately so each variant's time is attributable.
COMPILE_MODE = "max-autotune-no-cudagraphs"


def _row(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``table[index]`` for a 0-d device index, without a host read."""
    return table.index_select(0, index.reshape(1))[0]


def _qwen_attention_block(x, t, layer, cos, sin, cfg_eps, qh, kvh, d):
    xn = _rms(x, t["ln1"][layer], cfg_eps)
    q = (t["wq"][layer] @ xn).view(qh, d)
    k = (t["wk"][layer] @ xn).view(kvh, d)
    v = (t["wv"][layer] @ xn).view(kvh, d)
    q = _rms(q, t["qn"][layer], cfg_eps)
    k = _rms(k, t["kn"][layer], cfg_eps)
    q = q * cos.to(q.dtype) + _rot_half(q) * sin.to(q.dtype)
    k = k * cos.to(k.dtype) + _rot_half(k) * sin.to(k.dtype)
    ks = torch.cat((t["kcache"][layer], k[None]), dim=0)
    vs = torch.cat((t["vcache"][layer], v[None]), dim=0)
    o = attention(q, ks, vs).reshape(qh * d)
    return x + t["wo"][layer] @ o, k, v


def _rope_row(cfg, context: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode position's cos/sin row, built on the host before capture."""
    cos, sin = rope_tables(cfg, device)
    return cos[context].contiguous(), sin[context].contiguous()


def _dense(case: Case, device) -> Step:
    from .tasks.dense import _qwen3_config

    cfg = _qwen3_config(case)
    p = case.params
    qh, kvh, d = p["q_heads"], p["kv_heads"], p["head_dim"]
    cos, sin = _rope_row(cfg, p["context"], device)

    def step(t: Inputs) -> dict[str, torch.Tensor]:
        x = _row(t["embed"], t["token"])
        k_writes, v_writes = [], []
        for layer in range(p["layers"]):
            x, k, v = _qwen_attention_block(x, t, layer, cos, sin, cfg.eps, qh, kvh, d)
            k_writes.append(k)
            v_writes.append(v)
            xn = _rms(x, t["ln2"][layer], cfg.eps)
            gate = t["wg"][layer] @ xn
            up = t["wu"][layer] @ xn
            x = x + t["wd"][layer] @ (F.silu(gate) * up)
        logits = (t["embed"] @ _rms(x, t["fnorm"], cfg.eps)).float()
        return {"logits": logits, "next_token": logits.argmax().to(torch.int64),
                "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes)}

    return step


def _moe(case: Case, device) -> Step:
    from .tasks.moe import config

    cfg = config(case)
    qh, kvh, d = cfg.q_heads, cfg.kv_heads, cfg.head_dim
    cos, sin = _rope_row(cfg, case.params["context"], device)

    def step(t: Inputs) -> dict[str, torch.Tensor]:
        x = _row(t["embed"], t["token"])
        k_writes, v_writes = [], []
        for layer in range(cfg.layers):
            x, k, v = _qwen_attention_block(x, t, layer, cos, sin, cfg.eps, qh, kvh, d)
            k_writes.append(k)
            v_writes.append(v)
            xn = _rms(x, t["ln2"][layer], cfg.eps)
            router = t["wrt"][layer].float() @ xn.float()
            top_v, top_i = torch.topk(router, cfg.topk)
            probs = torch.softmax(top_v, dim=-1)
            # Gather the selected experts by device index instead of reading
            # each expert id on the host.
            gate = t["wg"][layer].index_select(0, top_i) @ xn
            up = t["wu"][layer].index_select(0, top_i) @ xn
            hidden = F.silu(gate) * up
            down = (t["wd"][layer].index_select(0, top_i) @ hidden.unsqueeze(-1)).squeeze(-1)
            acc = torch.zeros_like(x, dtype=torch.float32)
            for slot in range(cfg.topk):
                acc = acc + down[slot].float() * probs[slot]
            x = x + acc.to(x.dtype)
        logits = (t["lm_head"] @ _rms(x, t["fnorm"], cfg.eps)).float()
        return {"logits": logits, "next_token": logits.argmax().to(torch.int64),
                "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes)}

    return step


def _gemma(case: Case, device) -> Step:
    from .tasks.gemma import dequant

    p = case.params
    h, d, context, bits = p["hidden"], p["head_dim"], p["context"], p["bits"]
    qh, kvh = p["q_heads"], p["kv_heads"]
    embed_scale = torch.tensor(math.sqrt(h), device=device, dtype=torch.bfloat16)
    positions = torch.tensor([context], device=device)
    attn_scale = p.get("query_pre_attn_scalar", d) ** -0.5

    def step(t: Inputs) -> dict[str, torch.Tensor]:
        def linear(name: str, layer: int, vec: torch.Tensor) -> torch.Tensor:
            matrix = (t[name][layer] if bits == 16 else
                      dequant(t[name][layer], t[f"{name}_scale"][layer], bits).to(vec.dtype))
            return matrix @ vec

        x = _row(t["embed"], t["token"]) * embed_scale
        k_writes, v_writes = [], []
        for layer in range(p["layers"]):
            xn = rms(x, t["ln1"][layer], gemma=True)
            q = linear("wq", layer, xn).view(qh, d)
            k = linear("wk", layer, xn).view(kvh, d)
            v = linear("wv", layer, xn).view(kvh, d)
            q = rms(q, t["qn"][layer], gemma=True)
            k = rms(k, t["kn"][layer], gemma=True)
            local = (layer + 1) % 6 != 0
            theta = 10_000.0 if local else 1_000_000.0
            factor = 1.0 if local else 8.0
            q = rope(q, positions, theta, factor=factor)
            k = rope(k, positions, theta, factor=factor)
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            start = max(0, context + 1 - p["local_window"]) if local else 0
            ks = torch.cat((t["kcache"][layer, start:], k[None]), dim=0)
            vs = torch.cat((t["vcache"][layer, start:], v[None]), dim=0)
            attn = attention(q, ks, vs, scale=attn_scale).reshape(qh * d)
            x = x + rms(linear("wo", layer, attn), t["ln2"][layer], gemma=True)
            xn = rms(x, t["ln3"][layer], gemma=True)
            gate = linear("wg", layer, xn)
            up = linear("wu", layer, xn)
            mlp = linear("wd", layer, F.gelu(gate, approximate="tanh") * up)
            x = x + rms(mlp, t["ln4"][layer], gemma=True)
        logits = (t["embed"] @ rms(x, t["fnorm"], gemma=True)).float()
        return {"logits": logits, "next_token": logits.argmax().to(torch.int64),
                "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes)}

    return step


def _eagle3(case: Case, device) -> Step:
    p = case.params
    depth, context, d = p["draft_depth"], p["context"], p["head_dim"]
    qh, kvh, layers = p["q_heads"], p["kv_heads"], p["layers"]
    steps = depth + 1
    positions = torch.arange(context, context + steps, device=device)
    future_mask = (torch.arange(context + steps, device=device)[None, :] >
                   positions[:, None])
    slots = torch.arange(steps, device=device)
    feature_layers = (max(0, layers // 4 - 1), max(0, layers // 2 - 1), layers - 1)

    def step(t: Inputs) -> dict[str, torch.Tensor]:
        drafts = t["draft_tokens"]
        token_ids = torch.cat((t["token"].reshape(1), drafts))
        x = t["embed"].index_select(0, token_ids)
        k_writes, v_writes, features = [], [], {}
        for layer in range(layers):
            xn = rms(x, t["ln1"][layer], eps=1e-5)
            q = (xn @ t["wq"][layer].T).view(steps, qh, d)
            k = (xn @ t["wk"][layer].T).view(steps, kvh, d)
            v = (xn @ t["wv"][layer].T).view(steps, kvh, d)
            q = rope(q, positions, 500_000.0)
            k = rope(k, positions, 500_000.0)
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            ks = torch.cat((t["kcache"][layer], k), dim=0)
            vs = torch.cat((t["vcache"][layer], v), dim=0)
            attn = attention(q, ks, vs, allowed=~future_mask).reshape(steps, qh * d)
            x = x + attn @ t["wo"][layer].T
            xn = rms(x, t["ln2"][layer], eps=1e-5)
            gate = xn @ t["wg"][layer].T
            up = xn @ t["wu"][layer].T
            x = x + (F.silu(gate) * up) @ t["wd"][layer].T
            if layer in feature_layers:
                features[layer] = x
        logits = rms(x, t["fnorm"], eps=1e-5) @ t["lm_head"].T
        greedy = logits.argmax(-1)
        accepted = (greedy[:depth] == drafts).to(torch.int64).cumprod(0).sum()
        # committed[:accepted] = drafts, committed[accepted] = greedy, else -1.
        padded = torch.cat((drafts, drafts.new_full((1,), -1)))
        committed = torch.where(slots < accepted, padded, padded.new_full((), -1))
        committed = torch.where(slots == accepted, greedy, committed)
        keep = (slots <= accepted)[None, :, None, None]
        target_features = torch.stack(
            [_row(features[index], accepted) for index in feature_layers]).to(torch.bfloat16)
        return {
            "logits": logits.float(),
            "accepted_count": accepted.to(torch.int64),
            "committed_count": (accepted + 1).to(torch.int64),
            "committed_tokens": committed,
            "cache_length": (accepted + context + 1).to(torch.int64),
            "target_features": target_features,
            "k_write": torch.stack(k_writes).masked_fill(~keep, 0),
            "v_write": torch.stack(v_writes).masked_fill(~keep, 0),
        }

    return step


_BUILDERS = {"dense_step": _dense, "moe_step": _moe, "quant_step": _gemma,
             "spec_target_step": _eagle3}


def build(case: Case, device) -> Step:
    """Return a capture-safe eager step for one ready P0 case."""
    if case.family not in _BUILDERS:
        raise NotImplementedError(f"no capture-safe baseline for {case.family}")
    return _BUILDERS[case.family](case, device)


def _capture(fn: Step, inputs: Inputs, device) -> tuple[Step, dict]:
    """Capture ``fn(inputs)`` into one graph after a side-stream warmup."""
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(inputs)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = fn(inputs)

    def replay(_: Inputs) -> dict[str, torch.Tensor]:
        graph.replay()
        return static

    return replay, static


def timing_baselines(case: Case, inputs: Inputs, device: str, warmup: int,
                     reps: int, *, expected: dict, exact: dict) -> dict:
    """Time capture-safe eager, graph, compiled, and compiled+graph baselines.

    Each variant must pass the same comparison a candidate passes against
    ``expected`` (BF16 reference) and ``exact`` (FP32 oracle) on ``inputs``.
    A variant that fails to build, capture, compile, or match is reported
    with its error and excluded from ``best``.
    """
    result: dict = {"compile_mode": COMPILE_MODE, "torch": torch.__version__,
                    "variants": {}}
    eager = build(case, device)
    compiled_fn: list[Step] = []

    def compiled() -> Step:
        # One compile serves both the plain and the graph-replayed variant.
        if not compiled_fn:
            compiled_fn.append(torch.compile(eager, mode=COMPILE_MODE,
                                             fullgraph=True, dynamic=False))
        return compiled_fn[0]

    plans: dict[str, Callable[[], tuple[Step, float]]] = {}

    def plain(make: Callable[[], Step]) -> Callable[[], tuple[Step, float]]:
        def setup() -> tuple[Step, float]:
            start = time.perf_counter()
            fn = make()
            fn(inputs)
            torch.cuda.synchronize(device)
            return fn, (time.perf_counter() - start) * 1000
        return setup

    def graphed(make: Callable[[], Step]) -> Callable[[], tuple[Step, float]]:
        def setup() -> tuple[Step, float]:
            start = time.perf_counter()
            fn, _ = _capture(make(), inputs, device)
            return fn, (time.perf_counter() - start) * 1000
        return setup

    plans["capture_safe_eager"] = plain(lambda: eager)
    plans["capture_safe_graph"] = graphed(lambda: eager)
    plans["torch_compile"] = plain(compiled)
    plans["torch_compile_graph"] = graphed(compiled)

    best_name, best_ms = None, math.inf
    for name, setup in plans.items():
        entry: dict = {}
        try:
            fn, setup_ms = setup()
            entry["setup_ms"] = setup_ms
            entry["correctness"] = _compare(expected, fn(inputs), case, device, exact)
            entry["timing"] = _measure(fn, inputs, device, warmup, reps)
            p50 = entry["timing"]["cuda_event_p50_ms"]
            if p50 < best_ms:
                best_name, best_ms = name, p50
        except Exception as exc:  # report and continue with other variants
            entry["unavailable"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        result["variants"][name] = entry
    if best_name is not None:
        result["best"] = {"variant": best_name, "cuda_event_p50_ms": best_ms}
    return result
