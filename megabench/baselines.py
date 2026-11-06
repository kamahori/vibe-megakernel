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

import hashlib
import json
import math
import os
import time
from pathlib import Path
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


_TRITON: dict[str, Callable] = {}


def _triton_kernels() -> dict[str, Callable]:
    """Triton kernels that read INT8 weights with one FP32 scale per row.

    ``gemm`` computes ``(x @ w.T) * scale``: the weights convert to BF16 in
    registers and the scale applies to the FP32 accumulator. Few rows leave
    few output tiles, so K splits across programs that add into FP32. ``dequant``
    writes ``(w * scale)`` in BF16 for a cuBLAS GEMM at prefill sizes.
    """
    if _TRITON:
        return _TRITON
    import triton
    import triton.language as tl

    configs = [triton.Config({"BLOCK_M": 16, "BLOCK_N": n, "BLOCK_K": k, "SPLIT_K": split},
                             num_warps=w, num_stages=st)
               for n, k, w, st in ((16, 256, 4, 4), (32, 256, 4, 4), (64, 128, 4, 4),
                                   (32, 512, 4, 3), (64, 256, 4, 3), (128, 128, 8, 3))
               for split in (1, 2, 4, 8)]

    @triton.autotune(configs=configs, key=["M", "N", "K"], reset_to_zero=["out"])
    @triton.jit
    def gemm_kernel(x, w, scale, out, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
            a = tl.load(x + rm[:, None] * K + (k0 + rk)[None, :],
                        mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
            b = tl.load(w + rn[None, :] * K + (k0 + rk)[:, None],
                        mask=(rn[None, :] < N) & ((k0 + rk)[:, None] < K), other=0)
            acc = tl.dot(a, b.to(tl.bfloat16), acc)
        acc = acc * tl.load(scale + rn, mask=rn < N, other=0.0)[None, :]
        tl.atomic_add(out + rm[:, None] * N + rn[None, :], acc,
                      mask=(rm[:, None] < M) & (rn[None, :] < N), sem="relaxed")

    @triton.jit
    def dequant_kernel(w, scale, out, total, K, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        q = tl.load(w + offsets, mask=mask, other=0).to(tl.float32)
        s = tl.load(scale + offsets // K, mask=mask, other=0.0)
        tl.store(out + offsets, (q * s).to(tl.bfloat16), mask=mask)

    def gemm(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        M, (N, K) = x.shape[0], w.shape
        out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]),
                             meta["SPLIT_K"])
        gemm_kernel[grid](x, w, scale, out, M, N, K)
        return out.to(torch.bfloat16)

    def dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        out = torch.empty(w.shape, device=w.device, dtype=torch.bfloat16)
        dequant_kernel[(triton.cdiv(w.numel(), 4096),)](w, scale, out, w.numel(), w.shape[1],
                                                        BLOCK=4096)
        return out

    _TRITON.update(gemm=gemm, dequant=dequant)
    return _TRITON


# Rows at or above this count use dequant + cuBLAS; fewer use the fused GEMM.
_W8A16_CUBLAS_ROWS = 256


@torch.library.custom_op("megabench::w8a16_linear", mutates_args=())
def _w8a16_linear(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    kernels = _triton_kernels()
    x = x.contiguous()
    if x.shape[0] >= _W8A16_CUBLAS_ROWS:
        return x @ kernels["dequant"](w, scale).T
    return kernels["gemm"](x, w, scale)


@_w8a16_linear.register_fake
def _(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


# FlashInfer calls are opaque to torch.compile; each built step registers its
# attention functions here and calls them through one custom op.
_ATTENTION: dict[int, Callable] = {}


@torch.library.custom_op("megabench::gemma_attention", mutates_args=())
def _gemma_attention(key: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     kcache: torch.Tensor, vcache: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention output ``[rows, heads, dim]`` and its base-2 log-sum-exp."""
    return _ATTENTION[key](q, k, v, kcache, vcache)


@_gemma_attention.register_fake
def _(key: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
      kcache: torch.Tensor, vcache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), q.new_empty(q.shape[:2], dtype=torch.float32)


def _gemma(case: Case, device) -> Step:
    """Gemma 3 quantized decode (batch one or ragged batch) or prefill chunk.

    On CUDA, each INT8 projection is one ``megabench::w8a16_linear`` call
    (the weights are read as INT8), and FlashInfer attention reads only the
    visible cache positions. The KV lengths of the case are planned when the
    step is built, as a serving engine plans each batch. Both run as custom
    ops, so torch.compile fuses the norms, RoPE and residuals around them.
    """
    from .tasks.gemma import _batched, _prefill, dequant

    p = case.params
    h, d, context, bits, window = (p["hidden"], p["head_dim"], p["context"], p["bits"],
                                   p["local_window"])
    qh, kvh = p["q_heads"], p["kv_heads"]
    batched, prefill = _batched(case), _prefill(case)
    if batched and prefill:
        raise NotImplementedError("Gemma 3 prefill handles one sequence")
    attn_scale = p.get("query_pre_attn_scalar", d) ** -0.5
    embed_scale = torch.full((), math.sqrt(h), device=device, dtype=torch.bfloat16)
    lengths = (case.contexts or (context,) * p["batch"]) if batched else (context,)
    rows = p["tokens"] if prefill else len(lengths)
    positions = (context + torch.arange(rows, device=device) if prefill else
                 torch.tensor(lengths, device=device))
    cuda = torch.device(device).type == "cuda"
    keys: dict[bool, int] = {}
    if cuda:
        import flashinfer

        def register(fn: Callable) -> int:
            key = len(_ATTENTION)
            _ATTENTION[key] = fn
            return key

        if prefill:
            def prefill_attention(local: bool) -> Callable:
                # The batch kernel skips tiles outside the window; the single
                # prefill kernel does not.
                wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    torch.empty(256 << 20, dtype=torch.uint8, device=device), "NHD",
                    backend="fa2")
                wrapper.plan(torch.tensor([0, rows], dtype=torch.int32, device=device),
                             torch.tensor([0, context + rows], dtype=torch.int32, device=device),
                             qh, kvh, d, causal=True, window_left=window - 1 if local else -1,
                             sm_scale=attn_scale, q_data_type=torch.bfloat16,
                             kv_data_type=torch.bfloat16)

                def run(q, k, v, kcache, vcache):
                    return wrapper.run(q, torch.cat((kcache, k), 0), torch.cat((vcache, v), 0),
                                       return_lse=True)
                return run

            keys = {local: register(prefill_attention(local)) for local in (False, True)}
        else:
            # One page per sequence: the padded cache row of that sequence.
            count = len(lengths)
            shape = (count, context, kvh, d)

            def decode_attention(local: bool) -> Callable:
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    torch.empty(128 << 20, dtype=torch.uint8, device=device), "NHD",
                    use_cuda_graph=True, use_tensor_cores=True,
                    paged_kv_indptr_buffer=torch.arange(count + 1, dtype=torch.int32,
                                                        device=device),
                    paged_kv_indices_buffer=torch.arange(count, dtype=torch.int32, device=device),
                    paged_kv_last_page_len_buffer=torch.tensor(lengths, dtype=torch.int32,
                                                               device=device))
                # The query sits after the last cached position; its own key is merged below.
                wrapper.plan(torch.arange(count + 1, dtype=torch.int32, device=device),
                             torch.arange(count, dtype=torch.int32, device=device),
                             torch.tensor(lengths, dtype=torch.int32, device=device),
                             qh, kvh, d, context, window_left=window - 2 if local else -1,
                             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                             sm_scale=attn_scale)

                def run(q, k, v, kcache, vcache):
                    return wrapper.run(q, (kcache.view(shape), vcache.view(shape)),
                                       return_lse=True)
                return run

            keys = {local: register(decode_attention(local)) for local in (False, True)}

    def masked_attention(t, layer, local, q, k, v):
        # Off CUDA: the plain masked attention of the grading reference.
        if prefill:
            keys_ = torch.cat((t["kcache"][layer], k), 0)
            values = torch.cat((t["vcache"][layer], v), 0)
            slots = torch.arange(keys_.shape[0], device=q.device)[None]
            allowed = slots <= positions[:, None]
            if local:
                allowed = allowed & (slots > positions[:, None] - window)
            return attention(q, keys_, values, scale=attn_scale, allowed=allowed)
        shape = (len(lengths), context, kvh, d)
        caches = t["kcache"][layer].view(shape), t["vcache"][layer].view(shape)
        outs = []
        for b, n in enumerate(lengths):
            start = max(0, n + 1 - window) if local else 0
            keys_ = torch.cat((caches[0][b, start:n], k[b:b + 1]), 0)
            values = torch.cat((caches[1][b, start:n], v[b:b + 1]), 0)
            outs.append(attention(q[b], keys_, values, scale=attn_scale))
        return torch.stack(outs)

    def linear(t: Inputs, name: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        if bits == 16:
            return x @ t[name][layer].T
        if bits == 8 and cuda:
            return _w8a16_linear(x, t[name][layer], t[f"{name}_scale"][layer])
        return x @ dequant(t[name][layer], t[f"{name}_scale"][layer], bits).to(x.dtype).T

    def rope_tables(lens: torch.Tensor, theta: float, factor: float):
        # The same rotation as tasks.common.rope, built once per step.
        inv = theta ** (-torch.arange(0, d, 2, device=lens.device, dtype=torch.float64) / d)
        phase = torch.outer(lens.to(torch.float64) / factor, inv)
        phase = torch.cat((phase, phase), dim=-1)
        return (phase.cos().float().to(torch.bfloat16)[:, None],
                phase.sin().float().to(torch.bfloat16)[:, None])

    def rotate(x: torch.Tensor, table) -> torch.Tensor:
        cos, sin = table
        half = d // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    def merge_own_key(out, lse, q, k, v):
        # The decode query also sees its own new key. FlashInfer's LSE is base 2.
        own = (q.float().view(-1, kvh, qh // kvh, d) * k.float()[:, :, None]).sum(-1)
        own = own.view(-1, qh) * (attn_scale * math.log2(math.e))
        top = torch.maximum(lse, own)
        old, new = torch.exp2(lse - top), torch.exp2(own - top)
        value = v.float().repeat_interleave(qh // kvh, dim=1)
        merged = (out.float() * old[..., None] + value * new[..., None]) / (old + new)[..., None]
        return merged.to(q.dtype)

    def step(t: Inputs) -> dict[str, torch.Tensor]:
        ids = t["tokens"] if prefill else t["token"].reshape(-1)
        x = t["embed"].index_select(0, ids) * embed_scale
        lens = t["context_lens"].long() if batched else positions
        tables = {True: rope_tables(lens, 10_000.0, 1.0),
                  False: rope_tables(lens, 1_000_000.0, 8.0)}
        k_writes, v_writes = [], []
        for layer in range(p["layers"]):
            xn = rms(x, t["ln1"][layer], gemma=True)
            q = rms(linear(t, "wq", layer, xn).view(rows, qh, d), t["qn"][layer], gemma=True)
            k = rms(linear(t, "wk", layer, xn).view(rows, kvh, d), t["kn"][layer], gemma=True)
            v = linear(t, "wv", layer, xn).view(rows, kvh, d)
            local = (layer + 1) % 6 != 0
            q, k = rotate(q, tables[local]), rotate(k, tables[local])
            k_writes.append(k)
            v_writes.append(v)
            if cuda:
                attn, lse = _gemma_attention(keys[local], q, k, v, t["kcache"][layer],
                                             t["vcache"][layer])
                if not prefill:
                    attn = merge_own_key(attn, lse, q, k, v)
            else:
                attn = masked_attention(t, layer, local, q, k, v)
            x = x + rms(linear(t, "wo", layer, attn.reshape(rows, qh * d)),
                        t["ln2"][layer], gemma=True)
            xn = rms(x, t["ln3"][layer], gemma=True)
            gate = linear(t, "wg", layer, xn)
            up = linear(t, "wu", layer, xn)
            mlp = linear(t, "wd", layer, F.gelu(gate, approximate="tanh") * up)
            x = x + rms(mlp, t["ln4"][layer], gemma=True)
        last = x[-1:] if prefill else x
        logits = (rms(last, t["fnorm"], gemma=True) @ t["embed"].T).float()
        k_write, v_write = torch.stack(k_writes), torch.stack(v_writes)
        if not batched:
            logits = logits[0]
            if not prefill:
                k_write, v_write = k_write[:, 0], v_write[:, 0]
        return {"logits": logits, "next_token": logits.argmax(-1).to(torch.int64),
                "k_write": k_write.contiguous(), "v_write": v_write.contiguous()}

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
                     reps: int, *, expected: dict, exact: dict,
                     variants: tuple[str, ...] | None = None) -> dict:
    """Time capture-safe eager, graph, compiled, and compiled+graph baselines.

    Each variant must pass the same comparison a candidate passes against
    ``expected`` (BF16 reference) and ``exact`` (FP32 oracle) on ``inputs``.
    A variant that fails to build, capture, compile, or match is reported
    with its error and excluded from ``best``. ``variants`` limits the run
    to those names; None runs all four.
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
    if variants is not None:
        plans = {name: plans[name] for name in variants}

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


def baseline_cache_dir() -> Path:
    return Path(os.environ.get("MEGABENCH_BASELINE_CACHE",
                               Path.home() / ".cache/megabench/baselines"))


def cached_timing_baselines(case: Case, inputs: Inputs, device: str, warmup: int, reps: int,
                            *, expected: dict, exact: dict,
                            variants: tuple[str, ...] | None = None) -> dict:
    """``timing_baselines``, reused across processes for the same contract.

    Compiling a whole-model baseline takes minutes, and every evaluation of a
    case times the same baseline. The key covers the case, the variants, the
    timing loop, the GPU, the library versions and the baseline and task
    sources, so any change times the baseline again. Set
    ``MEGABENCH_BASELINE_CACHE`` to choose the directory.
    """
    import importlib

    from .tasks.workloads import _module

    versions = {"torch": torch.__version__}
    for name in ("triton", "flashinfer"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError:
            versions[name] = None
    sources = [Path(__file__), Path(_module(case).__file__)]
    key = hashlib.sha256(json.dumps({
        "case": case.to_dict(), "variants": variants, "warmup": warmup, "reps": reps,
        "gpu": torch.cuda.get_device_name(device), "versions": versions,
        "sources": [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources],
    }, sort_keys=True).encode()).hexdigest()
    path = baseline_cache_dir() / f"{case.id}-{key[:16]}.json"
    try:
        return json.loads(path.read_text()) | {"cached_from": str(path)}
    except (OSError, ValueError):
        pass
    result = timing_baselines(case, inputs, device, warmup, reps, expected=expected,
                              exact=exact, variants=variants)
    if "best" in result:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(result, default=str))
            temporary.replace(path)
        except OSError:
            pass
    return result
