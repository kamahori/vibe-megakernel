"""Experimental full-step MegaBench baseline using MPK for vocabulary projection.

This is a relaxed-launch hybrid baseline. MPK executes a chunked BF16
vocabulary projection, with selective FP32 row refinement to meet the strict
logit tolerance; PyTorch executes the remaining model operations. The
full output contract is checked by the separate relaxed evaluator. No oracle
module is imported by this candidate.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import torch


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_tables(length: int, dim: int, theta: float, device: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    positions = torch.arange(length, dtype=torch.float64)
    phase = torch.outer(positions, inv_freq)
    embedding = torch.cat((phase, phase), dim=-1)
    return embedding.cos().float().to(device), embedding.sin().float().to(device)


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float,
          factor: float = 1.0) -> torch.Tensor:
    width = x.shape[-1]
    inv = theta ** (-torch.arange(0, width, 2, device=x.device,
                                  dtype=torch.float64) / width)
    phase = torch.outer(positions.to(torch.float64) / factor, inv)
    phase = torch.cat((phase, phase), dim=-1)
    cos, sin = phase.cos().float(), phase.sin().float()
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return x * cos + _rot_half(x) * sin


def _gemma_rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                   + 1e-6) * (1.0 + weight.float())


def _dequant(weight: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        values = weight.float()
    else:
        low = (weight & 15).to(torch.int16) - 8
        high = (weight >> 4).to(torch.int16) - 8
        values = torch.stack((low, high), dim=-1).flatten(-2).float()
    return values * scale.float().unsqueeze(-1)


def _linear_grid(rows: int) -> int:
    if rows % 96 == 0:
        return rows // 96
    if rows % 64 == 0:
        return rows // 64
    raise ValueError(f"MPK linear output rows must divide into 64/96: {rows}")


class MPKPairLinear:
    """Two BF16 projections from one input in a reusable MPK task graph."""

    def __init__(self, batch: int, hidden: int, intermediate: int):
        import mirage
        from mirage.mpk.persistent_kernel import PersistentKernel

        self.batch = batch
        self.hidden = hidden
        self.intermediate = intermediate
        device = "cuda"
        dtype = torch.bfloat16
        self.input = torch.empty((batch, hidden), device=device, dtype=dtype)
        self.gate_weight = torch.empty((intermediate, hidden), device=device, dtype=dtype)
        self.up_weight = torch.empty_like(self.gate_weight)
        self.gate_output = torch.empty((batch, intermediate), device=device, dtype=dtype)
        self.up_output = torch.empty_like(self.gate_output)
        workers, schedulers = mirage.get_configurations_from_gpu(0)
        params = PersistentKernel.get_default_init_parameters()
        params.update(test_mode=True, num_workers=workers,
                      num_local_schedulers=schedulers,
                      max_num_batched_tokens=batch,
                      max_num_batched_requests=batch,
                      max_num_pages=batch)
        self.pk = PersistentKernel(**params)
        x = self.pk.attach_input(self.input, name="input")
        wg = self.pk.attach_input(self.gate_weight, name="gate_weight")
        wu = self.pk.attach_input(self.up_weight, name="up_weight")
        gate = self.pk.attach_input(self.gate_output, name="gate_output")
        up = self.pk.attach_input(self.up_output, name="up_output")
        grid = (_linear_grid(intermediate), 1, 1)
        block = (256, 1, 1)
        self.pk.linear_layer(x, wg, gate, grid, block)
        self.pk.linear_layer(x, wu, up, grid, block)
        cache_root = Path(os.environ.get("MPK_BASELINE_CACHE", tempfile.gettempdir()))
        cache_root.mkdir(parents=True, exist_ok=True)
        self.compile_dir = Path(tempfile.mkdtemp(prefix="mpk_pair_", dir=cache_root))
        self.pk.compile(output_dir=str(self.compile_dir))
        self.reset = self.pk.init_func.__self__.init_request_func

    def __call__(self, x: torch.Tensor, gate_weight: torch.Tensor,
                 up_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.input.copy_(x.reshape(self.batch, self.hidden).to(torch.bfloat16))
        self.gate_weight.copy_(gate_weight.to(torch.bfloat16))
        self.up_weight.copy_(up_weight.to(torch.bfloat16))
        self.reset()
        self.pk()
        return self.gate_output.float(), self.up_output.float()


class MPKHead:
    """A reusable MPK BF16 vocabulary projection with copied input weights."""

    def __init__(self, batch: int, hidden: int, vocab: int):
        import mirage
        from mirage.mpk.persistent_kernel import PersistentKernel

        self.batch = batch
        self.hidden = hidden
        self.vocab = vocab
        self.input = torch.empty((batch, hidden), device="cuda", dtype=torch.bfloat16)
        self.weight = torch.empty((vocab, hidden), device="cuda", dtype=torch.bfloat16)
        self.output = torch.empty((batch, vocab), device="cuda", dtype=torch.bfloat16)
        workers, schedulers = mirage.get_configurations_from_gpu(0)
        params = PersistentKernel.get_default_init_parameters()
        params.update(test_mode=True, num_workers=workers,
                      num_local_schedulers=schedulers,
                      max_num_batched_tokens=batch,
                      max_num_batched_requests=batch,
                      max_num_pages=batch)
        self.pk = PersistentKernel(**params)
        x = self.pk.attach_input(self.input, name="input")
        w = self.pk.attach_input(self.weight, name="weight")
        y = self.pk.attach_input(self.output, name="output")
        self.pk.linear_layer(x, w, y, (_linear_grid(vocab), 1, 1), (256, 1, 1))
        cache_root = Path(os.environ.get("MPK_BASELINE_CACHE", tempfile.gettempdir()))
        cache_root.mkdir(parents=True, exist_ok=True)
        self.compile_dir = Path(tempfile.mkdtemp(prefix="mpk_head_", dir=cache_root))
        self.pk.compile(output_dir=str(self.compile_dir))
        self.reset = self.pk.init_func.__self__.init_request_func

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self.input.copy_(x.reshape(self.batch, self.hidden).to(torch.bfloat16))
        self.weight.copy_(weight.to(torch.bfloat16))
        self.reset()
        self.pk()
        return self.output.float()


class MPKCompensatedHead:
    """MPK BF16 partial GEMVs summed in FP32 for strict logits tolerance."""

    def __init__(self, batch: int, hidden: int, vocab: int, chunks: int = 8,
                 atol: float = 0.002, rtol: float = 0.002,
                 refine_top: bool = False):
        import mirage
        from mirage.mpk.persistent_kernel import PersistentKernel

        if hidden % chunks:
            raise ValueError("hidden size must divide into MPK reduction chunks")
        self.batch, self.hidden, self.vocab = batch, hidden, vocab
        self.chunks = chunks
        self.chunk_hidden = hidden // chunks
        self.atol, self.rtol = atol, rtol
        self.refine_top = refine_top
        self.last_fallback_fraction = 0.0
        self.diagnostics: list[dict] = []
        workers, schedulers = mirage.get_configurations_from_gpu(0)
        params = PersistentKernel.get_default_init_parameters()
        params.update(test_mode=True, num_workers=workers,
                      num_local_schedulers=schedulers,
                      max_num_batched_tokens=batch,
                      max_num_batched_requests=batch,
                      max_num_pages=batch)
        self.pk = PersistentKernel(**params)
        self.inputs_hi = []
        self.inputs_lo = []
        self.weights = []
        self.outputs_hi = []
        self.outputs_lo = []
        grid = (_linear_grid(vocab), 1, 1)
        block = (256, 1, 1)
        for chunk in range(chunks):
            x_hi = torch.empty((batch, self.chunk_hidden), device="cuda",
                               dtype=torch.bfloat16)
            x_lo = torch.empty_like(x_hi)
            weight = torch.empty((vocab, self.chunk_hidden), device="cuda",
                                 dtype=torch.bfloat16)
            y_hi = torch.empty((batch, vocab), device="cuda", dtype=torch.bfloat16)
            y_lo = torch.empty_like(y_hi)
            self.inputs_hi.append(x_hi)
            self.inputs_lo.append(x_lo)
            self.weights.append(weight)
            self.outputs_hi.append(y_hi)
            self.outputs_lo.append(y_lo)
            hi_dt = self.pk.attach_input(x_hi, name=f"x_hi_{chunk}")
            lo_dt = self.pk.attach_input(x_lo, name=f"x_lo_{chunk}")
            weight_dt = self.pk.attach_input(weight, name=f"weight_{chunk}")
            out_hi_dt = self.pk.attach_input(y_hi, name=f"y_hi_{chunk}")
            out_lo_dt = self.pk.attach_input(y_lo, name=f"y_lo_{chunk}")
            self.pk.linear_layer(hi_dt, weight_dt, out_hi_dt, grid, block)
            self.pk.linear_layer(lo_dt, weight_dt, out_lo_dt, grid, block)
        cache_root = Path(os.environ.get("MPK_BASELINE_CACHE", tempfile.gettempdir()))
        cache_root.mkdir(parents=True, exist_ok=True)
        self.compile_dir = Path(tempfile.mkdtemp(prefix="mpk_comp_head_", dir=cache_root))
        self.pk.compile(output_dir=str(self.compile_dir))
        self.reset = self.pk.init_func.__self__.init_request_func

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = x.reshape(self.batch, self.hidden).float()
        for chunk in range(self.chunks):
            start = chunk * self.chunk_hidden
            end = start + self.chunk_hidden
            hi = x[:, start:end].to(torch.bfloat16)
            self.inputs_hi[chunk].copy_(hi)
            self.inputs_lo[chunk].copy_((x[:, start:end] - hi.float()).to(torch.bfloat16))
            self.weights[chunk].copy_(weight[:, start:end].to(torch.bfloat16))
        self.reset()
        self.pk()
        output = torch.zeros((self.batch, self.vocab), device="cuda",
                             dtype=torch.float32)
        abs_sum = torch.zeros_like(output)
        for hi, lo in zip(self.outputs_hi, self.outputs_lo):
            hi32, lo32 = hi.float(), lo.float()
            output += hi32 + lo32
            abs_sum += hi32.abs() + lo32.abs()
        if os.getenv("MPK_DIAGNOSTICS") == "1":
            exact = x @ weight.float().T
            tolerance_exact = self.atol + self.rtol * exact.abs()
            mismatch = (output - exact).abs() > tolerance_exact
            candidates = []
            for coefficient in (0.001, 0.002, 0.0025, 0.003, 0.0035, 0.004):
                mask = coefficient * abs_sum + 0.0002 > 0.9 * tolerance_exact
                candidates.append({
                    "coefficient": coefficient,
                    "fallback_fraction": float(mask.float().mean().item()),
                    "uncovered_mismatches": int((mismatch & ~mask).sum().item()),
                })
            self.diagnostics.append({
                "raw_mismatches": int(mismatch.sum().item()),
                "raw_max_abs_error": float((output - exact).abs().max().item()),
                "candidates": candidates,
            })
        # BF16 stores each partial output. Refine rows whose worst-case store
        # rounding can consume the MegaBench logit tolerance. Most rows stay
        # on MPK's result; the fallback uses an exact FP32 dot product.
        tolerance = self.atol + self.rtol * output.abs()
        uncertain = 0.0025 * abs_sum + 0.0002 > 0.9 * tolerance
        if self.refine_top:
            uncertain |= output >= output.amax(dim=-1, keepdim=True) - 0.05
        self.last_fallback_fraction = float(uncertain.float().mean().item())
        for row in range(self.batch):
            indices = uncertain[row].nonzero().flatten()
            if indices.numel():
                refined = weight.index_select(0, indices).float() @ x[row]
                output[row, indices] = refined
        return output


class DenseQwen3Hybrid:
    def __init__(self, case: dict):
        p = case["params"]
        self.p = p
        self.head = MPKCompensatedHead(
            p["batch"], p["hidden"], p["vocab"],
            atol=case["atol"], rtol=case["rtol"])
        self.cos, self.sin = _rope_tables(
            p["context"] + 1, p["head_dim"], 1e6, torch.device("cuda"))

    def __call__(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.p
        context, dim, qheads, kvheads = (
            p["context"], p["head_dim"], p["q_heads"], p["kv_heads"])
        token = int(t["token"].item())
        x = t["embed"][token].float().clone()
        k_writes: list[torch.Tensor] = []
        v_writes: list[torch.Tensor] = []
        cos, sin = self.cos[context], self.sin[context]
        for layer in range(p["layers"]):
            normed = _rms(x, t["ln1"][layer])
            q = (t["wq"][layer].float() @ normed).view(qheads, dim)
            k = (t["wk"][layer].float() @ normed).view(kvheads, dim)
            v = (t["wv"][layer].float() @ normed).view(kvheads, dim)
            q = _rms(q, t["qn"][layer])
            k = _rms(k, t["kn"][layer])
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            keys = torch.cat((t["kcache"][layer].float(), k.unsqueeze(0)))
            vals = torch.cat((t["vcache"][layer].float(), v.unsqueeze(0)))
            grouped = q.view(kvheads, qheads // kvheads, dim)
            scores = torch.einsum("gqd,tgd->gqt", grouped, keys) * dim ** -0.5
            probs = scores.softmax(-1)
            attn = torch.einsum("gqt,tgd->gqd", probs, vals).reshape(qheads * dim)
            x = x + t["wo"][layer].float() @ attn
            normed = _rms(x, t["ln2"][layer])
            gate = t["wg"][layer].float() @ normed
            up = t["wu"][layer].float() @ normed
            activated = torch.nn.functional.silu(gate) * up
            x = x + t["wd"][layer].float() @ activated
        logits = self.head(_rms(x, t["fnorm"]), t["embed"])[0]
        return {
            "logits": logits.float(),
            "next_token": logits.argmax().to(torch.int64),
            "k_write": torch.stack(k_writes),
            "v_write": torch.stack(v_writes),
        }


class MoeQwen3Hybrid:
    def __init__(self, case: dict):
        p = case["params"]
        self.p = p
        self.head = MPKCompensatedHead(
            p["batch"], p["hidden"], p["vocab"],
            atol=case["atol"], rtol=case["rtol"])
        self.cos, self.sin = _rope_tables(
            p["context"] + 1, p["head_dim"], 1e6, torch.device("cuda"))

    def __call__(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.p
        context, dim, qheads, kvheads = (
            p["context"], p["head_dim"], p["q_heads"], p["kv_heads"])
        x = t["embed"][int(t["token"].item())].float()
        k_writes: list[torch.Tensor] = []
        v_writes: list[torch.Tensor] = []
        cos, sin = self.cos[context], self.sin[context]
        for layer in range(p["layers"]):
            normed = _rms(x, t["ln1"][layer])
            q = _rms((t["wq"][layer].float() @ normed).view(qheads, dim),
                     t["qn"][layer])
            k = _rms((t["wk"][layer].float() @ normed).view(kvheads, dim),
                     t["kn"][layer])
            v = (t["wv"][layer].float() @ normed).view(kvheads, dim)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            keys = torch.cat((t["kcache"][layer].float(), k.unsqueeze(0)))
            vals = torch.cat((t["vcache"][layer].float(), v.unsqueeze(0)))
            grouped = q.view(kvheads, qheads // kvheads, dim)
            scores = torch.einsum("gqd,tgd->gqt", grouped, keys) * dim ** -0.5
            probs = scores.softmax(-1)
            attn = torch.einsum("gqt,tgd->gqd", probs, vals).reshape(qheads * dim)
            x = x + t["wo"][layer].float() @ attn
            normed = _rms(x, t["ln2"][layer])
            routing = t["wrt"][layer].float() @ normed
            top_values, top_indices = torch.topk(routing, p["topk"])
            top_probs = torch.softmax(top_values, dim=-1)
            accumulated = torch.zeros_like(x)
            for slot in range(p["topk"]):
                expert = int(top_indices[slot].item())
                gate = t["wg"][layer, expert].float() @ normed
                up = t["wu"][layer, expert].float() @ normed
                activated = torch.nn.functional.silu(gate) * up * top_probs[slot]
                accumulated = accumulated + t["wd"][layer, expert].float() @ activated
            x = x + accumulated
        logits = self.head(_rms(x, t["fnorm"]), t["lm_head"])[0]
        return {
            "logits": logits.float(),
            "next_token": logits.argmax().to(torch.int64),
            "k_write": torch.stack(k_writes),
            "v_write": torch.stack(v_writes),
        }


class GemmaHybrid:
    def __init__(self, case: dict):
        self.p = case["params"]
        p = self.p
        self.head = MPKCompensatedHead(
            p["batch"], p["hidden"], p["vocab"],
            atol=case["atol"], rtol=case["rtol"])

    def __call__(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.p
        hidden, dim, context, bits = (
            p["hidden"], p["head_dim"], p["context"], p["bits"])
        qheads, kvheads = p["q_heads"], p["kv_heads"]
        x = t["embed"][t["token"]].float() * math.sqrt(hidden)
        positions = torch.tensor([context], device=x.device)
        k_writes: list[torch.Tensor] = []
        v_writes: list[torch.Tensor] = []

        def linear(name: str, layer: int, vector: torch.Tensor) -> torch.Tensor:
            matrix = _dequant(t[name][layer], t[f"{name}_scale"][layer], bits)
            return matrix @ vector

        for layer in range(p["layers"]):
            normed = _gemma_rms(x, t["ln1"][layer])
            q = linear("wq", layer, normed).view(qheads, dim)
            k = linear("wk", layer, normed).view(kvheads, dim)
            v = linear("wv", layer, normed).view(kvheads, dim)
            q = _gemma_rms(q, t["qn"][layer])
            k = _gemma_rms(k, t["kn"][layer])
            local = (layer + 1) % 6 != 0
            theta = 10_000.0 if local else 1_000_000.0
            factor = 1.0 if local else 8.0
            q = _rope(q, positions, theta, factor)
            k = _rope(k, positions, theta, factor)
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            start = max(0, context + 1 - p["local_window"]) if local else 0
            keys = torch.cat((t["kcache"][layer, start:].float(), k[None]), dim=0)
            vals = torch.cat((t["vcache"][layer, start:].float(), v[None]), dim=0)
            grouped = q.view(kvheads, qheads // kvheads, dim)
            scores = torch.einsum("gqd,tgd->gqt", grouped, keys) * dim ** -0.5
            probs = scores.softmax(-1)
            attn = torch.einsum("gqt,tgd->gqd", probs, vals).reshape(qheads * dim)
            x = x + _gemma_rms(linear("wo", layer, attn), t["ln2"][layer])
            normed = _gemma_rms(x, t["ln3"][layer])
            gate = linear("wg", layer, normed)
            up = linear("wu", layer, normed)
            activated = torch.nn.functional.gelu(gate, approximate="tanh") * up
            mlp = linear("wd", layer, activated)
            x = x + _gemma_rms(mlp, t["ln4"][layer])
        logits = self.head(_gemma_rms(x, t["fnorm"]), t["embed"])[0]
        return {
            "logits": logits.float(),
            "next_token": logits.argmax().to(torch.int64),
            "k_write": torch.stack(k_writes),
            "v_write": torch.stack(v_writes),
        }


class EagleTargetHybrid:
    def __init__(self, case: dict):
        self.p = case["params"]
        p = self.p
        self.head = MPKCompensatedHead(
            p["draft_depth"] + 1, p["hidden"], p["vocab"],
            atol=case["atol"], rtol=case["rtol"], refine_top=True)

    def __call__(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p = self.p
        depth, context, dim = p["draft_depth"], p["context"], p["head_dim"]
        qheads, kvheads = p["q_heads"], p["kv_heads"]
        token_ids = torch.cat((t["token"].reshape(1), t["draft_tokens"]))
        steps = depth + 1
        x = t["embed"][token_ids].float()
        positions = torch.arange(context, context + steps, device=x.device)
        future_mask = (torch.arange(context + steps, device=x.device)[None, :]
                       > positions[:, None])
        k_writes: list[torch.Tensor] = []
        v_writes: list[torch.Tensor] = []
        layer_hidden: list[torch.Tensor] = []
        for layer in range(p["layers"]):
            normed = _rms(x, t["ln1"][layer], eps=1e-5)
            q = (normed @ t["wq"][layer].float().T).view(steps, qheads, dim)
            k = (normed @ t["wk"][layer].float().T).view(steps, kvheads, dim)
            v = (normed @ t["wv"][layer].float().T).view(steps, kvheads, dim)
            q = _rope(q, positions, 500_000.0)
            k = _rope(k, positions, 500_000.0)
            k_writes.append(k.to(torch.bfloat16))
            v_writes.append(v.to(torch.bfloat16))
            keys = torch.cat((t["kcache"][layer].float(), k), dim=0)
            vals = torch.cat((t["vcache"][layer].float(), v), dim=0)
            grouped = q.view(steps, kvheads, qheads // kvheads, dim)
            scores = torch.einsum("tgqd,sgd->tgqs", grouped, keys) * dim ** -0.5
            scores = scores.masked_fill(future_mask[:, None, None, :], -torch.inf)
            probs = scores.softmax(-1)
            attn = torch.einsum("tgqs,sgd->tgqd", probs, vals).reshape(
                steps, qheads * dim)
            x = x + attn @ t["wo"][layer].float().T
            normed = _rms(x, t["ln2"][layer], eps=1e-5)
            gate = normed @ t["wg"][layer].float().T
            up = normed @ t["wu"][layer].float().T
            x = x + (torch.nn.functional.silu(gate) * up) @ t["wd"][layer].float().T
            layer_hidden.append(x)

        logits = self.head(_rms(x, t["fnorm"], eps=1e-5), t["lm_head"])
        greedy = logits.argmax(-1)
        accepted = (greedy[:depth] == t["draft_tokens"]).to(torch.int64).cumprod(0).sum()
        count = int(accepted.item())
        committed = torch.full((steps,), -1, dtype=torch.int64, device=x.device)
        if count:
            committed[:count] = t["draft_tokens"][:count]
        committed[count] = greedy[count]
        keep = (torch.arange(steps, device=x.device) <= accepted)[None, :, None, None]
        feature_layers = (max(0, p["layers"] // 4 - 1),
                          max(0, p["layers"] // 2 - 1), p["layers"] - 1)
        target_features = torch.stack([layer_hidden[index][accepted]
                                       for index in feature_layers]).to(torch.bfloat16)
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


def build(case: dict):
    if case["id"] == "dense-step-qwen3-06b-b1-s128":
        return DenseQwen3Hybrid(case)
    if case["id"] == "moe-step-qwen3-30b-a3b-b1-s128":
        return MoeQwen3Hybrid(case)
    if case["family"] == "quant_step" and case["model"] == "google/gemma-3-4b-it":
        return GemmaHybrid(case)
    if case["id"] == "spec-target-step-llama31-8b-k4":
        return EagleTargetHybrid(case)
    raise NotImplementedError(f"MPK hybrid candidate not implemented for {case['id']}")
