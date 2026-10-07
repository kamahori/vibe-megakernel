"""Attempt the native MPK Qwen3 model graph on a MegaBench decode fixture.

The one-pass MPK mode accepts an already-populated KV cache. This probe keeps
the upstream Qwen3 graph builder intact except for exposing its logits tensor
and argmax scratch buffers, which online_notoken normally omits.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from megabench.cases import select_cases
from megabench.harness.benchmark import _measure
from megabench.tasks.workloads import make_inputs, reference


def _truncated_reference(case, values: dict[str, torch.Tensor], layers: int
                         ) -> dict[str, torch.Tensor]:
    from megabench.tasks.references.qwen3 import Config, RefDecoder

    p = case.params
    config = Config(hidden=p["hidden"], layers=layers,
                    q_heads=p["q_heads"], kv_heads=p["kv_heads"],
                    head_dim=p["head_dim"], inter=p["intermediate"],
                    vocab=p["vocab"], max_seq=p["context"] + 1)
    weights = {key: value[:layers] for key, value in values.items()
               if key in ("ln1", "wq", "wk", "wv", "qn", "kn", "wo",
                          "ln2", "wg", "wu", "wd")}
    weights["embed"] = values["embed"]
    weights["fnorm"] = values["fnorm"]
    decoder = RefDecoder(config, weights)
    decoder.k_cache[:, :p["context"]] = values["kcache"][:layers].float()
    decoder.v_cache[:, :p["context"]] = values["vcache"][:layers].float()
    decoder.pos = p["context"]
    logits = decoder.step(int(values["token"].item()))
    return {"logits": logits.float(), "next_token": logits.argmax().to(torch.int64),
            "k_write": decoder.k_cache[:, p["context"]].to(torch.bfloat16),
            "v_write": decoder.v_cache[:, p["context"]].to(torch.bfloat16)}


def _one_layer_diagnostics(case, values: dict[str, torch.Tensor], builder
                           ) -> dict[str, dict[str, float]]:
    from mirage.mpk.models.utils import shuffle_tensors
    from megabench.tasks.references.qwen3 import Config, rope_tables

    p = case.params
    token = int(values["token"].item())
    x = values["embed"][token].float()
    rms = lambda a, w: a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-6) * w.float()
    normed = rms(x, values["ln1"][0])
    q = (values["wq"][0].float() @ normed).view(p["q_heads"], p["head_dim"])
    k = (values["wk"][0].float() @ normed).view(p["kv_heads"], p["head_dim"])
    v = (values["wv"][0].float() @ normed).view(p["kv_heads"], p["head_dim"])
    raw_qkv = shuffle_tensors([q.flatten(), k.flatten(), v.flatten()],
                              p["kv_heads"], 0)
    cos, sin = rope_tables(Config(max_seq=p["context"] + 1), device="cuda")
    half = p["head_dim"] // 2
    rotate = lambda a: torch.cat((-a[..., half:], a[..., :half]), dim=-1)
    q = rms(q, values["qn"][0])
    k = rms(k, values["kn"][0])
    q = q * cos[p["context"]] + rotate(q) * sin[p["context"]]
    k = k * cos[p["context"]] + rotate(k) * sin[p["context"]]
    keys = torch.cat((values["kcache"][0].float(), k[None]), dim=0)
    vals = torch.cat((values["vcache"][0].float(), v[None]), dim=0)
    grouped = q.view(p["kv_heads"], p["q_heads"] // p["kv_heads"], p["head_dim"])
    scores = torch.einsum("gqd,tgd->gqt", grouped, keys) * p["head_dim"] ** -0.5
    probs = scores.softmax(-1)
    attn = torch.einsum("gqt,tgd->gqd", probs, vals).reshape(-1)
    x = x + values["wo"][0].float() @ attn
    normed = rms(x, values["ln2"][0])
    gate = values["wg"][0].float() @ normed
    up = values["wu"][0].float() @ normed
    fused_gate_up = shuffle_tensors([gate, up], builder.mpk_gate_up_groups, 0)
    activated = torch.nn.functional.silu(gate) * up
    x = x + values["wd"][0].float() @ activated
    final_norm = rms(x, values["fnorm"])
    actual_gate_up = builder.mlp_mid_tensor[0].float().view(builder.mpk_gate_up_groups, -1)
    half_tile = actual_gate_up.shape[1] // 2
    local_activated = (torch.nn.functional.silu(actual_gate_up[:, :half_tile]) *
                       actual_gate_up[:, half_tile:]).flatten()
    actual = {
        "qkv": builder.attn_in_tensor[0].float(),
        "attention": builder.attn_out_tensor[0].float(),
        "gate_up": builder.mlp_mid_tensor[0].float(),
        "activated": builder.silu_mul_out_tensor[0].float(),
        "final_x": builder.y_tensor[0].float(),
        "final_norm": builder.returned_hidden_state[0].float(),
    }
    wanted = {"qkv": raw_qkv, "attention": attn,
              "gate_up": fused_gate_up, "activated": activated,
              "final_x": x, "final_norm": final_norm}
    wanted["activation_from_actual_gate_up"] = local_activated
    actual["activation_from_actual_gate_up"] = builder.silu_mul_out_tensor[0].float()
    return {name: {
        "max_abs_error": float((actual[name] - target).abs().max().item()),
        "mean_abs_error": float((actual[name] - target).abs().mean().item()),
        "actual_max_abs": float(actual[name].abs().max().item()),
        "expected_max_abs": float(target.abs().max().item()),
    } for name, target in wanted.items()}


def _rope_tables(length: int, dim: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1e6 ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    phase = torch.outer(torch.arange(length, dtype=torch.float64), inv)
    angles = torch.cat((phase, phase), dim=-1)
    return (angles.cos().to(device=device, dtype=torch.bfloat16)[None],
            angles.sin().to(device=device, dtype=torch.bfloat16)[None])


class NativeQwen3:
    def __init__(self, case, initial: dict[str, torch.Tensor], layers: int,
                 cache_root: Path):
        import mirage
        import mirage.mpk.models as model_package
        from mirage.mpk.models.graph_builder import MirageModelConfig
        # The installed wheel excludes qwen3/ because upstream omitted its
        # __init__.py; load the pinned source builder through the package path.
        source_models = (Path(__file__).resolve().parents[2] /
                         "reproductions/mirage-mpk/python/mirage/mpk/models")
        model_package.__path__.append(str(source_models))
        from mirage.mpk.models.qwen3.builder import Qwen3Builder
        from mirage.mpk.models.utils import (grid_for_rmsnorm_linear_layer,
                                             shuffle_tensors)
        from mirage.mpk.persistent_kernel import PersistentKernel

        self.case = case
        self.layers = layers
        self.shuffle_tensors = shuffle_tensors
        p = case.params
        self.context = p["context"]
        self.vocab = p["vocab"]
        self.hidden = p["hidden"]
        self.gate_up_groups = grid_for_rmsnorm_linear_layer(
            2 * p["intermediate"]) // 2
        self.kcache = torch.zeros((layers, 1, 256, p["kv_heads"], p["head_dim"]),
                                  device="cuda", dtype=torch.bfloat16)
        self.vcache = torch.zeros_like(self.kcache)
        self.state: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": initial["embed"].clone(),
            "lm_head.weight": initial["embed"].clone(),
            "model.norm.weight": initial["fnorm"].clone(),
        }
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            self.state[prefix + "input_layernorm.weight"] = initial["ln1"][layer].clone()
            self.state[prefix + "self_attn.qkv_proj.weight"] = shuffle_tensors(
                [initial[name][layer] for name in ("wq", "wk", "wv")],
                p["kv_heads"], 0)
            self.state[prefix + "self_attn.q_norm.weight"] = initial["qn"][layer].clone()
            self.state[prefix + "self_attn.k_norm.weight"] = initial["kn"][layer].clone()
            self.state[prefix + "self_attn.o_proj.weight"] = initial["wo"][layer].clone()
            self.state[prefix + "post_attention_layernorm.weight"] = initial["ln2"][layer].clone()
            # Native MPK's fused SiLU task expects gate/up tiles interleaved.
            self.state[prefix + "mlp.gate_up_proj.weight"] = shuffle_tensors(
                [initial["wg"][layer], initial["wu"][layer]],
                self.gate_up_groups, 0)
            self.state[prefix + "mlp.down_proj.weight"] = initial["wd"][layer].clone()

        self.meta = {
            "step": torch.full((1,), self.context, device="cuda", dtype=torch.int32),
            "tokens": torch.zeros((1, self.context + 1), device="cuda", dtype=torch.int64),
            "input_tokens": torch.zeros((1, 1), device="cuda", dtype=torch.int64),
            "output_tokens": torch.zeros((1, 1), device="cuda", dtype=torch.int64),
            "num_new_tokens": torch.ones((1,), device="cuda", dtype=torch.int32),
            "prompt_lengths": torch.full((1,), self.context, device="cuda", dtype=torch.int32),
            "qo_indptr_buffer": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            "paged_kv_indptr_buffer": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            "paged_kv_indices_buffer": torch.tensor([0], device="cuda", dtype=torch.int32),
            "paged_kv_last_page_len_buffer": torch.tensor([self.context + 1], device="cuda", dtype=torch.int32),
        }
        workers, schedulers = mirage.get_configurations_from_gpu(0)
        params = PersistentKernel.get_default_init_parameters()
        params.update(mode="online_notoken", test_mode=True,
                      num_workers=workers, num_local_schedulers=schedulers,
                      max_seq_length=self.context + 1,
                      max_num_batched_tokens=1, max_num_batched_requests=1,
                      max_num_pages=1, page_size=256, meta_tensors=self.meta)
        self.pk = PersistentKernel(**params)

        class ExportBuilder(Qwen3Builder):
            def new_intermediate_tensors(builder_self):
                super().new_intermediate_tensors()
                builder_self.argmax_in_tensor = torch.empty(
                    (1, builder_self.padded_vocab_size), device="cuda",
                    dtype=torch.bfloat16)
                builder_self.argmax_in = builder_self.mpk.attach_input(
                    builder_self.argmax_in_tensor, name="argmax_in_export")
                builder_self.argmax_part_value_tensor = torch.empty(
                    (1, builder_self.mpk.num_workers), device="cuda",
                    dtype=torch.bfloat16)
                builder_self.argmax_part_value = builder_self.mpk.attach_input(
                    builder_self.argmax_part_value_tensor, name="argmax_part_value_export")
                builder_self.argmax_part_index_tensor = torch.empty(
                    (1, builder_self.mpk.num_workers), device="cuda",
                    dtype=torch.int64)
                builder_self.argmax_part_index = builder_self.mpk.attach_input(
                    builder_self.argmax_part_index_tensor, name="argmax_part_index_export")

        self.builder = ExportBuilder(self.pk)
        self.builder.mpk_gate_up_groups = self.gate_up_groups
        cos, sin = _rope_tables(4096, p["head_dim"], "cuda")
        config = MirageModelConfig(
            hidden_size=p["hidden"], intermediate_size=p["intermediate"],
            vocab_size=p["vocab"], local_num_q_heads=p["q_heads"],
            local_num_kv_heads=p["kv_heads"], head_dim=p["head_dim"],
            num_layers=layers, k_cache=self.kcache, v_cache=self.vcache,
            position_embeddings=(cos, sin), state_dict=self.state,
            with_lm_head=True)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            self.builder.build_from_config(config)
        finally:
            torch.set_default_dtype(original_dtype)
        cache_root.mkdir(parents=True, exist_ok=True)
        self.compile_dir = Path(tempfile.mkdtemp(prefix="mpk_native_qwen3_", dir=cache_root))
        self.pk.compile(output_dir=str(self.compile_dir))

    def load(self, values: dict[str, torch.Tensor]) -> None:
        p = self.case.params
        self.meta["input_tokens"][0, 0] = values["token"]
        self.meta["tokens"][0, self.context] = values["token"]
        self.meta["step"].fill_(self.context)
        self.meta["output_tokens"].fill_(-1)
        self.meta["qo_indptr_buffer"].copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
        self.meta["paged_kv_indptr_buffer"].copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
        self.meta["paged_kv_indices_buffer"].fill_(0)
        self.meta["paged_kv_last_page_len_buffer"].fill_(self.context + 1)
        self.kcache[:, 0, :self.context].copy_(values["kcache"][:self.layers])
        self.vcache[:, 0, :self.context].copy_(values["vcache"][:self.layers])
        self.state["model.embed_tokens.weight"].copy_(values["embed"])
        self.state["lm_head.weight"].copy_(values["embed"])
        self.builder.lm_head_weight[:self.vocab].copy_(values["embed"])
        self.state["model.norm.weight"].copy_(values["fnorm"])
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            for target, source in (
                ("input_layernorm.weight", "ln1"),
                ("self_attn.q_norm.weight", "qn"),
                ("self_attn.k_norm.weight", "kn"),
                ("self_attn.o_proj.weight", "wo"),
                ("post_attention_layernorm.weight", "ln2"),
                ("mlp.down_proj.weight", "wd"),
            ):
                self.state[prefix + target].copy_(values[source][layer])
            self.state[prefix + "self_attn.qkv_proj.weight"].copy_(
                self.shuffle_tensors([values[name][layer] for name in ("wq", "wk", "wv")],
                                     p["kv_heads"], 0))
            self.state[prefix + "mlp.gate_up_proj.weight"].copy_(
                self.shuffle_tensors([values["wg"][layer], values["wu"][layer]],
                                     self.gate_up_groups, 0))

    def __call__(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.load(values)
        self.pk()
        # online_notoken does not join worker/scheduler streams back to the
        # calling stream before returning; read outputs only after completion.
        torch.cuda.synchronize()
        logits = self.builder.argmax_in_tensor[0, :self.vocab].float().clone()
        return {
            "logits": logits,
            "next_token": self.meta["output_tokens"][0, 0].clone(),
            "k_write": self.kcache[:, 0, self.context].clone(),
            "v_write": self.vcache[:, 0, self.context].clone(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--reps", type=int, default=0)
    args = parser.parse_args()
    case = select_cases("all", ["dense-step-qwen3-06b-b1-s128"])[0]
    result: dict = {"case": case.id, "layers": args.layers,
                    "gpu": torch.cuda.get_device_name(0),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "upstream native Qwen3 MPK graph, one-pass decode adapter"}
    try:
        cache = Path(os.environ.get("MPK_NATIVE_CACHE", tempfile.gettempdir()))
        initial = make_inputs(case, 12345, "cuda")
        start = time.perf_counter()
        candidate = NativeQwen3(case, initial, args.layers, cache)
        result["compile_and_build_ms"] = (time.perf_counter() - start) * 1000
        result["compile_dir"] = str(candidate.compile_dir)
        del initial
        result["trials"] = []
        for trial in range(args.trials):
            values = make_inputs(case, secrets.randbits(32), "cuda")
            expected = (reference(case, values) if args.layers == case.params["layers"]
                        else _truncated_reference(case, values, args.layers))
            with torch.inference_mode():
                actual = candidate(values)
            torch.cuda.synchronize()
            info = {"trial": trial,
                    "token": int(actual["next_token"].item()),
                    "logits_finite": bool(torch.isfinite(actual["logits"]).all().item())}
            if args.layers == 1:
                info["intermediates"] = _one_layer_diagnostics(case, values, candidate.builder)
            info["outputs"] = {}
            for name, wanted in expected.items():
                got = actual[name]
                delta = (got.float() - wanted.float()).abs() if wanted.is_floating_point() else None
                mismatch = (
                    ~torch.isclose(got.float(), wanted.float(),
                                   atol=case.atol,
                                   rtol=case.bf16_rtol if wanted.dtype == torch.bfloat16 else case.rtol)
                    if delta is not None else got != wanted)
                info["outputs"][name] = {
                    "mismatched_elements": int(mismatch.sum().item()),
                    "max_abs_error": float(delta.max().item()) if delta is not None else None,
                    "actual_max_abs": float(got.float().abs().max().item()),
                    "expected_max_abs": float(wanted.float().abs().max().item()),
                }
            result["trials"].append(info)
            del values, expected, actual
        if args.reps:
            perf_values = make_inputs(case, secrets.randbits(32), "cuda")
            result["adapter_timing"] = _measure(
                candidate, perf_values, "cuda", 1, args.reps)
            graph_host_ms = []
            graph_cuda_ms = []
            for repetition in range(args.reps + 1):
                candidate.load(perf_values)
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                stop_event = torch.cuda.Event(enable_timing=True)
                start_time = time.perf_counter()
                start_event.record()
                candidate.pk()
                torch.cuda.synchronize()
                stop_event.record()
                stop_event.synchronize()
                host_ms = (time.perf_counter() - start_time) * 1000
                if repetition > 0:
                    graph_host_ms.append(host_ms)
                    graph_cuda_ms.append(start_event.elapsed_time(stop_event))
            result["native_graph_timing"] = {
                "host_p50_ms": statistics.median(graph_host_ms),
                "cuda_event_p50_ms": statistics.median(graph_cuda_ms),
                "host_ms": graph_host_ms, "cuda_event_ms": graph_cuda_ms,
                "scope": "MPK graph launch and completion, input copies excluded",
            }
        if all(
            all(v["mismatched_elements"] == 0 for v in t["outputs"].values())
            for t in result["trials"]):
            result["status"] = "pass"
        else:
            result["status"] = "incorrect"
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("traceback",)}, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
