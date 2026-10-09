"""Compare the Waypoint-1.5 frame step with the pinned upstream model on CPU.

``--src`` is a local copy of ``Overworld/Waypoint-1.5-1B`` at the pinned
revision (``transformer/model.py``, ``modular_blocks.py``,
``transformer/config.json``). Those files are GPL-3 and are never vendored.
They are loaded from the path at verify time after their SHA-256 is checked.
Only the named math/cache classes are executed. Small shims replace the
diffusers config mixins and ``TensorDict``.

Eager ``flex_attention`` ignores block sparsity when ``mask_mod`` is None,
which the README's compiled serving path does not do. The probe therefore
re-expresses the upstream ``make_block_mask`` result as a ``mask_mod``,
using the same written-slot flags and torch's eager flex math.

Checks, for each development frame index:
1. Replay: upstream cache passes over the fixture's seeded history rebuild
   the input ``k_cache``/``v_cache`` (ring layout and written slots).
2. Step, BF16: upstream denoise loop plus cache pass, from the input caches,
   compared with ``waypoint.reference``.
3. Step, FP32: the same in FP32, compared with the FP32 oracle.
4. Config: case params and module constants against the pinned config.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.nn.attention.flex_attention as flex
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask

from .cases import select_cases
from .tasks import waypoint
from .tasks.oracle import fp32_reference
from .tests.test_waypoint import development

MODEL = "Overworld/Waypoint-1.5-1B"
CASE_ID = "world-frame-step-waypoint15-1b-f128"
MODEL_NAMES = ["NoCastModule", "rms_norm", "MLP", "AdaLN", "ada_rmsnorm", "ada_gate",
               "NoiseConditioner", "OrthoRoPEAngles", "OrthoRoPE", "Attn",
               "ControllerInputEmbedding", "MLPFusion", "CFG", "CondHead",
               "WorldDiTBlock", "WorldDiT", "WorldModel"]
BLOCK_NAMES = ["make_block_mask", "LayerKVCache", "StaticKVCache", "WorldEngineDenoiseLoop"]


def pins() -> dict:
    tasks = Path(__file__).parent / "tasks"
    return json.loads((tasks / "model_specs.json").read_text())[MODEL]


def checked(src: Path, name: str, spec: dict) -> bytes:
    data = (src / name).read_bytes()
    want = (spec["source_files"].get(name) or {}).get("sha256")
    if name == "transformer/config.json":
        want = spec["config_snapshot_sha256"]
    if hashlib.sha256(data).hexdigest() != want:
        raise ValueError(f"primary source hash differs: {src / name}")
    return data


def load_nodes(source: bytes, path: str, names: list[str], namespace: dict) -> None:
    selected = [node for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"missing primary definitions in {path}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)


class TensorDict(dict):
    def __init__(self, values, batch_size=None):
        super().__init__(values)


def register_to_config(init):
    signature = inspect.signature(init)

    def wrapper(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        nn.Module.__init__(self)
        self.config = SimpleNamespace(**values)
        init(self, *args, **kwargs)
    return wrapper


def upstream(src: Path, spec: dict) -> dict:
    namespace = {"torch": torch, "nn": nn, "F": F, "Tensor": Tensor, "eo": einops,
                 "warnings": __import__("warnings"), "TensorDict": TensorDict,
                 "ModelMixin": nn.Module, "ConfigMixin": object,
                 "register_to_config": register_to_config, "HAS_FBGEMM": False,
                 "BlockMask": BlockMask, "_DEFAULT_SPARSE_BLOCK_SIZE": _DEFAULT_SPARSE_BLOCK_SIZE,
                 "ModularPipelineBlocks": object, "InputParam": None, "OutputParam": None,
                 "ComponentSpec": None, "ModularPipeline": object, "PipelineState": object}
    load_nodes(checked(src, "transformer/model.py", spec), "transformer/model.py",
               MODEL_NAMES, namespace)
    load_nodes(checked(src, "modular_blocks.py", spec), "modular_blocks.py",
               BLOCK_NAMES, namespace)
    return namespace


_EAGER_FLEX = flex.flex_attention


def block_sparse_flex(query, key, value, block_mask=None, **kwargs):
    """Eager flex math restricted to the block mask's full blocks."""
    if block_mask is not None:
        if int(block_mask.kv_num_blocks.sum()):
            raise ValueError("upstream masks have only full blocks")
        dense = block_mask.to_dense()[0, 0].bool()
        if not bool((dense == dense[:1]).all()):
            raise ValueError("upstream masks are identical across query blocks")
        keep = dense[0].repeat_interleave(_DEFAULT_SPARSE_BLOCK_SIZE)
        block_mask = flex.create_block_mask(lambda b, h, q, kv: keep[kv], None, None,
                                            query.shape[-2], key.shape[-2], device="cpu")
    return _EAGER_FLEX(query, key, value, block_mask=block_mask, **kwargs)


def config_kwargs(case, cfg: dict) -> dict:
    p = case.params
    geo = waypoint.geometry(case)
    kwargs = {k: cfg[k] for k in (
        "global_attn_offset", "value_residual", "gated_attn", "ctrl_conditioning",
        "ctrl_cond_dropout", "prompt_conditioning", "noise_conditioning", "scheduler_sigmas",
        "base_fps", "causal", "rope_impl", "moe", "temporal_compression", "inference_fps",
        "taehv_ae", "rope_nyquist_frac", "rope_theta", "n_frames")}
    return kwargs | {
        "d_model": p["hidden"], "n_heads": p["q_heads"], "n_kv_heads": p["kv_heads"],
        "n_layers": p["layers"], "mlp_ratio": p["mlp_ratio"], "channels": p["latent_channels"],
        "height": geo["grid_h"], "width": geo["grid_w"], "patch": [p["patch"]] * 2,
        "tokens_per_frame": p["tokens_per_frame"], "local_window": p["local_window"],
        "global_window": p["global_window"], "global_attn_period": p["global_period"],
        "global_pinned_dilation": p["global_dilation"], "n_buttons": p["buttons"],
        "ctrl_conditioning_period": p["control_period"]}


def state_dict(case, values: dict, dtype) -> dict:
    p = case.params
    s = {"patchify.weight": values["patch_w"], "unpatchify.weight": values["unpatch_w"],
         "unpatchify.bias": values["unpatch_b"], "out_norm.fc.weight": values["out_norm_w"],
         "denoise_step_emb.mlp.fc1.weight": values["noise_fc1"],
         "denoise_step_emb.mlp.fc2.weight": values["noise_fc2"],
         "ctrl_emb.mlp.fc1.weight": values["ctrl_fc1"], "ctrl_emb.mlp.fc2.weight": values["ctrl_fc2"],
         "ctrl_cfg.null_emb": torch.zeros(1, 1, p["hidden"])}
    fusion = {layer: i for i, layer in enumerate(waypoint.control_layers(case))}
    for l in range(p["layers"]):
        b = f"transformer.blocks.{l}."
        s |= {b + "attn.q_proj.weight": values["wq"][l], b + "attn.k_proj.weight": values["wk"][l],
              b + "attn.v_proj.weight": values["wv"][l], b + "attn.out_proj.weight": values["wo"][l],
              b + "attn.v_lamb": values["v_lamb"][l],
              b + "dit_mlp.fc1.weight": values["fc1"][l], b + "dit_mlp.fc2.weight": values["fc2"][l],
              b + "attn_cond_head.bias_in": values["attn_cond_bias"][l],
              b + "mlp_cond_head.bias_in": values["mlp_cond_bias"][l]}
        for i in range(3):
            s[b + f"attn_cond_head.cond_proj.{i}.weight"] = values["attn_cond_w"][l, i]
            s[b + f"mlp_cond_head.cond_proj.{i}.weight"] = values["mlp_cond_w"][l, i]
        if l in fusion:
            for ours, theirs in (("fuse_x", "fc1_x"), ("fuse_c", "fc1_c"), ("fuse_out", "fc2")):
                s[b + f"ctrl_mlpfusion.{theirs}.weight"] = values[ours][fusion[l]]
    # Only the noise conditioner is FP32 in BF16 serving.
    return {k: v.float() if k.startswith("denoise_step_emb") else v.to(dtype) for k, v in s.items()}


def build_model(ns, case, cfg, values, dtype):
    """Mirror diffusers ``from_pretrained(torch_dtype=...)`` with FP32 modules.

    ``_keep_in_fp32_modules`` makes diffusers skip ``model.to(dtype)``.
    ``NoCastModule`` would otherwise round FP32 weights and buffers through
    the target dtype. The FP32 modules are therefore rebuilt after the cast.
    """
    model = ns["WorldModel"](**config_kwargs(case, cfg)).eval()
    model.load_state_dict(state_dict(case, values, dtype), strict=True)
    model.to(dtype)
    noise = ns["NoiseConditioner"](case.params["hidden"])
    noise.mlp.fc1.weight.data.copy_(values["noise_fc1"])
    noise.mlp.fc2.weight.data.copy_(values["noise_fc2"])
    model.denoise_step_emb = noise.eval()
    model.transformer.rope_angles = ns["OrthoRoPEAngles"](model.config)
    return model


def load_cache(ns, model, case, k_cache, v_cache, dtype):
    """Place MegaBench ring slots into upstream LayerKVCache buffers."""
    cache = ns["StaticKVCache"](model.config, 1, dtype)
    T = case.params["tokens_per_frame"]
    for l, layer in enumerate(cache.layers):
        for slot, g in enumerate(waypoint.slot_frames(case, l, case.params["frame_index"])):
            if g is None:
                continue
            span = slice(slot * T, (slot + 1) * T)
            layer.kv[0, 0, :, span] = k_cache[l, slot].permute(1, 0, 2).to(dtype)
            layer.kv[1, 0, :, span] = v_cache[l, slot].permute(1, 0, 2).to(dtype)
            layer.written[span] = True
    return cache


def controller(values, dtype):
    return {name: values[name].to(dtype).view(1, 1, -1) for name in ("mouse", "button", "scroll")}


def ts_mult(cfg) -> int:
    return int(cfg["base_fps"]) // int(cfg["inference_fps"] / cfg["temporal_compression"])


def step(ns, model, cfg, case, values, cache, dtype):
    loop = ns["WorldEngineDenoiseLoop"]
    f = case.params["frame_index"]
    stamp = torch.tensor([[f]], dtype=torch.long)
    sigmas = torch.tensor(cfg["scheduler_sigmas"], dtype=torch.bfloat16).to(dtype)
    ctrl = controller(values, dtype)
    x = values["noise"].to(dtype)[None, None]
    x = loop._denoise_pass(model, x, sigmas, stamp * ts_mult(cfg), stamp, None, None,
                           ctrl["mouse"], ctrl["button"], ctrl["scroll"], cache).clone()
    loop._cache_pass(model, x, stamp * ts_mult(cfg), stamp, None, None,
                     ctrl["mouse"], ctrl["button"], ctrl["scroll"], cache)
    k = torch.stack([layer.kv[0, 0, :, layer.L:].permute(1, 0, 2) for layer in cache.layers])
    v = torch.stack([layer.kv[1, 0, :, layer.L:].permute(1, 0, 2) for layer in cache.layers])
    return {"latent": x[0, 0], "k_write": k, "v_write": v}, cache


def replay(ns, model, cfg, case, seed, dtype):
    """Upstream cache passes over the seeded history drawn by make_inputs."""
    p = case.params
    gen = torch.Generator().manual_seed(seed)
    waypoint.make_weights(case, gen, "cpu")
    cache = ns["StaticKVCache"](model.config, 1, dtype)
    loop = ns["WorldEngineDenoiseLoop"]
    for g in range(p["frame_index"]):
        latent = torch.randn((p["latent_channels"], p["latent_height"], p["latent_width"]),
                             generator=gen).to(torch.bfloat16)
        ctrl = controller(waypoint._random_controller(gen, case, "cpu"), dtype)
        stamp = torch.tensor([[g]], dtype=torch.long)
        loop._cache_pass(model, latent.to(dtype)[None, None], stamp * ts_mult(cfg), stamp,
                         None, None, ctrl["mouse"], ctrl["button"], ctrl["scroll"], cache)
    return cache


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def check_config(cfg, problems: list[str]) -> None:
    p = select_cases("all", [CASE_ID])[0].params
    want = {"layers": cfg["n_layers"], "hidden": cfg["d_model"], "q_heads": cfg["n_heads"],
            "kv_heads": cfg["n_kv_heads"], "head_dim": cfg["d_model"] // cfg["n_heads"],
            "mlp_ratio": cfg["mlp_ratio"], "latent_channels": cfg["channels"],
            "latent_height": cfg["height"] * cfg["patch"][0],
            "latent_width": cfg["width"] * cfg["patch"][1], "patch": cfg["patch"][0],
            "tokens_per_frame": cfg["tokens_per_frame"], "local_window": cfg["local_window"],
            "global_window": cfg["global_window"], "global_period": cfg["global_attn_period"],
            "global_dilation": cfg["global_pinned_dilation"], "buttons": cfg["n_buttons"],
            "control_period": cfg["ctrl_conditioning_period"],
            "denoise_steps": len(cfg["scheduler_sigmas"]) - 1, "batch": 1}
    for key, value in want.items():
        if p.get(key) != value:
            problems.append(f"config: case param {key}={p.get(key)}, upstream {value}")
    sigmas = torch.tensor(cfg["scheduler_sigmas"], dtype=torch.bfloat16).tolist()
    constants = {"SIGMAS": (tuple(sigmas), waypoint.SIGMAS),
                 "GLOBAL_ATTN_OFFSET": (cfg["global_attn_offset"], waypoint.GLOBAL_ATTN_OFFSET),
                 "TS_MULT": (ts_mult(cfg), waypoint.TS_MULT),
                 "ROPE_THETA": (cfg["rope_theta"], waypoint.ROPE_THETA),
                 "ROPE_NYQUIST_FRAC": (cfg["rope_nyquist_frac"], waypoint.ROPE_NYQUIST_FRAC)}
    for name, (theirs, ours) in constants.items():
        if theirs != ours:
            problems.append(f"config: {name}={ours}, upstream {theirs}")
    if not (cfg["value_residual"] and not cfg["gated_attn"] and cfg["prompt_conditioning"] is None
            and cfg["noise_conditioning"] == "wan" and not cfg["moe"] and cfg["rope_impl"] == "ortho"):
        problems.append("config: architecture switches differ from the reference")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--frames", type=int, nargs="+", default=[2, 8, 9])
    parser.add_argument("--bf16-tol", type=float, default=0.02)
    parser.add_argument("--fp32-tol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    spec = pins()
    cfg = json.loads(checked(args.src, "transformer/config.json", spec))
    ns = upstream(args.src, spec)
    flex.flex_attention = block_sparse_flex
    problems, report = [], {}
    try:
        with torch.inference_mode():
            for f in args.frames:
                case = development(f)
                values = waypoint.make_inputs(case, args.seed, "cpu")
                ours = waypoint.reference(case, values)
                exact = fp32_reference(waypoint, case, values)
                row = {}
                model = build_model(ns, case, cfg, values, torch.bfloat16)
                rebuilt = replay(ns, model, cfg, case, args.seed, torch.bfloat16)
                staged = load_cache(ns, model, case, values["k_cache"], values["v_cache"], torch.bfloat16)
                for l, (a, b) in enumerate(zip(rebuilt.layers, staged.layers)):
                    if not torch.equal(a.written, b.written):
                        problems.append(f"f={f} layer {l}: written slots differ")
                    span = slice(0, a.L)
                    row[f"replay_kv_rel_l{l}"] = rel(a.kv[:, :, :, span], b.kv[:, :, :, span]) \
                        if bool(b.written[:a.L].any()) else 0.0
                theirs, _ = step(ns, model, cfg, case, values, staged, torch.bfloat16)
                for name in ours:
                    row[f"bf16_{name}"] = rel(theirs[name], ours[name])
                    row[f"bf16_ref_vs_oracle_{name}"] = rel(ours[name], exact[name])
                model32 = build_model(ns, case, cfg, values, torch.float32)
                staged32 = load_cache(ns, model32, case, values["k_cache"], values["v_cache"], torch.float32)
                theirs32, _ = step(ns, model32, cfg, case, values, staged32, torch.float32)
                for name in ours:
                    row[f"fp32_{name}"] = rel(theirs32[name], exact[name])
                for key, value in row.items():
                    tol = args.fp32_tol if key.startswith("fp32") else args.bf16_tol
                    if "vs_oracle" not in key and not value <= tol:
                        problems.append(f"f={f} {key}={value:.3g} > {tol}")
                report[f"frame_{f}"] = row
    finally:
        flex.flex_attention = _EAGER_FLEX
    check_config(cfg, problems)
    print(json.dumps({"report": report, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
