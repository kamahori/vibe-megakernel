"""Compare the pi0.5 task with pinned LeRobot ``PI05Pytorch`` code on CPU.

``--src`` is a LeRobot checkout (or a directory with the same relative
layout) at the revision pinned in ``tasks/model_specs.json``. Every
file is sha256-checked and only the named classes, functions and constants
are AST-loaded; the LeRobot package itself is never imported. The model is
built with ``dtype=bfloat16`` at a development geometry and receives the
task's synthetic weights. Three checks:

1. timed step: LeRobot ``euler_integrate`` + ``denoise_step`` over the task's
   prefix K/V cache must reproduce ``pi05.reference`` (same BF16 schedule);
2. fixture: LeRobot ``embed_prefix`` + PaliGemma prefill (its SigLIP runs in
   FP32 on CPU) must reproduce the task's BF16 prefix K/V within BF16 noise;
3. end to end: LeRobot ``sample_actions`` from pixels and tokens.

The adapters replace only ``get_gemma_config`` (development widths), the
SigLIP/projector geometry (PaliGemma's So400m defaults are hardcoded), and
drop ``cache_position`` from ``create_causal_mask`` (LeRobot pins
transformers 5.4-5.5; the venv has 5.17, where a 4D mask passes unchanged).
"""

from __future__ import annotations

import argparse
import ast
import enum
import hashlib
import json
import logging
import math
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .cases import CASES, Case
from .tasks import pi05

MODEL = "lerobot/pi05_base"
CASE_ID = "vla-action-step-pi05-b1"
DEV_PARAMS = {"batch": 1, "images": 3, "image_tokens": 16, "image_size": 56, "patch_size": 14,
              "text_tokens": 160, "vision_hidden": 64, "vision_layers": 2,
              "vision_intermediate": 96, "vision_heads": 4, "layers": 3, "hidden": 128,
              "intermediate": 256, "q_heads": 8, "kv_heads": 1, "head_dim": 32,
              "expert_hidden": 64, "expert_intermediate": 128, "horizon": 50,
              "action_dim": 32, "denoise_steps": 10}


def development() -> Case:
    case = next(case for case in CASES if case.id == CASE_ID)
    return replace(case, ready=True, params=dict(DEV_PARAMS))


def pinned_spec() -> dict:
    tasks = Path(__file__).parent / "tasks"
    return json.loads((tasks / "model_specs.json").read_text())[MODEL]


def load_nodes(path: Path, names: list[str], namespace: dict, pin: dict) -> None:
    """Execute only the named top-level defs/classes/assignments of a pinned file."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != pin["sha256"]:
        raise ValueError(f"primary source hash differs: {path}")

    def name(node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return None

    selected = [node for node in ast.parse(source).body if name(node) in names]
    if {name(node) for node in selected} != set(names):
        raise ValueError(f"missing primary definitions in {path}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def load_primary(src: Path) -> dict:
    from transformers import DynamicCache, PaliGemmaConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration, PaliGemmaModel)

    pins = pinned_spec()["source_files"]

    def load(rel: str, names: list[str], namespace: dict) -> None:
        load_nodes(src / "src/lerobot" / rel, names, namespace, pins["lerobot/src/lerobot/" + rel])

    def require_package(*args, **kwargs):
        return None

    base = {"torch": torch, "nn": nn, "F": F, "Tensor": Tensor, "math": math, "logging": logging,
            "enum": enum, "Callable": Callable, "Literal": Literal, "Any": Any, "cast": cast,
            "nullcontext": nullcontext, "SDPBackend": SDPBackend, "sdpa_kernel": sdpa_kernel,
            "DynamicCache": DynamicCache, "require_package": require_package}
    utils = dict(base)
    load("utils/constants.py", ["OPENPI_ATTENTION_MASK_VALUE"], utils)
    load("utils/device_utils.py", ["get_safe_dtype"], utils)
    load("policies/common/vla_utils.py", ["create_sinusoidal_pos_embedding", "make_att_2d_masks",
                                          "prepare_attention_masks_4d", "clone_past_key_values"], utils)
    load("policies/common/flow_matching.py", ["FlowConvention", "sample_noise", "euler_integrate"], utils)
    def causal_mask(*, cache_position=None, **kwargs):
        # LeRobot pins transformers 5.4-5.5; 5.17 dropped ``cache_position``.
        # Both return the prepared 4D masks used here unchanged.
        return create_causal_mask(**kwargs)

    gemma = dict(base, create_causal_mask=causal_mask,
                 GradientCheckpointingLayer=GradientCheckpointingLayer,
                 BaseModelOutputWithPast=BaseModelOutputWithPast,
                 GemmaAttention=modeling_gemma.GemmaAttention, GemmaMLP=modeling_gemma.GemmaMLP,
                 GemmaModel=modeling_gemma.GemmaModel, GemmaForCausalLM=modeling_gemma.GemmaForCausalLM,
                 PaliGemmaModel=PaliGemmaModel,
                 PaliGemmaForConditionalGeneration=PaliGemmaForConditionalGeneration)
    load("policies/pi_gemma.py", ["_gated_residual", "layernorm_forward", "PiGemmaRMSNorm",
                                  "_get_pi_gemma_decoder_layer_base", "PiGemmaModel",
                                  "PiGemmaForCausalLM", "PaliGemmaModelWithPiGemma",
                                  "PaliGemmaForConditionalGenerationWithPiGemma"], gemma)
    model = dict(base, **{name: utils[name] for name in (
        "OPENPI_ATTENTION_MASK_VALUE", "create_sinusoidal_pos_embedding", "make_att_2d_masks",
        "prepare_attention_masks_4d", "clone_past_key_values", "euler_integrate", "sample_noise")})
    model.update({name: gemma[name] for name in (
        "PaliGemmaForConditionalGenerationWithPiGemma", "PiGemmaForCausalLM", "_gated_residual",
        "layernorm_forward")}, modeling_gemma=modeling_gemma, DEFAULT_IMAGE_SIZE=224)
    load("policies/pi05/modeling_pi05.py", ["_no_grad_if", "GemmaConfig", "PaliGemmaWithExpertModel",
                                            "PI05Pytorch"], model)
    load("policies/pi05/modeling_pi05.py", ["get_gemma_config"], model)
    model["primary_get_gemma_config"] = model["get_gemma_config"]
    model["PaliGemmaConfig"] = PaliGemmaConfig
    model["primary_CONFIG_MAPPING"] = CONFIG_MAPPING
    return model


def build_model(case: Case, ns: dict):
    """Instantiate the primary PI05Pytorch at development geometry, BF16 precision."""
    from transformers import PaliGemmaConfig, SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    p = case.params
    full = {variant: ns["primary_get_gemma_config"](variant) for variant in ("gemma_2b", "gemma_300m")}
    if (full["gemma_2b"].width, full["gemma_300m"].width, full["gemma_2b"].head_dim) != (2048, 1024, 256):
        raise AssertionError("pinned Gemma variants changed")
    dev = {"gemma_2b": ns["GemmaConfig"](p["hidden"], p["layers"], p["intermediate"], p["q_heads"],
                                         p["kv_heads"], p["head_dim"]),
           "gemma_300m": ns["GemmaConfig"](p["expert_hidden"], p["layers"], p["expert_intermediate"],
                                           p["q_heads"], p["kv_heads"], p["head_dim"])}
    vision = dict(hidden_size=p["vision_hidden"], num_hidden_layers=p["vision_layers"],
                  intermediate_size=p["vision_intermediate"], num_attention_heads=p["vision_heads"],
                  patch_size=p["patch_size"], image_size=p["image_size"], vision_use_head=False)
    real = ns["primary_CONFIG_MAPPING"]
    ns["get_gemma_config"] = dev.__getitem__
    # Avoid building a full So400m tower; its geometry is restored below.
    ns["CONFIG_MAPPING"] = {"paligemma": lambda: PaliGemmaConfig(vision_config=vision), "gemma": real["gemma"]}
    config = SimpleNamespace(
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", dtype=torch.bfloat16,
        image_resolution=(p["image_size"], p["image_size"]), freeze_vision_encoder=False,
        train_expert_only=False, max_action_dim=p["action_dim"], max_state_dim=p["action_dim"],
        use_proprioceptive_memory=False, compile_model=False, chunk_size=p["horizon"],
        num_inference_steps=p["denoise_steps"], min_period=4e-3, max_period=4.0, rtc_config=None)
    model = ns["PI05Pytorch"](config).eval()
    paligemma = model.paligemma_with_expert.paligemma.model
    # PaliGemmaWithExpertModel hardcodes So400m's 4304 MLP and 2048 projection.
    paligemma.vision_tower = SiglipVisionModel(SiglipVisionConfig(**vision)).float().eval()
    paligemma.multi_modal_projector.linear = nn.Linear(p["vision_hidden"], p["hidden"]).float()
    set_rope_frequencies(model, torch.float32)
    return model


def set_rope_frequencies(model, dtype: torch.dtype) -> None:
    """Set the RoPE inverse-frequency buffers' precision.

    ``PaliGemmaWithExpertModel.to_bfloat16_for_selected_params`` calls
    ``self.to(bfloat16)``, which also rounds the rotary ``inv_freq`` buffers to
    BF16 (phase errors grow with position). openpi's JAX model, which trained
    the checkpoints, keeps FP32 frequencies, and so does MegaBench's serving
    convention (tests/test_non_p0.serving_model). The checks restore FP32;
    the report also measures the BF16-buffer variant.
    """
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "compute_default_rope_parameters"):
            inv, _ = module.compute_default_rope_parameters(module.config)
            module.inv_freq = inv.to(dtype)
            module.original_inv_freq = inv.to(dtype)


def load_weights(case: Case, model, vlm: dict, values: dict) -> None:
    p = case.params
    pg = model.paligemma_with_expert.paligemma.model
    tower = getattr(pg.vision_tower, "vision_model", pg.vision_tower)
    expert = model.paligemma_with_expert.gemma_expert.model

    def put(parameter, value):
        if parameter.shape != value.shape:
            raise AssertionError(f"shape {tuple(parameter.shape)} != {tuple(value.shape)}")
        parameter.copy_(value)

    put(tower.embeddings.patch_embedding.weight, vlm["patch_weight"])
    put(tower.embeddings.patch_embedding.bias, vlm["patch_bias"])
    put(tower.embeddings.position_embedding.weight, vlm["position"])
    for i, layer in enumerate(tower.encoder.layers):
        for module, key in ((layer.layer_norm1, "norm1"), (layer.layer_norm2, "norm2")):
            put(module.weight, vlm[key][i])
            put(module.bias, vlm[key + "_bias"][i])
        for module, key in ((layer.self_attn.q_proj, "q"), (layer.self_attn.k_proj, "k"),
                            (layer.self_attn.v_proj, "v"), (layer.self_attn.out_proj, "o"),
                            (layer.mlp.fc1, "up"), (layer.mlp.fc2, "down")):
            put(module.weight, vlm[key][i])
            put(module.bias, vlm[key + "_bias"][i])
    put(tower.post_layernorm.weight, vlm["final_norm"])
    put(tower.post_layernorm.bias, vlm["final_bias"])
    put(pg.multi_modal_projector.linear.weight, vlm["projector"].T)
    put(pg.multi_modal_projector.linear.bias, vlm["projector_bias"])
    lm = pg.language_model
    put(lm.embed_tokens.weight, vlm["embed"])
    for i, layer in enumerate(lm.layers):
        put(layer.input_layernorm.weight, vlm["ln1"][i])
        put(layer.post_attention_layernorm.weight, vlm["ln2"][i])
        for module, key in ((layer.self_attn.q_proj, "lq"), (layer.self_attn.k_proj, "lk"),
                            (layer.self_attn.v_proj, "lv"), (layer.self_attn.o_proj, "lo"),
                            (layer.mlp.gate_proj, "lg"), (layer.mlp.up_proj, "lu"),
                            (layer.mlp.down_proj, "ld")):
            put(module.weight, vlm[key][i])
    for i, layer in enumerate(expert.layers):
        for module, key in ((layer.input_layernorm.dense, "attn_norm"),
                            (layer.post_attention_layernorm.dense, "ffn_norm")):
            put(module.weight, values[key + "_weight"][i])
            put(module.bias, values[key + "_bias"][i])
        for module, key in ((layer.self_attn.q_proj, "wq"), (layer.self_attn.k_proj, "wk"),
                            (layer.self_attn.v_proj, "wv"), (layer.self_attn.o_proj, "wo"),
                            (layer.mlp.gate_proj, "wg"), (layer.mlp.up_proj, "wu"),
                            (layer.mlp.down_proj, "wd")):
            put(module.weight, values[key][i])
    put(expert.norm.dense.weight, values["final_norm_weight"])
    put(expert.norm.dense.bias, values["final_norm_bias"])
    for module, key in ((model.action_in_proj, "action_in"), (model.action_out_proj, "action_out"),
                        (model.time_mlp_in, "time_in"), (model.time_mlp_out, "time_out")):
        put(module.weight, values[key + "_weight"])
        put(module.bias, values[key + "_bias"])
    dtypes = {name: param.dtype for name, param in model.named_parameters()}
    if dtypes["paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"] != torch.bfloat16 or \
            dtypes["paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight"] != torch.float32 or \
            dtypes["action_in_proj.weight"] != torch.float32:
        raise AssertionError("unexpected primary precision layout")


def relative(actual: Tensor, expected: Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


def check(case: Case, ns: dict, seed: int) -> dict:
    from transformers import DynamicCache

    p = case.params
    values = pi05.make_inputs(case, seed, "cpu")
    vlm = pi05.vlm_inputs(case, seed, "cpu")
    tokens, text_mask = pi05.prompt(case, seed, "cpu")
    model = build_model(case, ns)
    load_weights(case, model, vlm, values)
    expected = pi05.reference(case, values)["actions"]
    noise = values["noise"][None]
    mask = values["prefix_mask"][None]

    # 1. Timed step from the task's own prefix cache.
    cache = DynamicCache()
    for layer in range(p["layers"]):
        cache.update(values["prefix_k"][layer].permute(1, 0, 2)[None].clone(),
                     values["prefix_v"][layer].permute(1, 0, 2)[None].clone(), layer)
    def timed() -> Tensor:
        return ns["euler_integrate"](
            lambda x, t: model.denoise_step(prefix_pad_masks=mask, past_key_values=cache, x_t=x, timestep=t),
            noise.clone(), p["denoise_steps"])[0]

    step = timed()
    set_rope_frequencies(model, torch.bfloat16)
    bf16_frequencies = timed()
    set_rope_frequencies(model, torch.float32)

    # 2. Fixture prefill. Upstream SigLIP runs in FP32 on CPU, the task's in
    # BF16, so image features differ by BF16 noise. Given the same prefix
    # embeddings, the task's PaliGemma prefill must reproduce the upstream
    # cache exactly, padded positions included.
    images = [vlm["pixels"][i:i + 1].float() for i in range(p["images"])]
    image_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    valid = values["prefix_mask"]
    embeddings, pad, att = model.embed_prefix(images, image_masks, tokens[None], text_mask[None])
    if not torch.equal(pad[0], valid) or bool(att.any()):
        raise AssertionError("prefix pad/attention layout differs")
    images_end = p["images"] * p["image_tokens"]
    scale = torch.tensor(math.sqrt(p["hidden"]), dtype=torch.bfloat16)
    text_exact = torch.equal(embeddings[0, images_end:], (vlm["embed"][tokens] * scale).float())
    image_error = relative(pi05.siglip(case, vlm), embeddings[0, :images_end])
    att4 = ns["prepare_attention_masks_4d"](ns["make_att_2d_masks"](pad, att))
    # As in PI05Pytorch.sample_actions.
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, primary_cache = model.paligemma_with_expert.forward(
        attention_mask=att4, position_ids=torch.cumsum(pad, 1) - 1, past_key_values=None,
        inputs_embeds=[embeddings, None], use_cache=True)
    task_k, task_v = pi05.prefill(case, vlm, embeddings[0], valid)
    prefill_exact = all(torch.equal(task_k[layer], primary_cache.layers[layer].keys[0].transpose(0, 1)) and
                        torch.equal(task_v[layer], primary_cache.layers[layer].values[0].transpose(0, 1))
                        for layer in range(p["layers"]))
    fixture_k = max(relative(values["prefix_k"][layer][valid],
                             primary_cache.layers[layer].keys[0].transpose(0, 1)[valid])
                    for layer in range(p["layers"]))

    # 3. End to end.
    end = model.sample_actions(images, image_masks, tokens[None], text_mask[None],
                               noise=noise.clone(), num_steps=p["denoise_steps"])[0]
    record = {"seed": seed, "valid_prefix": int(valid.sum()), "prefix_length": int(valid.numel()),
              "timed_step_max_abs": float((step - expected).abs().max()),
              "timed_step_rel_l2": relative(expected, step),
              "text_embeddings_exact": text_exact, "prefill_exact": prefill_exact,
              "siglip_bf16_vs_fp32_rel_l2": image_error, "fixture_prefix_k_rel_l2_max": fixture_k,
              "end_to_end_rel_l2": relative(expected, end),
              "upstream_bf16_inv_freq_timed_rel_l2": relative(expected, bf16_frequencies),
              "noise_rel_change": relative(expected, values["noise"]) }
    # The timed step and the prefill run the same BF16 schedule as upstream
    # and must match exactly; the SigLIP and end-to-end bands are BF16 noise.
    record["pass"] = (record["timed_step_max_abs"] == 0.0 and text_exact and prefill_exact
                      and image_error < 2e-2 and fixture_k < 3e-2
                      and record["end_to_end_rel_l2"] < 1e-2)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="LeRobot checkout root")
    parser.add_argument("--output", type=Path, help="new JSON report path (exclusive create)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    args = parser.parse_args()
    torch.manual_seed(0)
    namespace = load_primary(args.src.resolve())
    case = development()
    with torch.no_grad():
        trials = [check(case, namespace, seed) for seed in args.seeds]
    report = {"case": case.to_dict(), "tier": "development_geometry",
              "primary_sources": {name: pin for name, pin in pinned_spec()["source_files"].items()
                                  if name.startswith("lerobot/")},
              "trials": trials, "status": "pass" if all(t["pass"] for t in trials) else "fail"}
    if args.output:
        with args.output.open("x") as file:
            json.dump(report, file, indent=2)
    print(json.dumps(trials, indent=1))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
