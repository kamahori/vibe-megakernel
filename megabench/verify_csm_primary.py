"""Check the CSM-1B frame oracle against the installed transformers CSM model.

The installed ``modeling_csm.py``/``generation_csm.py`` must match the pinned
sha256 values. The script copies ``csm.make_inputs`` weights into a BF16
``CsmForConditionalGeneration``, runs the backbone forward on the previous
frame over the given KV cache, then calls the upstream depth-decoder
``generate`` (greedy, as ``CsmGenerationMixin._sample`` does) and a
teacher-forced depth-decoder forward. Codes, every codebook's logits and the
backbone K/V writes are compared with ``csm.reference``.

    # development geometry on CPU (default)
    .venv/bin/python -m megabench.verify_csm_primary
    # full geometry, inside a GPU allocation
    .venv/bin/python -m megabench.verify_csm_primary --full --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from .cases import CASES, Case
from .tasks import csm

CASE_ID = "tts-frame-step-csm-1b-b1-s128"
MODEL = "sesame/csm-1b"
DEV_PARAMS = {"batch": 1, "context": 6, "layers": 2, "hidden": 64, "q_heads": 4,
              "kv_heads": 2, "head_dim": 16, "intermediate": 96, "text_vocab": 32,
              "codebooks": 32, "audio_vocab": 67, "depth_layers": 2, "depth_hidden": 48,
              "depth_q_heads": 4, "depth_kv_heads": 2, "depth_head_dim": 16,
              "depth_intermediate": 80}


def full_case() -> Case:
    return replace(next(case for case in CASES if case.id == CASE_ID), ready=True)


def development() -> Case:
    return replace(full_case(), params=dict(DEV_PARAMS))


def pinned_spec() -> dict:
    tasks = Path(__file__).parent / "tasks"
    return json.loads((tasks / "model_specs.json").read_text())[MODEL]


def check_sources() -> dict:
    """Fail unless the installed transformers CSM files are the pinned ones."""
    import transformers

    root = Path(transformers.__file__).parent
    pins = pinned_spec()["source_files"]
    for name, pin in pins.items():
        path = root / Path(name).relative_to("transformers")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pin["sha256"]:
            raise AssertionError(f"{path}: sha256 {digest} != pinned {pin['sha256']}")
    return pins


def hf_config(case: Case):
    """The pinned config, with geometry fields replaced by the case params."""
    from transformers.models.csm.configuration_csm import CsmConfig

    p = case.params
    config = json.loads(json.dumps(pinned_spec()["config"]))
    config.update(num_hidden_layers=p["layers"], hidden_size=p["hidden"],
                  num_attention_heads=p["q_heads"], num_key_value_heads=p["kv_heads"],
                  head_dim=p["head_dim"], intermediate_size=p["intermediate"],
                  text_vocab_size=p["text_vocab"], num_codebooks=p["codebooks"],
                  vocab_size=p["audio_vocab"])
    config["depth_decoder_config"].update(
        num_hidden_layers=p["depth_layers"], hidden_size=p["depth_hidden"],
        num_attention_heads=p["depth_q_heads"], num_key_value_heads=p["depth_kv_heads"],
        head_dim=p["depth_head_dim"], intermediate_size=p["depth_intermediate"],
        num_codebooks=p["codebooks"], vocab_size=p["audio_vocab"],
        backbone_hidden_size=p["hidden"], max_position_embeddings=p["codebooks"] + 1)
    config = CsmConfig(**config)
    for sub in (config, config.depth_decoder_config):
        sub._attn_implementation = "eager"
    return config


def serving_model(model):
    """BF16 parameters/activations, keeping FP32 rotary frequency buffers."""
    frequencies = {name: value.clone() for name, value in model.named_buffers()
                   if "inv_freq" in name}
    model = model.to(torch.bfloat16).eval()
    for name, value in frequencies.items():
        parent, _, field = name.rpartition(".")
        setattr(model.get_submodule(parent), field, value)
    return model


def _copy_layers(layers, values: dict, prefix: str) -> None:
    for index, layer in enumerate(layers):
        layer.input_layernorm.weight.copy_(values[prefix + "ln1"][index])
        layer.post_attention_layernorm.weight.copy_(values[prefix + "ln2"][index])
        for name in ("q", "k", "v", "o"):
            getattr(layer.self_attn, name + "_proj").weight.copy_(values[prefix + "w" + name][index])
        layer.mlp.gate_proj.weight.copy_(values[prefix + "wg"][index])
        layer.mlp.up_proj.weight.copy_(values[prefix + "wu"][index])
        layer.mlp.down_proj.weight.copy_(values[prefix + "wd"][index])


def hf_model(case: Case, values: dict, device: str = "cpu"):
    from transformers.models.csm.modeling_csm import CsmForConditionalGeneration

    with torch.device(device):
        model = serving_model(CsmForConditionalGeneration(hf_config(case)))
    with torch.no_grad():
        backbone, depth = model.backbone_model, model.depth_decoder
        backbone.embed_tokens.embed_audio_tokens.weight.copy_(values["embed_audio"])
        depth.model.embed_tokens.weight.copy_(values["embed_audio"])
        backbone.norm.weight.copy_(values["fnorm"])
        model.lm_head.weight.copy_(values["lm_head"])
        _copy_layers(backbone.layers, values, "")
        depth.model.inputs_embeds_projector.weight.copy_(values["depth_proj"])
        depth.model.norm.weight.copy_(values["depth_fnorm"])
        depth.codebooks_head.weight.copy_(values["codebooks_head"])
        _copy_layers(depth.model.layers, values, "depth_")
    return model


def hf_frame(model, case: Case, values: dict) -> dict:
    """One greedy frame through the upstream modules, as ``_sample`` runs it."""
    from transformers import DynamicCache

    p = case.params
    cache = DynamicCache()
    for layer in range(p["layers"]):
        cache.update(values["kcache"][layer].permute(1, 0, 2)[None].clone(),
                     values["vcache"][layer].permute(1, 0, 2)[None].clone(), layer)
    with torch.no_grad():
        out = model(input_ids=values["prev_codes"].reshape(1, 1, -1), past_key_values=cache,
                    use_cache=True, output_hidden_states=True)
        backbone_logits = out.logits[0, -1].float()
        code0 = backbone_logits.argmax()
        hidden = out.hidden_states[-1][:, -1, :]
        generated = model.depth_decoder.generate(
            input_ids=torch.nn.functional.pad(code0.reshape(1, 1), (1, 0), value=0),
            backbone_last_hidden_state=hidden.clone(), do_sample=False,
            min_new_tokens=p["codebooks"] - 1, max_new_tokens=p["codebooks"] - 1,
            return_dict_in_generate=True, output_logits=True)
        codes = generated.sequences[0, 1:]
        greedy_logits = torch.stack([backbone_logits] + [s[0].float() for s in generated.logits])
        forced = model.depth_decoder(
            input_ids=torch.nn.functional.pad(codes[:-1].reshape(1, -1), (1, 0), value=0),
            backbone_last_hidden_state=hidden.clone())
        forced_logits = torch.cat((backbone_logits[None], forced.logits[0].float()))
    layers = range(p["layers"])
    return {"codes": codes, "logits": greedy_logits, "forced_logits": forced_logits,
            "k_write": torch.stack([cache.layers[i].keys[0, :, -1] for i in layers]),
            "v_write": torch.stack([cache.layers[i].values[0, :, -1] for i in layers])}


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = actual.double() - expected.double()
    return {"relative_l2": float(difference.norm() / expected.double().norm()),
            "max_abs": float(difference.abs().max())}


def compare(case: Case, values: dict, model) -> dict:
    """Raise on disagreement; return error statistics."""
    upstream = hf_frame(model, case, values)
    ours = csm.reference(case, values)
    rtol, atol = case.bf16_rtol, case.atol
    report = {"codes_equal": bool(torch.equal(ours["codes"], upstream["codes"]))}
    if not report["codes_equal"]:
        # A BF16 near tie may separate the trajectories; grade both on the
        # upstream codes, which is what the harness does for submissions.
        report["first_code_mismatch"] = int((ours["codes"] != upstream["codes"]).nonzero()[0])
    forced = csm.reference(case, values, forced_codes=upstream["codes"])
    # Upstream's incremental generate and its parallel teacher-forced pass use
    # different GEMM shapes, so on GPU their BF16 logits can differ by a few
    # ULPs. That self-disagreement scales the band, as in the harness grader.
    noise = _errors(upstream["forced_logits"], upstream["logits"])
    report["upstream_schedule_error"] = noise
    for name in ("forced_logits", "logits"):
        error = _errors(forced["logits"], upstream[name])
        peak = float(upstream[name].abs().max())
        if (error["relative_l2"] > 2 * noise["relative_l2"] + rtol
                or error["max_abs"] > 2 * noise["max_abs"] + atol + rtol * peak):
            raise AssertionError(f"logits differ from upstream {name}: {error}, upstream noise {noise}")
    for name in ("k_write", "v_write"):
        torch.testing.assert_close(ours[name], upstream[name], rtol=rtol, atol=atol)
    if not report["codes_equal"]:
        index = report["first_code_mismatch"]
        top = upstream["logits"][index].topk(2).values
        if float(top[0] - top[1]) > 4 * rtol * float(top[0].abs()) + atol:
            raise AssertionError(f"codes diverge at codebook {index} without a near tie")
    report["max_logit_error"] = float((forced["logits"] - upstream["logits"]).abs().max())
    report["max_forced_logit_error"] = float(
        (forced["logits"] - upstream["forced_logits"]).abs().max())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="full CSM-1B geometry")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    args = parser.parse_args()
    pins = check_sources()
    case = full_case() if args.full else development()
    result = {"case": case.id, "tier": "full_geometry" if args.full else "development_geometry",
              "params": case.params, "primary_sources": pins, "trials": []}
    ok = True
    for seed in args.seeds:
        values = csm.make_inputs(case, seed, args.device)
        before = {name: value.clone() for name, value in values.items()}
        model = hf_model(case, values, args.device)
        try:
            trial = compare(case, values, model)
            if not all(torch.equal(values[name], before[name]) for name in values):
                raise AssertionError("reference mutated its inputs")
            trial["pass"] = True
        except AssertionError as error:
            trial, ok = {"pass": False, "error": str(error)}, False
        result["trials"].append({"seed": seed, **trial})
        del model, values, before
    print(json.dumps(result | {"pass": ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
