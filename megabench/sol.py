"""Optimistic one-read HBM traffic floors for ready MegaBench model steps.

This is a useful-byte bound, not a prediction of achievable kernel latency.
It counts each needed weight once, the old KV values needed by the step, and
required outputs. It omits scratch traffic, synchronization, and instruction
cost. P0 model weights are larger than cache, so weight streaming is a useful
first-order bound for those cases.
"""

from __future__ import annotations

import math

from .cases import Case


DEFAULT_B200_BANDWIDTH_TB_S = 8.0
DEFAULT_B200_BANDWIDTH_SOURCE = "https://www.nvidia.com/en-eu/data-center/dgx-b200/"
BF16_BYTES = 2
FP32_BYTES = 4
INT64_BYTES = 8


def estimate_case(case: Case, bandwidth_tb_s: float = DEFAULT_B200_BANDWIDTH_TB_S,
                  measured_ms: float | None = None) -> dict:
    """Return a transparent HBM floor for one ready, batch-one case."""
    if not math.isfinite(bandwidth_tb_s) or bandwidth_tb_s <= 0:
        raise ValueError("bandwidth_tb_s must be finite and positive")
    if measured_ms is not None and (not math.isfinite(measured_ms) or measured_ms <= 0):
        raise ValueError("measured_ms must be finite and positive")
    result: dict = {"case_id": case.id, "status": "estimated"}
    if not case.ready:
        return result | {"status": "unavailable", "reason": case.note}
    if case.family in WORK_MODELS:
        return _work_floor(case, result, bandwidth_tb_s, measured_ms)
    if case.family not in {"dense_step", "moe_step", "quant_step",
                           "spec_target_step"}:
        return result | {"status": "unavailable",
                         "reason": f"no traffic model for {case.family}"}
    p = case.params
    if p["batch"] != 1:
        return result | {"status": "unavailable",
                         "reason": "traffic model currently supports batch one"}

    layers, hidden, head_dim = p["layers"], p["hidden"], p["head_dim"]
    qdim, kdim = p["q_heads"] * head_dim, p["kv_heads"] * head_dim
    inter, vocab, context = p["intermediate"], p["vocab"], p["context"]
    steps = p["draft_depth"] + 1 if case.family == "spec_target_step" else 1
    projection_elements = layers * (
        qdim * hidden + 2 * kdim * hidden + hidden * qdim)
    mlp_elements = layers * 3 * inter * hidden
    components: list[dict] = []

    def add(name: str, bytes_: int, detail: str) -> None:
        components.append({"name": name, "bytes": bytes_, "detail": detail})

    if case.family == "quant_step":
        bits = p["bits"]
        if bits not in (4, 8):
            return result | {"status": "unavailable",
                             "reason": f"unsupported quantization width {bits}"}
        weight_elements = projection_elements + mlp_elements
        if weight_elements * bits % 8:
            raise ValueError(f"packed weights do not contain whole bytes: {case.id}")
        add("quantized_projection_and_mlp_weights", weight_elements * bits // 8,
            f"all seven projection stacks, packed {bits}-bit weights")
        output_rows = layers * (qdim + 2 * kdim + 2 * hidden + 2 * inter)
        add("quantization_scales", output_rows * FP32_BYTES,
            "one FP32 scale per projection output row")
        add("normalization_weights",
            (layers * (4 * hidden + 2 * head_dim) + hidden) * BF16_BYTES,
            "four layer norms, Q/K norms, and final norm in BF16")
        add("tied_embedding_and_head", vocab * hidden * BF16_BYTES,
            "full BF16 vocabulary head, including the token embedding row")
        local_layers = layers - layers // 6
        local_tokens = min(context, max(0, p["local_window"] - 1))
        cache_positions = local_layers * local_tokens + (layers // 6) * context
    else:
        add("attention_projection_weights", projection_elements * BF16_BYTES,
            "Q/K/V/output projection weights, read once")
        cache_positions = layers * context
        if case.family == "moe_step":
            experts, topk = p["experts"], p["topk"]
            if topk > experts:
                raise ValueError(f"topk exceeds expert count: {case.id}")
            add("router_weights", layers * experts * hidden * BF16_BYTES,
                "all router rows in each layer")
            add("active_expert_weights", mlp_elements * topk * BF16_BYTES,
                f"only the {topk} selected experts per layer")
            add("normalization_weights",
                (layers * (2 * hidden + 2 * head_dim) + hidden) * BF16_BYTES,
                "two layer norms, Q/K norms, and final norm in BF16")
            add("input_embedding_row", hidden * BF16_BYTES,
                "one BF16 embedding row")
            add("lm_head", vocab * hidden * BF16_BYTES,
                "separate BF16 vocabulary head")
        elif case.family == "dense_step":
            add("mlp_weights", mlp_elements * BF16_BYTES,
                "gate/up/down weights in every layer")
            add("normalization_weights",
                (layers * (2 * hidden + 2 * head_dim) + hidden) * BF16_BYTES,
                "two layer norms, Q/K norms, and final norm in BF16")
            add("tied_embedding_and_head", vocab * hidden * BF16_BYTES,
                "full BF16 vocabulary head, including the token embedding row")
        else:
            add("mlp_weights", mlp_elements * BF16_BYTES,
                "gate/up/down weights shared across verification positions")
            add("normalization_weights",
                (layers * 2 * hidden + hidden) * BF16_BYTES,
                "two layer norms and final norm in BF16")
            add("input_embedding_rows", steps * hidden * BF16_BYTES,
                "one BF16 embedding row per verified position")
            add("lm_head", vocab * hidden * BF16_BYTES,
                "separate BF16 vocabulary head read once across positions")

    add("old_kv_cache_reads", 2 * cache_positions * kdim * BF16_BYTES,
        "one read of the needed old K and V positions")
    add("logits_writes", steps * vocab * FP32_BYTES,
        "all required FP32 logits")
    add("new_kv_writes", 2 * layers * steps * kdim * BF16_BYTES,
        "required BF16 K and V outputs, including masked speculative slots")
    if case.family == "spec_target_step":
        depth = p["draft_depth"]
        add("token_and_feature_io",
            (steps + 3 + steps) * INT64_BYTES + 3 * hidden * BF16_BYTES,
            "input token/drafts, accepted/committed/cache scalars, "
            "committed tokens, and three target features")
    else:
        add("token_io", 2 * INT64_BYTES, "input and next-token IDs")

    minimum_bytes = sum(item["bytes"] for item in components)
    floor_ms = minimum_bytes / (bandwidth_tb_s * 1e12) * 1e3
    result.update(
        minimum_bytes=minimum_bytes,
        minimum_gb=minimum_bytes / 1e9,
        hbm_floor_ms=floor_ms,
        components=components,
    )
    if measured_ms is not None:
        result.update(measured_ms=measured_ms,
                      gap_to_hbm_floor_x=measured_ms / floor_ms,
                      percent_of_hbm_sol=100 * floor_ms / measured_ms)
    return result


# Dense tensor-core peaks per B200 (the datasheet's sparse figures halved) and
# NVLink 5 bandwidth per direction. FP8 x FP4 MMAs issue at the FP8 rate.
B200_DENSE_TFLOPS = {"bf16": 2250.0, "fp8": 4500.0}
B200_NVLINK_GB_S = 900.0
B200_PEAKS_SOURCE = "https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet"


def _megamoe_work(case: Case) -> tuple[list[dict], list[dict], int]:
    """One rank of the EP routed-expert layer under uniform routing.

    Every local expert is assumed hit, which holds for the catalog's token
    counts. A token crosses NVLink once per distinct remote owner rank, both
    to dispatch FP8 rows and to return one pre-summed BF16 row per owner.
    """
    p = case.params
    tokens, experts, topk = p["tokens"], p["experts"], p["topk"]
    hidden, inter, ep = p["hidden"], p["intermediate"], case.ep
    local = experts // ep
    weight_elements = local * 3 * inter * hidden
    rows = tokens * topk
    bytes_ = [
        {"name": "local_expert_weights", "bytes": weight_elements // 2 + weight_elements // 32,
         "detail": f"{local} MXFP4 experts (E2M1 plus one E8M0 scale per 32)"},
        {"name": "fp8_tokens", "bytes": tokens * (hidden + hidden // 32),
         "detail": "E4M3 rows and per-32-channel E8M0 scales"},
        {"name": "routing", "bytes": tokens * topk * (INT64_BYTES + FP32_BYTES),
         "detail": "top-k expert ids and weights"},
        {"name": "output", "bytes": tokens * hidden * BF16_BYTES, "detail": "BF16 combined rows"},
    ]
    flops = [{"name": "expert_gemms", "dtype": "fp8", "flops": 2 * rows * 3 * inter * hidden,
              "detail": f"{rows} routed rows per rank on average, gate/up and down"}]
    # Expected distinct remote owners per token for distinct uniform experts.
    miss = math.comb(experts - local, topk) / math.comb(experts, topk)
    remote_owners = (ep - 1) * (1 - miss)
    link = round(tokens * remote_owners * (hidden + hidden // 32 + hidden * BF16_BYTES))
    return bytes_, flops, link


def _llama_layer_elements(hidden: int, qdim: int, kvdim: int, inter: int) -> int:
    return hidden * (2 * qdim + 2 * kvdim) + 3 * inter * hidden + 2 * hidden


def _csm_work(case: Case) -> tuple[list[dict], list[dict], int]:
    """One CSM frame: backbone decode, then 31 depth-decoder positions.

    The depth decoder re-reads its ~0.2 GB of weights at each of its 31
    dependent positions; this one-read floor counts them once.
    """
    p = case.params
    hidden, books, vocab = p["hidden"], p["codebooks"], p["audio_vocab"]
    qdim, kvdim = p["q_heads"] * p["head_dim"], p["kv_heads"] * p["head_dim"]
    dhidden = p["depth_hidden"]
    dq, dkv = p["depth_q_heads"] * p["depth_head_dim"], p["depth_kv_heads"] * p["depth_head_dim"]
    backbone = p["layers"] * _llama_layer_elements(hidden, qdim, kvdim, p["intermediate"]) + hidden
    depth = (p["depth_layers"] * _llama_layer_elements(dhidden, dq, dkv, p["depth_intermediate"])
             + dhidden + dhidden * hidden)
    heads = (books - 1) * dhidden * vocab
    bytes_ = [
        {"name": "backbone_weights", "bytes": backbone * BF16_BYTES, "detail": "layers and final norm"},
        {"name": "codebook0_head", "bytes": vocab * hidden * BF16_BYTES, "detail": "backbone lm_head"},
        {"name": "depth_decoder_weights", "bytes": depth * BF16_BYTES,
         "detail": "layers, final norm and input projector, counted once"},
        {"name": "codebook_heads", "bytes": heads * BF16_BYTES, "detail": "one head per codebook 1..31"},
        {"name": "embedding_rows", "bytes": (2 * books - 1) * hidden * BF16_BYTES,
         "detail": "32 frame rows plus 31 depth-input rows"},
        {"name": "old_kv_cache_reads", "bytes": 2 * p["layers"] * p["context"] * kvdim * BF16_BYTES,
         "detail": "backbone K and V"},
        {"name": "outputs", "bytes": books * (INT64_BYTES + vocab * FP32_BYTES)
         + 2 * p["layers"] * kvdim * BF16_BYTES, "detail": "codes, FP32 logits and K/V writes"},
    ]
    # Position 0 of the depth decoder is a two-token prefill.
    flops = [{"name": "projections", "dtype": "bf16",
              "flops": 2 * (backbone + vocab * hidden + (books + 1) * depth + heads),
              "detail": "matrix-vector products; attention is negligible"}]
    return bytes_, flops, 0


def _waypoint_work(case: Case) -> tuple[list[dict], list[dict], int]:
    """Four denoise passes and the sigma-0 cache pass over one frame."""
    from .tasks.waypoint import NOISE_FOURIER_DIM, NOISE_HIDDEN_MULT, MOUSE_DIMS, SCROLL_DIMS
    from .tasks.waypoint import control_layers, visible_slots

    p = case.params
    d, layers, tokens = p["hidden"], p["layers"], p["tokens_per_frame"]
    inner, ctrl_in = d * p["mlp_ratio"], MOUSE_DIMS + p["buttons"] + SCROLL_DIMS
    qdim, kvdim = p["q_heads"] * p["head_dim"], p["kv_heads"] * p["head_dim"]
    fused = len(control_layers(case))
    patch = p["latent_channels"] * p["patch"] ** 2
    linear = 2 * qdim * d + 2 * kvdim * d + 2 * inner * d
    bf16_elements = (layers * (linear + 6 * d * d + 2 * d + 1) + fused * 3 * d * d
                     + inner * (ctrl_in + d) + 2 * d * d + 2 * d * patch + p["latent_channels"])
    noise_hidden = NOISE_HIDDEN_MULT * d
    visible = [len(visible_slots(case, layer, p["frame_index"])) for layer in range(layers)]
    latent = p["latent_channels"] * p["latent_height"] * p["latent_width"]
    bytes_ = [
        {"name": "dit_weights", "bytes": bf16_elements * BF16_BYTES,
         "detail": "blocks, AdaLN condition heads, controller fusion and patch layers"},
        {"name": "noise_mlp", "bytes": noise_hidden * (NOISE_FOURIER_DIM + d) * FP32_BYTES,
         "detail": "FP32 sigma embedding MLP"},
        {"name": "old_kv_cache_reads", "bytes": 2 * sum(visible) * tokens * kvdim * BF16_BYTES,
         "detail": "visible local and dilated global frames, read once"},
        {"name": "io", "bytes": 2 * latent * BF16_BYTES + 2 * layers * tokens * kvdim * BF16_BYTES
         + ctrl_in * BF16_BYTES, "detail": "noise, controller, latent and K/V writes"},
    ]
    attention = sum(4 * tokens * (count + 1) * tokens * qdim for count in visible)
    full = 2 * tokens * (layers * linear + fused * 2 * d * d) + attention
    # The cache pass stops once the last layer's K and V exist.
    last = 2 * tokens * (2 * qdim * d + 2 * inner * d) + 4 * tokens * (visible[-1] + 1) * tokens * qdim
    passes = p["denoise_steps"]
    flops = [{"name": "dit_passes", "dtype": "bf16", "flops": (passes + 1) * full - last,
              "detail": f"{passes} denoise passes and one cache pass over {tokens} tokens"}]
    return bytes_, flops, 0


def _pi05_work(case: Case) -> tuple[list[dict], list[dict], int]:
    """Ten flow-matching steps of the action expert over a cached prefix.

    Time conditioning depends only on the fixed schedule, so the FP32 adaRMS
    weights can be read once for all steps; every weight counts once.
    """
    p = case.params
    w, layers, steps, horizon = p["expert_hidden"], p["layers"], p["denoise_steps"], p["horizon"]
    qdim, kvdim, act = p["q_heads"] * p["head_dim"], p["kv_heads"] * p["head_dim"], p["action_dim"]
    linear = 2 * qdim * w + 2 * kvdim * w + 3 * p["expert_intermediate"] * w
    fp32 = (2 * layers + 1) * 3 * w * (w + 1) + 2 * w * (w + 1) + 2 * act * w + w + act
    prefix = p["images"] * p["image_tokens"] + p["text_tokens"]
    bytes_ = [
        {"name": "expert_weights", "bytes": layers * linear * BF16_BYTES,
         "detail": "Gemma-300M attention and MLP projections"},
        {"name": "fp32_conditioning_and_action_weights", "bytes": fp32 * FP32_BYTES,
         "detail": "adaRMS, time MLP and action projections"},
        {"name": "prefix_kv_reads", "bytes": 2 * layers * prefix * kvdim * BF16_BYTES,
         "detail": "cached PaliGemma prefix K and V, read once"},
        {"name": "io", "bytes": 2 * horizon * act * FP32_BYTES + prefix, "detail": "noise, mask and actions"},
    ]
    attention = 4 * horizon * (prefix + horizon) * qdim * layers
    flops = [{"name": "expert_steps", "dtype": "bf16",
              "flops": steps * (2 * horizon * layers * linear + attention),
              "detail": f"{steps} steps over {horizon} action tokens"}]
    return bytes_, flops, 0


WORK_MODELS = {"megamoe_layer": _megamoe_work, "vla_action_step": _pi05_work, "tts_frame_step": _csm_work,
               "world_frame_step": _waypoint_work}


def _work_floor(case: Case, result: dict, bandwidth_tb_s: float,
                measured_ms: float | None) -> dict:
    """Floor = the largest of the HBM, tensor-core, and NVLink time bounds."""
    components, flops, link_bytes = WORK_MODELS[case.family](case)
    minimum_bytes = sum(item["bytes"] for item in components)
    hbm_ms = minimum_bytes / (bandwidth_tb_s * 1e12) * 1e3
    compute_ms = sum(item["flops"] / (B200_DENSE_TFLOPS[item["dtype"]] * 1e12)
                     for item in flops) * 1e3
    link_ms = link_bytes / (B200_NVLINK_GB_S * 1e9) * 1e3
    bounds = {"hbm": hbm_ms, "tensor_core": compute_ms, "nvlink": link_ms}
    bound = max(bounds, key=bounds.get)
    result.update(minimum_bytes=minimum_bytes, minimum_gb=minimum_bytes / 1e9,
                  hbm_floor_ms=hbm_ms, components=components,
                  tensor_flops=sum(item["flops"] for item in flops), flop_components=flops,
                  compute_floor_ms=compute_ms, nvlink_bytes=link_bytes, nvlink_floor_ms=link_ms,
                  floor_ms=bounds[bound], bound=bound, peaks_source=B200_PEAKS_SOURCE)
    if measured_ms is not None:
        result.update(measured_ms=measured_ms,
                      gap_to_hbm_floor_x=measured_ms / hbm_ms,
                      percent_of_hbm_sol=100 * hbm_ms / measured_ms,
                      gap_to_floor_x=measured_ms / bounds[bound],
                      percent_of_sol=100 * bounds[bound] / measured_ms)
    return result
