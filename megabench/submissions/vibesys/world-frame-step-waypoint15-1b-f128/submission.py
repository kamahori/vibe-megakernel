"""Waypoint-1.5-1B frame step as one persistent tcgen05 megakernel (see mk/)."""
import math
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
SIGMAS = (1.0, 0.8984375, 0.75, 0.30078125, 0.0)
GLOBAL_ATTN_OFFSET = -1
TS_MULT = 1
ROPE_THETA = 10000.0
ROPE_NYQUIST_FRAC = 0.8
NOISE_FOURIER_DIM = 512
NOISE_FOURIER_BASE = 10000.0


def _load_ext(timeline=False):
    from torch.utils.cpp_extension import load
    name = "waypoint_mega_tl" if timeline else "waypoint_mega"
    bdir = os.path.join(_HERE, "mk", "_build_tl" if timeline else "_build")
    os.makedirs(bdir, exist_ok=True)
    flags = ["-gencode=arch=compute_100a,code=sm_100a", "-O3", "-std=c++17", "-lineinfo"]
    if timeline:
        flags.append("-DMEGABENCH_TIMELINE")
    return load(name, [os.path.join(_HERE, "mk", "ext.cu")], extra_cuda_cflags=flags,
                extra_include_paths=[os.path.join(_HERE, "mk"), _HERE], build_directory=bdir, verbose=False)


def _is_global(p, layer):
    period = p["global_period"]
    return (layer - (GLOBAL_ATTN_OFFSET % period)) % period == 0


def _ring(p, layer):
    if _is_global(p, layer):
        return p["global_window"] // p["global_dilation"], p["global_dilation"]
    return p["local_window"], 1


def _visible(p, layer, frame):
    slots, dil = _ring(p, layer)
    hidden = None if frame % dil else (frame // dil) % slots
    held = [None] * slots
    first = max(0, (frame - 1) // dil - slots + 1) * dil
    for g in range(first, frame, dil):
        held[(g // dil) % slots] = g
    return [s for s, g in enumerate(held) if g is not None and s != hidden]


def _tables(p, device):
    d = p["head_dim"]
    gh, gw = p["latent_height"] // p["patch"], p["latent_width"] // p["patch"]
    d_xy, d_t = d // 8, d // 4
    max_freq = min(gh, gw) * ROPE_NYQUIST_FRAC
    xy = (torch.linspace(1.0, max_freq / 2, (d_xy + 1) // 2, dtype=torch.float32, device=device) * torch.pi).repeat_interleave(2)[:d_xy]
    inv_t = 1.0 / (ROPE_THETA ** (torch.arange(0, d_t, 2, dtype=torch.float32, device=device) / d_t))
    inv_t = inv_t.repeat_interleave(2)
    token = torch.arange(p["tokens_per_frame"], device=device)
    xs = (2.0 * (token % gw).float() + 1.0) / gw - 1.0
    ys = (2.0 * (token // gw).float() + 1.0) / gh - 1.0
    ts = torch.full_like(xs, float(p["frame_index"] * TS_MULT))
    phase = torch.cat((xs[:, None] * xy, ys[:, None] * xy, ts[:, None] * inv_t), -1)
    cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
    half = NOISE_FOURIER_DIM // 2
    freq = torch.logspace(0, -1, steps=half, base=NOISE_FOURIER_BASE, dtype=torch.float32, device=device)
    rows = []
    for sg in SIGMAS:
        s = torch.full((1,), sg, dtype=torch.bfloat16, device=device).float() * 1000
        ph = s[:, None] * freq[None, :]
        rows.append(torch.cat((ph.sin(), ph.cos()), -1)[0] * 2 ** 0.5)
    return cos, sin, torch.stack(rows).contiguous()


def _make(case, timeline=False):
    p = case["params"]
    assert p["tokens_per_frame"] == 512 and p["hidden"] == 2048 and p["layers"] == 24
    dev = "cuda"
    ext = _load_ext(timeline)
    bf, f32 = torch.bfloat16, torch.float32
    cos, sin, fourier = _tables(p, dev)
    vis = torch.zeros(24, 16, dtype=torch.int32, device=dev)
    nvis = torch.zeros(24, dtype=torch.int32, device=dev)
    for l in range(24):
        v = _visible(p, l, p["frame_index"])
        assert len(v) <= 16
        vis[l, :len(v)] = torch.tensor(v, dtype=torch.int32)
        nvis[l] = len(v)
    z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=dev)
    state = dict(
        x=z(512, 2048), h=z(512, 2048), att=z(512, 2048), mlp=z(512, 8192), patch_in=z(512, 128), qbuf=z(512, 2048),
        kbuf=z(512, 1024), vbuf=z(512, 1024), vfirst=z(512, 1024), acc=z(512, 2048, dt=f32),
        fourier=fourier, rope_cos=cos, rope_sin=sin, n_h1=z(5, 8192, dt=f32), cond=z(5, 2048), ctrl_h1=z(8192),
        ctrl=z(2048), mod=z(5 * 48 * 6144), outmod=z(5, 4096), cvec=z(8, 2048), vis=vis, nvis=nvis,
        bar=torch.zeros(2, dtype=torch.int64, device=dev),
    )
    if timeline:
        state["tl"] = None
    return ext, state


def build(case: dict):
    ext, state = _make(case)
    names = ["noise", "mouse", "button", "scroll", "k_cache", "v_cache", "noise_fc1", "noise_fc2", "ctrl_fc1", "ctrl_fc2",
             "attn_cond_bias", "attn_cond_w", "mlp_cond_bias", "mlp_cond_w", "wq", "wk", "wv", "wo", "v_lamb", "fuse_x",
             "fuse_c", "fuse_out", "fc1", "fc2", "out_norm_w", "unpatch_w", "unpatch_b"]
    p = case["params"]

    def run(inputs):
        d = dict(state)
        for n in names:
            d[n] = inputs[n]
        d["patch_w"] = inputs["patch_w"].view(2048, 128)
        d["latent"] = torch.empty(p["latent_channels"], p["latent_height"], p["latent_width"], dtype=torch.bfloat16, device="cuda")
        d["k_write"] = torch.empty(24, 512, 16, 64, dtype=torch.bfloat16, device="cuda")
        d["v_write"] = torch.empty(24, 512, 16, 64, dtype=torch.bfloat16, device="cuda")
        ext.mega_run(d, False)
        return {"latent": d["latent"], "k_write": d["k_write"], "v_write": d["v_write"]}

    return run


def build_timeline(case: dict):
    from megabench.integrations import timeline as _tl  # noqa: F401  (decoder lives in the framework)
    ext, state = _make(case, timeline=True)
    names = ["noise", "mouse", "button", "scroll", "k_cache", "v_cache", "noise_fc1", "noise_fc2", "ctrl_fc1", "ctrl_fc2",
             "attn_cond_bias", "attn_cond_w", "mlp_cond_bias", "mlp_cond_w", "wq", "wk", "wv", "wo", "v_lamb", "fuse_x",
             "fuse_c", "fuse_out", "fc1", "fc2", "out_norm_w", "unpatch_w", "unpatch_b"]
    p = case["params"]
    words = 1 + 148 * 1 * 8192
    events = ["cond", "patchify", "rms", "qkv", "attn", "wo", "fusion", "fc1", "fc2", "head"]

    def run(inputs):
        d = dict(state)
        for n in names:
            d[n] = inputs[n]
        d["patch_w"] = inputs["patch_w"].view(2048, 128)
        d["latent"] = torch.empty(p["latent_channels"], p["latent_height"], p["latent_width"], dtype=torch.bfloat16, device="cuda")
        d["k_write"] = torch.empty(24, 512, 16, 64, dtype=torch.bfloat16, device="cuda")
        d["v_write"] = torch.empty(24, 512, 16, 64, dtype=torch.bfloat16, device="cuda")
        d["tl"] = torch.zeros(words, dtype=torch.int64, device="cuda")
        ext.mega_run(d, True)
        return d["tl"]

    return run, events
