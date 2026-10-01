"""Llama 3.1 target verification with synthetic EAGLE3-head proposals.

The timed task starts after drafting. The proposal fixture uses a one-layer,
feature-fused EAGLE3-shaped head; its weights and feature inputs are generated
from the same seed and are deliberately outside the timed target call. This is
an architecture/semantics tier, not a substitute for paired checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..cases import Case
from .common import fill_layer_weights, random_norm, random_weight, rms, rope


def _eagle3_proposals(case: Case, seed: int, token: torch.Tensor,
                      device: str) -> torch.Tensor:
    """Produce a linear chain from a feature-fused, one-layer draft head."""
    p = case.params
    h, d, qh, kvh = p["hidden"], p["head_dim"], p["q_heads"], p["kv_heads"]
    inter, depth, draft_vocab = p["intermediate"], p["draft_depth"], p["draft_vocab"]
    g = torch.Generator(device=device).manual_seed(seed ^ 0xEA613)
    features = random_weight(g, device, (3 * h,), h).float()
    fused = random_weight(g, device, (h, 3 * h), 3 * h).float() @ features
    embedding = random_weight(g, device, (draft_vocab, h), h)
    head = random_weight(g, device, (draft_vocab, h), h)
    wq = random_weight(g, device, (qh * d, h), h)
    wk = random_weight(g, device, (kvh * d, h), h)
    wv = random_weight(g, device, (kvh * d, h), h)
    wo = random_weight(g, device, (h, qh * d), qh * d)
    wg = random_weight(g, device, (inter, h), h)
    wu = random_weight(g, device, (inter, h), h)
    wd = random_weight(g, device, (h, inter), inter)
    n1 = random_norm(g, device, (h,))
    n2 = random_norm(g, device, (h,))
    nf = random_norm(g, device, (h,))
    keys, values, proposals = [], [], []
    current = int(token.item()) % draft_vocab
    state = fused
    for step in range(depth):
        x = state + embedding[current].float()
        xn = rms(x, n1, eps=1e-5)
        q = (wq.float() @ xn).view(qh, d)
        k = (wk.float() @ xn).view(kvh, d)
        v = (wv.float() @ xn).view(kvh, d)
        pos = torch.tensor([p["context"] + step], device=device)
        q = rope(q, pos, 500_000.0)
        k = rope(k, pos, 500_000.0)
        keys.append(k)
        values.append(v)
        ks, vs = torch.stack(keys), torch.stack(values)
        grouped = q.view(kvh, qh // kvh, d)
        prob = (torch.einsum("gqd,tgd->gqt", grouped, ks) * d ** -0.5).softmax(-1)
        attn = torch.einsum("gqt,tgd->gqd", prob, vs).reshape(qh * d)
        x = x + wo.float() @ attn
        xn = rms(x, n2, eps=1e-5)
        state = x + wd.float() @ (F.silu(wg.float() @ xn) * (wu.float() @ xn))
        current = int((head.float() @ rms(state, nf, eps=1e-5)).argmax().item())
        proposals.append(current)
    return torch.tensor(proposals, dtype=torch.int64, device=device)


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    if p["batch"] != 1:
        raise NotImplementedError("EAGLE3 target fixture currently handles batch one")
    h, layers, inter, d = (p["hidden"], p["layers"], p["intermediate"],
                           p["head_dim"])
    qdim, kdim = p["q_heads"] * d, p["kv_heads"] * d
    g = torch.Generator(device=device).manual_seed(seed)
    token = torch.randint(p["vocab"], (), generator=g, device=device,
                          dtype=torch.int64)
    values = {
        "token": token,
        "draft_tokens": _eagle3_proposals(case, seed, token, device),
        "ln1": random_norm(g, device, (layers, h)),
        "ln2": random_norm(g, device, (layers, h)),
        "fnorm": random_norm(g, device, (h,)),
        "wq": fill_layer_weights(g, device, (layers, qdim, h), h),
        "wk": fill_layer_weights(g, device, (layers, kdim, h), h),
        "wv": fill_layer_weights(g, device, (layers, kdim, h), h),
        "wo": fill_layer_weights(g, device, (layers, h, qdim), qdim),
        "wg": fill_layer_weights(g, device, (layers, inter, h), h),
        "wu": fill_layer_weights(g, device, (layers, inter, h), h),
        "wd": fill_layer_weights(g, device, (layers, h, inter), inter),
        "embed": random_weight(g, device, (p["vocab"], h), h),
        "lm_head": random_weight(g, device, (p["vocab"], h), h),
    }
    cache_shape = (layers, p["context"], p["kv_heads"], d)
    values["kcache"] = random_weight(g, device, cache_shape, d)
    values["vcache"] = random_weight(g, device, cache_shape, d)
    return values


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    p = case.params
    depth, context, d = p["draft_depth"], p["context"], p["head_dim"]
    qh, kvh = p["q_heads"], p["kv_heads"]
    token_ids = torch.cat((values["token"].reshape(1), values["draft_tokens"]))
    steps = depth + 1
    x = values["embed"][token_ids].float()
    positions = torch.arange(context, context + steps, device=x.device)
    future_mask = (torch.arange(context + steps, device=x.device)[None, :] >
                   positions[:, None])
    k_writes, v_writes = [], []
    layer_hidden = []

    for layer in range(p["layers"]):
        xn = rms(x, values["ln1"][layer], eps=1e-5)
        q = (xn @ values["wq"][layer].float().T).view(steps, qh, d)
        k = (xn @ values["wk"][layer].float().T).view(steps, kvh, d)
        v = (xn @ values["wv"][layer].float().T).view(steps, kvh, d)
        q = rope(q, positions, 500_000.0)
        k = rope(k, positions, 500_000.0)
        k_writes.append(k.to(torch.bfloat16))
        v_writes.append(v.to(torch.bfloat16))
        ks = torch.cat((values["kcache"][layer].float(), k), dim=0)
        vs = torch.cat((values["vcache"][layer].float(), v), dim=0)
        grouped = q.view(steps, kvh, qh // kvh, d)
        scores = torch.einsum("tgqd,sgd->tgqs", grouped, ks) * d ** -0.5
        scores = scores.masked_fill(future_mask[:, None, None, :], -torch.inf)
        probs = scores.softmax(dim=-1)
        attn = torch.einsum("tgqs,sgd->tgqd", probs, vs).reshape(steps, qh * d)
        x = x + attn @ values["wo"][layer].float().T
        xn = rms(x, values["ln2"][layer], eps=1e-5)
        gate = xn @ values["wg"][layer].float().T
        up = xn @ values["wu"][layer].float().T
        x = x + (F.silu(gate) * up) @ values["wd"][layer].float().T
        layer_hidden.append(x)

    logits = rms(x, values["fnorm"], eps=1e-5) @ values["lm_head"].float().T
    greedy = logits.argmax(-1)
    accepted = (greedy[:depth] == values["draft_tokens"]).to(torch.int64).cumprod(0).sum()
    count = int(accepted.item())
    committed = torch.full((steps,), -1, dtype=torch.int64, device=x.device)
    if count:
        committed[:count] = values["draft_tokens"][:count]
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


def set_acceptance_scenario(case: Case, values: dict[str, torch.Tensor],
                            accepted_prefix: int) -> None:
    """Construct a target-derived acceptance bucket for correctness trials.

    The normal fixture uses raw EAGLE3-head proposals. Controlled trials edit
    those proposals so the verifier must handle both long acceptance and
    rollback. This setup is outside the timed target-verification call.
    """
    depth = case.params["draft_depth"]
    if not 0 <= accepted_prefix <= depth:
        raise ValueError("accepted prefix is outside the proposal depth")
    for index in range(accepted_prefix):
        predicted = reference(case, values)["logits"][index].argmax()
        values["draft_tokens"][index] = predicted
    if accepted_prefix < depth:
        predicted = reference(case, values)["logits"][accepted_prefix].argmax()
        values["draft_tokens"][accepted_prefix] = (
            predicted + 1) % case.params["vocab"]
