"""A complete EAGLE3 draft-tree, Llama target verify, and commit iteration.

The draft head fuses three previous target features, normalizes hidden state
and target token embedding separately, and projects their concatenation.
Draft tokens map explicitly into the target vocabulary. Tree construction,
all draft passes, full target verification, selection, and both KV commits
are inside reference(). No target logits or proposals are fixture inputs.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from ..cases import Case
from . import eagle3
from .common import llama31_rope, random_norm, random_weight, rms, rope


def make_inputs(case: Case, seed: int, device: str) -> dict[str, torch.Tensor]:
    p = case.params
    # Share the existing target fixture, but never hand precomputed proposals
    # to the full-iteration candidate. Target and draft are independent heads.
    target_case = replace(case, params=p | {"draft_vocab": p["draft_vocab"]})
    values = eagle3.make_inputs(target_case, seed, device, proposals=False)
    g = torch.Generator(device=device).manual_seed(seed ^ 0xEA613F)
    h, d = p["hidden"], p["head_dim"]
    qdim, kdim = p["q_heads"] * d, p["kv_heads"] * d
    inter = p["intermediate"]
    values["target_features"] = random_weight(g, device, (3, h), h)
    for name, shape, fan_in in (
        ("draft_fc", (h, 3 * h), 3 * h),
        ("draft_wq", (qdim, 2 * h), 2 * h),
        ("draft_wk", (kdim, 2 * h), 2 * h),
        ("draft_wv", (kdim, 2 * h), 2 * h),
        ("draft_wo", (h, qdim), qdim),
        ("draft_wg", (inter, h), h), ("draft_wu", (inter, h), h),
        ("draft_wd", (h, inter), inter),
        ("draft_lm_head", (p["draft_vocab"], h), h),
    ):
        values[name] = random_weight(g, device, shape, fan_in)
    for name in ("draft_hidden_norm", "draft_input_norm", "draft_post_norm", "draft_final_norm"):
        values[name] = random_norm(g, device, (h,))
    values["draft_vocab_map"] = torch.randperm(p["vocab"], generator=g, device=device)[:p["draft_vocab"]]
    shape = (p["context"], p["kv_heads"], d)
    values["draft_kcache"] = random_weight(g, device, shape, d)
    values["draft_vcache"] = random_weight(g, device, shape, d)
    return values


def draft_step(case: Case, values: dict[str, torch.Tensor], token: torch.Tensor,
               hidden: torch.Tensor, keys: torch.Tensor, content: torch.Tensor,
               position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = case.params
    qh, kvh, d = p["q_heads"], p["kv_heads"], p["head_dim"]
    embedding = rms(values["embed"][token].float(), values["draft_input_norm"], eps=1e-5)
    normalized = rms(hidden, values["draft_hidden_norm"], eps=1e-5)
    combined = torch.cat((embedding, normalized))
    query = (values["draft_wq"].float() @ combined).view(qh, d)
    key = (values["draft_wk"].float() @ combined).view(kvh, d)
    value = (values["draft_wv"].float() @ combined).view(kvh, d)
    pos = torch.tensor([position], device=hidden.device)
    query, key = rope(query, pos, 10_000.0), rope(key, pos, 10_000.0)
    all_keys, all_content = torch.cat((keys.float(), key[None])), torch.cat((content.float(), value[None]))
    grouped = query.view(kvh, qh // kvh, d)
    probability = (torch.einsum("gqd,tgd->gqt", grouped, all_keys) * d ** -0.5).softmax(-1)
    attention = torch.einsum("gqt,tgd->gqd", probability, all_content).flatten()
    hidden = hidden + values["draft_wo"].float() @ attention
    normalized = rms(hidden, values["draft_post_norm"], eps=1e-5)
    hidden = hidden + values["draft_wd"].float() @ (
        F.silu(values["draft_wg"].float() @ normalized) *
        (values["draft_wu"].float() @ normalized))
    return hidden, key.to(torch.bfloat16), value.to(torch.bfloat16)


def draft_tree(case: Case, values: dict[str, torch.Tensor]) -> dict:
    p = case.params
    budget, width = p["draft_depth"], p.get("tree_width", 1)
    tokens, parents, depths = [values["token"]], [-1], [0]
    incoming = [values["draft_fc"].float() @ values["target_features"].float().flatten()]
    hidden, k_writes, v_writes = [], [], []
    node = 0
    while node < len(tokens):
        path, parent = [], parents[node]
        while parent >= 0:
            path.append(parent)
            parent = parents[parent]
        path.reverse()
        keys = torch.cat((values["draft_kcache"], torch.stack([k_writes[i] for i in path]))) if path else values["draft_kcache"]
        content = torch.cat((values["draft_vcache"], torch.stack([v_writes[i] for i in path]))) if path else values["draft_vcache"]
        state, key, value = draft_step(case, values, tokens[node], incoming[node],
                                      keys, content, p["context"] + depths[node])
        hidden.append(state)
        k_writes.append(key)
        v_writes.append(value)
        if len(tokens) <= budget:
            scores = values["draft_lm_head"].float() @ rms(state, values["draft_final_norm"], eps=1e-5)
            count = min(width, budget + 1 - len(tokens), p["draft_vocab"])
            chosen = torch.argsort(scores, descending=True, stable=True)[:count]
            for draft_id in chosen:
                tokens.append(values["draft_vocab_map"][draft_id])
                parents.append(node)
                depths.append(depths[node] + 1)
                incoming.append(state)
        node += 1
    device = values["token"].device
    return {"tokens": torch.stack(tokens), "parents": torch.tensor(parents, device=device),
            "depths": torch.tensor(depths, device=device), "hidden": torch.stack(hidden),
            "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes)}


def target_tree(case: Case, values: dict[str, torch.Tensor], tree: dict) -> dict:
    p = case.params
    qh, kvh, d = p["q_heads"], p["kv_heads"], p["head_dim"]
    context, steps = p["context"], tree["tokens"].numel()
    x = values["embed"][tree["tokens"]].float()
    positions = tree["depths"] + context
    allowed = torch.zeros((steps, context + steps), dtype=torch.bool, device=x.device)
    allowed[:, :context] = True
    for node in range(steps):
        parent = node
        while parent >= 0:
            allowed[node, context + parent] = True
            parent = int(tree["parents"][parent])
    k_writes, v_writes, features = [], [], []
    # EAGLE3 fuses the states entering target layers 2, L//2, and L-3.
    # Clamp only for development geometries with fewer than three layers.
    feature_layers = (min(2, p["layers"] - 1), p["layers"] // 2, max(0, p["layers"] - 3))
    for layer in range(p["layers"]):
        features.append(x)
        normalized = rms(x, values["ln1"][layer], eps=1e-5)
        query = (normalized @ values["wq"][layer].float().T).view(steps, qh, d)
        key = (normalized @ values["wk"][layer].float().T).view(steps, kvh, d)
        content = (normalized @ values["wv"][layer].float().T).view(steps, kvh, d)
        query, key = llama31_rope(query, positions), llama31_rope(key, positions)
        k_writes.append(key.to(torch.bfloat16))
        v_writes.append(content.to(torch.bfloat16))
        keys = torch.cat((values["kcache"][layer].float(), key))
        all_content = torch.cat((values["vcache"][layer].float(), content))
        scores = torch.einsum("tgqd,sgd->tgqs", query.view(steps, kvh, qh // kvh, d), keys) * d ** -0.5
        probability = scores.masked_fill(~allowed[:, None, None], -torch.inf).softmax(-1)
        attention = torch.einsum("tgqs,sgd->tgqd", probability, all_content).reshape(steps, qh * d)
        x = x + attention @ values["wo"][layer].float().T
        normalized = rms(x, values["ln2"][layer], eps=1e-5)
        x = x + (F.silu(normalized @ values["wg"][layer].float().T) *
                 (normalized @ values["wu"][layer].float().T)) @ values["wd"][layer].float().T
    logits = rms(x, values["fnorm"], eps=1e-5) @ values["lm_head"].float().T
    return {"logits": logits, "k_write": torch.stack(k_writes), "v_write": torch.stack(v_writes),
            "features": torch.stack([features[index] for index in feature_layers], dim=1)}


def reference(case: Case, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tree = draft_tree(case, values)
    target = target_tree(case, values, tree)
    greedy = target["logits"].argmax(-1)
    path, committed = [0], []
    node = 0
    while True:
        children = torch.nonzero(tree["parents"] == node).flatten()
        matched = [int(child) for child in children if bool(tree["tokens"][child] == greedy[node])]
        if not matched:
            committed.append(greedy[node])
            break
        node = matched[0]
        committed.append(tree["tokens"][node])
        path.append(node)
    device = values["token"].device
    accepted = len(path) - 1
    indices = torch.tensor(path, device=device)
    padded_tokens = torch.full((case.params["draft_depth"] + 1,), -1, device=device, dtype=torch.int64)
    padded_tokens[:len(committed)] = torch.stack(committed)
    k_write, v_write = torch.zeros_like(target["k_write"]), torch.zeros_like(target["v_write"])
    k_write[:, :len(path)], v_write[:, :len(path)] = target["k_write"][:, indices], target["v_write"][:, indices]
    draft_k, draft_v = torch.zeros_like(tree["k_write"]), torch.zeros_like(tree["v_write"])
    draft_k[:len(path)], draft_v[:len(path)] = tree["k_write"][indices], tree["v_write"][indices]
    return {"logits": target["logits"], "proposed_tokens": tree["tokens"][1:],
            "tree_parents": tree["parents"], "accepted_count": torch.tensor(accepted, device=device),
            "proposed_count": torch.tensor(case.params["draft_depth"], device=device),
            "committed_count": torch.tensor(len(committed), device=device), "committed_tokens": padded_tokens,
            "cache_length": torch.tensor(case.params["context"] + len(path), device=device),
            "draft_cache_length": torch.tensor(case.params["context"] + len(path), device=device),
            "target_features": target["features"][node].to(torch.bfloat16),
            "draft_hidden": tree["hidden"][node].to(torch.bfloat16),
            "k_write": k_write, "v_write": v_write,
            "draft_k_write": draft_k, "draft_v_write": draft_v}


def set_acceptance_scenario(case: Case, values: dict[str, torch.Tensor], accepted_prefix: int) -> None:
    """Choose head weights, rather than supplied proposals, for acceptance tests.

    The target's zero head ties to token zero. A small linear system chooses
    the actual draft head to propose zero for the requested prefix and one
    afterward. Both decoders still compute all hidden/KV outputs. Random raw
    trials separately exercise nonzero target heads and unmodified drafting.
    """
    p = case.params
    if p.get("tree_width", 1) != 1 or not 0 <= accepted_prefix <= p["draft_depth"]:
        raise ValueError("controlled acceptance requires a valid linear-chain prefix")
    values["lm_head"].zero_()
    values["draft_vocab_map"] = torch.arange(p["draft_vocab"], device=values["token"].device)
    state = values["draft_fc"].float() @ values["target_features"].float().flatten()
    token = values["token"]
    keys, content = values["draft_kcache"], values["draft_vcache"]
    rows = []
    for index in range(p["draft_depth"]):
        state, key, value = draft_step(case, values, token, state, keys, content, p["context"] + index)
        rows.append(rms(state, values["draft_final_norm"], eps=1e-5))
        keys, content = torch.cat((keys, key[None])), torch.cat((content, value[None]))
        token = torch.tensor(0 if index < accepted_prefix else 1, device=token.device)
    matrix = torch.stack(rows).double()
    target = torch.tensor([4.0 if index < accepted_prefix else -4.0
                           for index in range(p["draft_depth"])], device=matrix.device, dtype=torch.float64)
    weights = matrix.T @ torch.linalg.solve(matrix @ matrix.T, target)
    values["draft_lm_head"].zero_()
    values["draft_lm_head"][0].copy_(weights)
    # The regression must produce the requested bucket after quantization.
    proposal = draft_tree(case, values)["tokens"][1:]
    actual = int((proposal == 0).to(torch.int64).cumprod(0).sum().item())
    if actual != accepted_prefix:
        raise AssertionError(f"draft-head acceptance fixture: expected {accepted_prefix}, got {actual}")
