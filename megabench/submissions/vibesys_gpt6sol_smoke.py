"""One-program-per-row fused implementations for the five smoke families."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _mv(W, vec, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    w = tl.load(W + r[:, None] * N + c[None, :], (r[:, None] < M) & (c[None, :] < N), 0).to(tl.float32)
    return tl.sum(w * vec[None, :], 1)


@tr.jit
def _norm(v, W, H: tl.constexpr):
    h = tl.arange(0, H)
    scale = tl.rsqrt(tl.sum(v * v, 0) / H + 1e-5)
    return v * scale * tl.load(W + h).to(tl.float32)


@tr.jit
def _silu(v):
    return v / (1.0 + tl.exp(-v))


@tr.jit
def _stencil(X, A, Y, N: tl.constexpr, T: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, N)
    v = tl.load(X + b * N + i)
    a = tl.load(A)
    for _ in range(T):
        left = tl.gather(v, (i + N - 1) % N, 0)
        right = tl.gather(v, (i + 1) % N, 0)
        v = (1.0 - 2.0 * a) * v + a * (left + right)
    tl.store(Y + b * N + i, v)


@tr.jit
def _decoder(X, KC, VC, N1, N2, WQ, WK, WV, WO, WG, WU, WD, Y, KO, VO,
             H: tl.constexpr, S: tl.constexpr, I: tl.constexpr, SP: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    j = tl.arange(0, I)
    s = tl.arange(0, SP)
    x = tl.load(X + b * H + h).to(tl.float32)
    n = _norm(x, N1, H)
    q = _mv(WQ, n, H, H, H, H)
    k = _mv(WK, n, H, H, H, H)
    v = _mv(WV, n, H, H, H, H)
    oldk = tl.load(KC + b * S * H + s[:, None] * H + h[None, :], s[:, None] < S, 0).to(tl.float32)
    oldv = tl.load(VC + b * S * H + s[:, None] * H + h[None, :], s[:, None] < S, 0).to(tl.float32)
    keys = tl.where(s[:, None] == S, k[None, :], oldk)
    vals = tl.where(s[:, None] == S, v[None, :], oldv)
    tl.store(KO + b * (S + 1) * H + s[:, None] * H + h[None, :], keys, s[:, None] <= S)
    tl.store(VO + b * (S + 1) * H + s[:, None] * H + h[None, :], vals, s[:, None] <= S)
    score = tl.sum(keys * q[None, :], 1) * (H ** -0.5)
    score = tl.where(s <= S, score, -float('inf'))
    weight = tl.exp(score - tl.max(score, 0))
    weight = weight / tl.sum(weight, 0)
    context = tl.sum(weight[:, None] * vals, 0)
    z = x + _mv(WO, context, H, H, H, H)
    n2 = _norm(z, N2, H)
    gate = _silu(_mv(WG, n2, I, H, I, H))
    up = _mv(WU, n2, I, H, I, H)
    out = z + _mv(WD, gate * up, H, I, H, I)
    tl.store(Y + b * H + h, out)


@tr.jit
def _moe(X, ROUTER, UP, DOWN, Y, IDS, H: tl.constexpr, E: tl.constexpr, I: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    e = tl.arange(0, E)
    x = tl.load(X + b * H + h).to(tl.float32)
    scores = _mv(ROUTER, x, E, H, E, H)
    first = tl.argmax(scores, 0)
    second = tl.argmax(tl.where(e == first, -float('inf'), scores), 0)
    tl.store(IDS + b * 2, first)
    tl.store(IDS + b * 2 + 1, second)
    vmax = tl.maximum(tl.sum(tl.where(e == first, scores, 0.), 0), tl.sum(tl.where(e == second, scores, 0.), 0))
    p0 = tl.exp(tl.sum(tl.where(e == first, scores, 0.), 0) - vmax)
    p1 = tl.exp(tl.sum(tl.where(e == second, scores, 0.), 0) - vmax)
    up0 = _silu(_mv(UP + first * I * H, x, I, H, I, H))
    up1 = _silu(_mv(UP + second * I * H, x, I, H, I, H))
    down0 = _mv(DOWN + first * H * I, up0, H, I, H, I)
    down1 = _mv(DOWN + second * H * I, up1, H, I, H, I)
    tl.store(Y + b * H + h, x + (p0 * down0 + p1 * down1) / (p0 + p1))


@tr.jit
def _qmv(W, scale, vec, M: tl.constexpr, N: tl.constexpr, BITS: tl.constexpr,
         R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    if BITS == 8:
        q = tl.load(W + r[:, None] * N + c[None, :], (r[:, None] < M) & (c[None, :] < N), 0).to(tl.float32)
    else:
        packed = tl.load(W + r[:, None] * (N // 2) + c[None, :] // 2,
                         (r[:, None] < M) & (c[None, :] < N), 0).to(tl.int32)
        q = tl.where(c[None, :] % 2 == 0, packed & 15, (packed >> 4) & 15).to(tl.float32) - 8.0
    sc = tl.load(scale + r, r < M, 0)
    return tl.sum(q * vec[None, :], 1) * sc


@tr.jit
def _quant(X, NORM, WG, WU, WD, SG, SU, SD, Y,
           H: tl.constexpr, I: tl.constexpr, BITS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    x = tl.load(X + b * H + h).to(tl.float32)
    n = _norm(x, NORM, H)
    gate = _silu(_qmv(WG, SG, n, I, H, BITS, I, H))
    up = _qmv(WU, SU, n, I, H, BITS, I, H)
    down = _qmv(WD, SD, gate * up, H, I, BITS, H, I)
    tl.store(Y + b * H + h, x + down)


@tr.jit
def _spec(D, T, AC, CC, CT, K: tl.constexpr, V: tl.constexpr, KP: tl.constexpr):
    b = tl.program_id(0)
    steps = tl.arange(0, KP)
    tok = tl.arange(0, V)
    draft = tl.load(D + b * K * V + steps[:, None] * V + tok[None, :], steps[:, None] < K, -float('inf'))
    target = tl.load(T + b * (K + 1) * V + steps[:, None] * V + tok[None, :], steps[:, None] <= K, -float('inf'))
    did = tl.argmax(draft, 1, tie_break_left=True)
    tid = tl.argmax(target, 1, tie_break_left=True)
    match = (did == tid) & (steps < K)
    # First mismatch is the acceptance length.
    first_bad = tl.min(tl.where(~match & (steps < K), steps, K), 0)
    tl.store(AC + b, first_bad.to(tl.int64))
    tl.store(CC + b, (first_bad + 1).to(tl.int64))
    committed = tl.where(steps < first_bad, did, tl.where(steps == first_bad, tid, -1))
    tl.store(CT + b * (K + 1) + steps, committed.to(tl.int64), steps <= K)


def build(case: dict):
    family = case['family']
    p = case['params']
    def run(t: dict):
        if family == 'stencil':
            y = torch.empty_like(t['x'])
            _stencil[(p['batch'],)](t['x'], t['alpha'], y, p['width'], p['steps'])
            return {'y': y}
        if family == 'decoder':
            b, h, s, i = (p[k] for k in ('batch','hidden','context','intermediate'))
            y = torch.empty((b,h), device=t['x'].device, dtype=torch.bfloat16)
            ko = torch.empty((b,s+1,h), device=t['x'].device, dtype=torch.bfloat16)
            vo = torch.empty_like(ko)
            _decoder[(b,)](*(t[k] for k in ('x','kcache','vcache','norm1','norm2','wq','wk','wv','wo','wg','wu','wd')), y, ko, vo, h, s, i, tr.next_power_of_2(s+1), num_warps=8)
            return {'y':y,'kcache':ko,'vcache':vo}
        if family == 'moe':
            b,h,e,i = (p[k] for k in ('batch','hidden','experts','intermediate'))
            y = torch.empty((b,h),device=t['x'].device,dtype=torch.bfloat16)
            ids = torch.empty((b,2),device=t['x'].device,dtype=torch.int32)
            _moe[(b,)](t['x'],t['router'],t['w_up'],t['w_down'],y,ids,h,e,i,num_warps=8)
            return {'y':y,'expert_ids':ids}
        if family == 'quant_mlp':
            b,h,i,bits = (p[k] for k in ('batch','hidden','intermediate','bits'))
            y = torch.empty((b,h),device=t['x'].device,dtype=torch.bfloat16)
            _quant[(b,)](*(t[k] for k in ('x','norm','w_gate','w_up','w_down','s_gate','s_up','s_down')),y,h,i,bits,num_warps=8)
            return {'y':y}
        if family == 'spec_verify':
            b,k,v = (p[key] for key in ('batch','draft_len','vocab'))
            ac = torch.empty((b,),device=t['draft_logits'].device,dtype=torch.int64)
            cc = torch.empty_like(ac)
            ct = torch.empty((b,k+1),device=ac.device,dtype=torch.int64)
            _spec[(b,)](t['draft_logits'],t['target_logits'],ac,cc,ct,k,v,tr.next_power_of_2(k+1))
            return {'accepted_count':ac,'committed_count':cc,'committed_tokens':ct}
        raise ValueError(family)
    return run
