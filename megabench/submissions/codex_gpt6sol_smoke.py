"""Single-launch Triton implementations for the five smoke cases."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mv(W, x, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    w = tl.load(W + r[:, None] * C + c[None, :]).to(tl.float32)
    return tl.sum(w * x[None, :], 1)


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _stencil(X, A, Y, W: tl.constexpr, T: tl.constexpr):
    b = tl.program_id(0)
    c = tl.arange(0, W)
    x = tl.load(X + b * W + c)
    a = tl.load(A)
    for _ in range(T):
        left = tl.gather(x, (c + W - 1) % W, 0)
        right = tl.gather(x, (c + 1) % W, 0)
        x = (1.0 - 2.0 * a) * x + a * (left + right)
    tl.store(Y + b * W + c, x)


@triton.jit
def _decoder(X, KC, VC, N1, N2, WQ, WK, WV, WO, WG, WU, WD,
             Y, KOUT, VOUT, H: tl.constexpr, S: tl.constexpr,
             I: tl.constexpr, SP: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    s = tl.arange(0, SP)
    x = tl.load(X + b * H + h).to(tl.float32)
    n1w = tl.load(N1 + h)
    n1 = x * tl.rsqrt(tl.sum(x * x, 0) / H + 1.0e-5) * n1w
    q = _mv(WQ, n1, H, H)
    k = _mv(WK, n1, H, H)
    v = _mv(WV, n1, H, H)
    cache_off = b * S * H + s[:, None] * H + h[None, :]
    kc = tl.load(KC + cache_off, s[:, None] < S, 0).to(tl.float32)
    vc = tl.load(VC + cache_off, s[:, None] < S, 0).to(tl.float32)
    kc = tl.where(s[:, None] == S, k[None, :], kc)
    vc = tl.where(s[:, None] == S, v[None, :], vc)
    out_off = b * (S + 1) * H + s[:, None] * H + h[None, :]
    tl.store(KOUT + out_off, kc, s[:, None] < S + 1)
    tl.store(VOUT + out_off, vc, s[:, None] < S + 1)
    scores = tl.sum(kc * q[None, :], 1) * (H ** -0.5)
    scores = tl.where(s < S + 1, scores, float('-inf'))
    weights = tl.exp(scores - tl.max(scores, 0))
    weights = weights / tl.sum(weights, 0)
    context = tl.sum(vc * weights[:, None], 0)
    z = x + _mv(WO, context, H, H)
    n2w = tl.load(N2 + h)
    n2 = z * tl.rsqrt(tl.sum(z * z, 0) / H + 1.0e-5) * n2w
    gate = _silu(_mv(WG, n2, I, H))
    up = _mv(WU, n2, I, H)
    y = z + _mv(WD, gate * up, H, I)
    tl.store(Y + b * H + h, y)


@triton.jit
def _moe(X, ROUTER, WUP, WDOWN, Y, IDS,
         H: tl.constexpr, E: tl.constexpr, I: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    e = tl.arange(0, E)
    x = tl.load(X + b * H + h).to(tl.float32)
    router = tl.load(ROUTER + e[:, None] * H + h[None, :]).to(tl.float32)
    scores = tl.sum(router * x[None, :], 1)
    first = tl.argmax(scores, 0)
    first_score = tl.max(scores, 0)
    second_scores = tl.where(e == first, float('-inf'), scores)
    second = tl.argmax(second_scores, 0)
    second_score = tl.max(second_scores, 0)
    p0 = 1.0 / (1.0 + tl.exp(second_score - first_score))
    p1 = 1.0 - p0
    tl.store(IDS + b * 2, first)
    tl.store(IDS + b * 2 + 1, second)
    up0 = _mv(WUP + first * I * H, x, I, H)
    down0 = _mv(WDOWN + first * H * I, _silu(up0), H, I)
    up1 = _mv(WUP + second * I * H, x, I, H)
    down1 = _mv(WDOWN + second * H * I, _silu(up1), H, I)
    tl.store(Y + b * H + h, x + p0 * down0 + p1 * down1)


@triton.jit
def _qmv(W, SCALE, x, R: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, R)
    c = tl.arange(0, C)
    q = tl.load(W + r[:, None] * C + c[None, :]).to(tl.float32)
    scale = tl.load(SCALE + r)
    return tl.sum(q * x[None, :], 1) * scale


@triton.jit
def _quant(X, N, WG, WU, WD, SG, SU, SD, Y,
           H: tl.constexpr, I: tl.constexpr):
    b = tl.program_id(0)
    h = tl.arange(0, H)
    x = tl.load(X + b * H + h).to(tl.float32)
    n = x * tl.rsqrt(tl.sum(x * x, 0) / H + 1.0e-5) * tl.load(N + h)
    gate = _qmv(WG, SG, n, I, H)
    up = _qmv(WU, SU, n, I, H)
    down = _qmv(WD, SD, _silu(gate) * up, H, I)
    tl.store(Y + b * H + h, x + down)


@triton.jit
def _spec(D, T, AC, CC, TOK, V: tl.constexpr):
    b = tl.program_id(0)
    v = tl.arange(0, V)
    d0 = tl.argmax(tl.load(D + b * 2 * V + v), 0)
    d1 = tl.argmax(tl.load(D + b * 2 * V + V + v), 0)
    t0 = tl.argmax(tl.load(T + b * 3 * V + v), 0)
    t1 = tl.argmax(tl.load(T + b * 3 * V + V + v), 0)
    t2 = tl.argmax(tl.load(T + b * 3 * V + 2 * V + v), 0)
    a0 = d0 == t0
    a1 = d1 == t1
    count = a0.to(tl.int64) + (a0 & a1).to(tl.int64)
    tl.store(AC + b, count)
    tl.store(CC + b, count + 1)
    tl.store(TOK + b * 3, tl.where(a0, d0, t0))
    tl.store(TOK + b * 3 + 1, tl.where(a0, tl.where(a1, d1, t1), -1))
    tl.store(TOK + b * 3 + 2, tl.where(a0 & a1, t2, -1))


def build(case: dict):
    family = case['family']
    p = case['params']
    if family == 'stencil':
        b, w, t = p['batch'], p['width'], p['steps']

        def run(inputs):
            y = torch.empty_like(inputs['x'])
            _stencil[(b,)](inputs['x'], inputs['alpha'], y, w, t)
            return {'y': y}
    elif family == 'decoder':
        b, h, s, i = p['batch'], p['hidden'], p['context'], p['intermediate']

        def run(inputs):
            y = torch.empty_like(inputs['x'])
            kc = torch.empty((b, s + 1, h), device=inputs['x'].device, dtype=torch.bfloat16)
            vc = torch.empty_like(kc)
            _decoder[(b,)](*(inputs[n] for n in ('x', 'kcache', 'vcache', 'norm1', 'norm2',
                          'wq', 'wk', 'wv', 'wo', 'wg', 'wu', 'wd')),
                          y, kc, vc, h, s, i, triton.next_power_of_2(s + 1))
            return {'y': y, 'kcache': kc, 'vcache': vc}
    elif family == 'moe':
        b, h, e, i = p['batch'], p['hidden'], p['experts'], p['intermediate']

        def run(inputs):
            y = torch.empty_like(inputs['x'])
            ids = torch.empty((b, 2), device=inputs['x'].device, dtype=torch.int32)
            _moe[(b,)](inputs['x'], inputs['router'], inputs['w_up'],
                       inputs['w_down'], y, ids, h, e, i)
            return {'y': y, 'expert_ids': ids}
    elif family == 'quant_mlp':
        b, h, i = p['batch'], p['hidden'], p['intermediate']
        if p['bits'] != 8:
            raise NotImplementedError('Only the smoke 8-bit case is implemented')

        def run(inputs):
            y = torch.empty_like(inputs['x'])
            _quant[(b,)](*(inputs[n] for n in ('x', 'norm', 'w_gate', 'w_up',
                        'w_down', 's_gate', 's_up', 's_down')), y, h, i)
            return {'y': y}
    elif family == 'spec_verify':
        b, k, v = p['batch'], p['draft_len'], p['vocab']
        if k != 2:
            raise NotImplementedError('Only the smoke length-2 case is implemented')

        def run(inputs):
            device = inputs['draft_logits'].device
            ac = torch.empty((b,), device=device, dtype=torch.int64)
            cc = torch.empty_like(ac)
            tok = torch.empty((b, 3), device=device, dtype=torch.int64)
            _spec[(b,)](inputs['draft_logits'], inputs['target_logits'],
                        ac, cc, tok, v)
            return {'accepted_count': ac, 'committed_count': cc,
                    'committed_tokens': tok}
    else:
        raise NotImplementedError(family)
    return run
