"""CPU tests for the Method 1 statistics, analysis, and report (no GPU, no MPK)."""

from __future__ import annotations

import json
import random

from megabench.sota import method1, timing


def _noisy(center: float, n: int, seed: int, spread: float = 0.05) -> list[float]:
    rng = random.Random(seed)
    return [center + rng.uniform(-spread, spread) for _ in range(n)]


def test_diff_ci_contains_known_shift():
    a = _noisy(100.8, 150, 1)
    b = _noisy(100.0, 150, 2)
    d = timing.diff_ci(a, b, seed=3)
    assert abs(d["diff"] - 0.8) < 0.03
    lo, hi = d["ci95"]
    assert lo < 0.8 < hi and hi - lo < 0.1


def test_diff_ci_empty():
    assert timing.diff_ci([], [1.0]) == {"diff": None, "ci95": None}


def test_linear_fit_exact_line():
    xs = [129, 130, 146, 162]
    fit = timing.linear_fit(xs, [2.0 + 0.6 * x for x in xs])
    assert abs(fit["slope"] - 0.6) < 1e-12
    assert abs(fit["intercept"] - 2.0) < 1e-9
    assert all(abs(r) < 1e-9 for r in fit["residuals"])


def test_slope_ci_recovers_step():
    groups = {m: _noisy(10.0 + 0.5 * m, 60, m) for m in (129, 130, 146, 162, 178, 193)}
    s = timing.slope_ci(groups, seed=5)
    assert abs(s["slope"] - 0.5) < 0.01
    assert s["ci95"][0] < 0.5 < s["ci95"][1]
    assert set(s["residuals"]) == set(groups)


def _offline_rows(step_ms: float, base_ms: float, msls=(129, 130), rounds=3, reps=20,
                  tag="main") -> list[dict]:
    rows = []
    for m in msls:
        rows.append({"kind": "mpk_build", "mode": "offline", "msl": m, "mbt": 1,
                     "cutlass": "on", "ok": True, "compile_ms": 30000.0,
                     "generated": m - 128, "expected_generated": m - 128,
                     "final_step": m - 1, "workers": 128, "schedulers": 80})
    for r in range(rounds):
        for m in msls:
            x = _noisy(base_ms + step_ms * (m - 129), reps, 100 * r + m, spread=0.02)
            rows.append({"kind": "mpk_timing", "mode": "offline", "msl": m, "mbt": 1,
                         "cutlass": "on", "round": r, "samples_ms": x,
                         "median_ms": sorted(x)[len(x) // 2], "final_step": m - 1,
                         "generated": m - 128, "tokens_head": [1, 2, 3],
                         "clock_after": {"sm_mhz": 1965, "throttle_mask": 0},
                         "gpu": "NVIDIA B200", "mpk_commit": "6ce3a6b", "tag": tag})
    return rows


def _ours_records() -> list[dict]:
    recs = [{"record_type": "correctness", "arm": "opus", "variant": "direct",
             "grade": "pass", "grading": "oracle_band", "mutated_inputs": [],
             "config": {"bitexact_vs_submission": True}}]
    for mode, med in (("steady", 0.48), ("isolated", 0.50)):
        x = _noisy(med, 30, 7, spread=0.005)
        recs.append({"record_type": "timing", "arm": "opus", "variant": "direct",
                     "mode": mode, "l2": "warm", "samples_ms": x,
                     "stats": timing.summarize(x), "n_steps": 50 if mode == "steady" else 1})
    return recs


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_analyze_and_render(tmp_path):
    (tmp_path / "ours").mkdir()
    _write_jsonl(tmp_path / "ours" / "records.jsonl", _ours_records())
    _write_jsonl(tmp_path / "mpk-main.jsonl",
                 _offline_rows(0.7, 100.0, msls=method1.SLOPE_MSL))
    _write_jsonl(tmp_path / "mpk-notoken.jsonl", [
        {"kind": "mpk_check", "mode": "notoken", "strict_pass": False,
         "outputs": {"logits": {"mismatched": 3, "max_abs_error": 0.2},
                     "next_token": {"mismatched": 0, "got": 5, "want": 5}}},
        {"kind": "mpk_timing", "mode": "notoken", "round": 0,
         "samples_ms": _noisy(1.43, 30, 9, spread=0.01), "median_ms": 1.43}])
    s = method1.analyze(tmp_path)
    main = s["mpk"]["main"]
    assert abs(main["step"]["diff_ms"] - 0.7) < 0.02
    assert main["generated_length_ok"] and main["tokens_deterministic"]
    assert abs(main["slope"]["slope"] - 0.7) < 0.01
    assert s["ours"]["correctness"]["direct"]["grade"] == "pass"
    text = method1.render(s)
    assert "Device step" in text and "Call contract" in text
    assert "## Slope (main)" in text and "strict pass **False**" in text
    assert len(main["slope"]["round_slopes_ms"]) == 3 and main["step"]["ci95"]
    json.dumps(s, default=str)  # the summary must serialize


def test_generated_length_mismatch_is_flagged(tmp_path):
    rows = _offline_rows(0.7, 100.0)
    for r in rows:
        if r["kind"] == "mpk_timing" and r["msl"] == 130:
            r["generated"] = 1  # e.g. an EOS stop
    out = method1.analyze_offline(rows)
    assert out["generated_length_ok"] is False


def test_t_interval():
    t = timing.t_interval([1.0, 2.0, 3.0])
    assert t["mean"] == 2.0 and t["n"] == 3
    assert abs(t["ci95"][1] - (2.0 + 4.303 / 3 ** 0.5)) < 1e-9
    assert timing.t_interval([1.0])["ci95"] is None


def test_round_ci_wider_than_rep_bootstrap():
    # A per-round offset shared by every rep in the round: the pooled bootstrap
    # misses it, the interval over rounds does not.
    rows = _offline_rows(0.7, 100.0, rounds=5, reps=30)
    for r in rows:
        if r["kind"] == "mpk_timing" and r["msl"] == 130:
            shift = 0.2 * (r["round"] - 2)
            r["samples_ms"] = [v + shift for v in r["samples_ms"]]
            r["median_ms"] += shift
    st = method1.analyze_offline(rows)["step"]
    width = st["ci95"][1] - st["ci95"][0]
    rep_width = st["ci95_rep_bootstrap"][1] - st["ci95_rep_bootstrap"][0]
    assert width > 3 * rep_width and st["ci95"][0] < 0.7 < st["ci95"][1]


def test_report_falls_back_to_earlier_run(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "ours").mkdir(parents=True)
    new.mkdir()
    _write_jsonl(old / "ours" / "records.jsonl", _ours_records())
    _write_jsonl(old / "mpk-main.jsonl", _offline_rows(0.7, 100.0, msls=method1.SLOPE_MSL))
    _write_jsonl(new / "mpk-main-nocutlass.jsonl",
                 _offline_rows(0.6, 90.0, msls=method1.SLOPE_MSL, tag="main-nocutlass"))
    s = method1.analyze(new, (old,))
    assert set(s["mpk"]) == {"main", "main-nocutlass"}
    assert s["sources"]["mpk:main"].startswith(str(old))
    text = method1.render(s)
    assert "Parts taken from other runs" in text and "CUTLASS off: slope" in text


def test_paired_round_diffs():
    rows = _offline_rows(0.7, 100.0, rounds=4)
    pairs = method1._paired_rounds([r for r in rows if r["kind"] == "mpk_timing"], 129, 130)
    assert len(pairs) == 4 and all(abs(p - 0.7) < 0.05 for p in pairs)


def test_nsys_summarize_mpk_spans(tmp_path):
    from megabench.sota import nsys_check
    rows = ["Start (ns),Duration (ns),CorrId,Name"]
    for call in range(3):
        t0 = call * 10_000_000
        rows += [f"{t0},5000,1,void kernel::prepare_kernel<1>(RuntimeConfig)",
                 f"{t0 + 8000},1300000,2,void kernel::persistent_kernel(RuntimeConfig)",
                 f"{t0 + 8000},1290000,3,void kernel::scheduler_kernel(RuntimeConfig)",
                 # the next call's step.fill_() runs before its prepare kernel
                 f'{t0 + 9_000_000},1300,4,"void at::native::vectorized_elementwise_kernel'
                 f'<(int)4, at::native::FillFunctor<int>>(int, T2, T3)"']
    rows.append(f"{50_000_000},4000,9,[CUDA memcpy Host-to-Device]")
    path = tmp_path / "mpk_cuda_gpu_trace.csv"
    path.write_text("\n".join(rows) + "\n")
    s = nsys_check.summarize(path)
    assert s["kernels"]["persistent_kernel"]["n"] == 3
    assert s["kernels"]["vectorized_elementwise_kernel"]["n"] == 3
    assert abs(s["kernels"]["persistent_kernel"]["median_us"] - 1300.0) < 1e-9
    assert abs(s["mpk_call_span_us"]["median_us"] - 1308.0) < 1e-9
    assert method1._worker_kernel(s)["median_us"] == 1300.0
