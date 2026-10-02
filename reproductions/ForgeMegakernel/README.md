# ForgeMegakernel agent-pipeline reproduction

The objective here is to reproduce the **agent workflow** in [ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels](https://arxiv.org/html/2609.12379) (Li et al., arXiv:2609.12379v1, September 2026): fresh coding rounds, milestone guidance, a protected measurement gate, a second-agent audit, and a recorded keep/revert decision. The generated kernels are outputs of that workflow, not code this reproduction needs to hand-implement. This is a **partial pipeline reproduction**: the campaign control flow is implemented, while the paper's full checkpoint GPU oracle and deployment gate are not.

For MegaBench P0, `p0_campaign.py` adapts the same edit/gate/review loop to five independent, synthetic-weight cases. It uses one fresh `gpt-6-sol` Codex CLI editor per round and a fresh reviewer after a passing official MegaBench evaluation. Each case has its own minimal workspace and ledger. A MegaBench pass is a development result for that case, **not** a Forge paper milestone pass. The campaign checks CUDA visibility before calling an agent because the paper's loop requires measured GPU feedback after each round.

The `seeds/qwen3_b1s128` directory holds a copied, compile-only M5-shaped Qwen3 CUDA candidate from the isolated MegaBench Codex trial described below. It is a one-CTA source seed, not a passed Forge milestone; its original experiment and reports were not modified.

## Run

From the repository root:

```bash
.venv/bin/python -m unittest reproductions.ForgeMegakernel.test_reproduction reproductions.ForgeMegakernel.test_campaign reproductions.ForgeMegakernel.test_codex_adapter reproductions.ForgeMegakernel.test_qwen3_gate reproductions.ForgeMegakernel.test_p0_campaign -v
.venv/bin/python -m reproductions.ForgeMegakernel.run
.venv/bin/python -m reproductions.ForgeMegakernel.run_checkpoint \
  --snapshot reproductions/.hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --tokens 100 200 --hf
```

The command uses deterministic random bf16 weights for a tiny two-layer Qwen3-style decoder. It runs independent fp64 and fp32 eager references and a batch-vectorized candidate. The positions are ragged. It prints a JSON gate envelope with the configuration hash, errors, poison and lane checks, and an offline schedule audit. The reported Eq. 2 bytes are the **nominal cell** with every lane at the maximum position; they are not the bytes read by this ragged CPU run. The `QWEN3_06B` geometry is included for the paper's byte formula; the full 0.6B checkpoint is **not** loaded or decoded by this CPU command.

To attempt a P0 case, use a new campaign path for that case:

```bash
.venv/bin/python -m reproductions.ForgeMegakernel.p0_campaign \
  --case dense-step-qwen3-06b-b1-s128 \
  --campaign-dir megabench/experiments/forge-p0/dense \
  --rounds 2
```

Run the same command with distinct campaign paths for the other four P0 case IDs in `megabench/cases.py`. A case can be resumed at its existing path. If CUDA is invisible, the command records `preflight.json` with `abstain` and does not start a coding agent. On a GPU host, it saves each agent prompt/report, source diff and snapshot, official MegaBench JSONL result, reviewer verdict, and keep/revert ledger. Passing changes are committed to that case's private workspace Git repository.

The five P0 campaigns are under `megabench/experiments/2026-10-02/06-19-37-forge-pipeline-p0/` (UTC timestamp). Their first preflight recorded `CUDA not visible` because the default command sandbox hides `/dev/nvidia*`. Host-side execution outside that sandbox sees eight B200 GPUs. One `gpt-6-sol` editor round was subsequently attempted per case on the host GPU path. The nested Codex editor sandbox still hides CUDA, so its `./check_candidate` calls reported unavailable; the **independent campaign gate did run on a GPU** after each editor returned.

| P0 case | Trusted gate | Reviewer / campaign decision | Candidate CUDA-event p50 | Provisional speedup vs eager |
| --- | --- | --- | ---: | ---: |
| Qwen3 dense | Correct; one launch | Pass / keep | 347.4 ms | 0.042× |
| Qwen3 MoE | Original worker timed out at 180 s; frozen-source regrade passed correctness and one launch with a 1200 s limit | Not run / revert | 52.1 ms on regrade | 1.24× on regrade |
| Gemma W8 | Correct; one launch | Abstain / revert: compiled helper revision was not proven to reviewer | 36.8 ms | 2.12× |
| Gemma W4 | Correct; one launch | Pass / keep | 30.4 ms | 1.84× |
| EAGLE3 target | Correct; one launch, three acceptance scenarios | Pass / keep | 629.0 ms | 0.165× |

These are synthetic-weight MegaBench P0 results, not paper checkpoint or MBU results. The server had other jobs using several B200s during the campaign, so the latencies are provisional. No five-case campaign score is claimed because the MoE and W8 rounds were not accepted by the gate/review loop. The adapter now passes its worker timeout to MegaBench, records early failures without mislabeling them as digest mismatches, uses a fresh extension cache for each gate, and includes the full source manifest and gate result in reviewer input.

## Paper mapping and status

| Paper component | Local implementation | Evidence and limit |
| --- | --- | --- |
| Eq. 2, byte accounting | `config.py` derives projection, KV, and gathered embedding bytes | Unit test confirms the full embedding table is not counted as a read; no hardware traffic measurement |
| Functional prefix, M0–M4 | `reference.py` and `candidate.py` implement RMSNorm, QK norm, RoPE, GQA, SwiGLU, residuals, LM head, ragged positions. `checkpoint.py` maps a real Qwen3 bf16 checkpoint read-only into these paths | Tiny random-weight CPU case and a three-step continuation pass; a two-tap real-checkpoint CPU diagnostic also runs, but the paper's GPU milestone gates are **not passed** |
| Mid-state oracle, §4.2 | `oracle.py` computes Eq. 4 KV-row error, Eq. 5 top-logprob error, Eq. 6 precision projection, poison and per-lane checks, Eq. 3 MBU helper, and the 16/80-step timing slope. `production_numerical_evidence` evaluates the three production accuracy bars when matching baselines are supplied | Mutation tests verify that corrupt KV writes, out-of-bounds reads, and position broadcasts are rejected; the toy runner still abstains on production bars without checkpoint-matched HF/SGLang baselines |
| Per-SM streams and counters, M6–M7 | `schedule.py` builds typed per-SM queues with explicit producer IDs, recycles slot descriptors, and simulates monotonic completion counters | Offline model meets the paper's static count/imbalance targets; no device interpreter, timeline, or performance measurement, so these milestones are **not passed** |
| One-launch GPU execution, M5; shared-memory pool, M8; long rollout, M9 | Agents generated P0 one-launch CUDA candidates; M8 and M9 are not implemented or claimed | MegaBench GPU checks observed one launch on all five frozen candidates, but synthetic-weight P0 checks cannot establish the paper's checkpoint M5 gate, occupancy, Nsight traffic, MBU, or stability result |
| Agent campaign, §4.3 | `campaign.py` runs a fresh agent process per round, passes the milestone matrix and last measured gate summary, records hypothesis and diff, runs an external gate, and requires a separate reviewer pass before keeping a candidate or advancing. `codex_adapter.py` provides fresh `gpt-6-sol` Codex CLI editor/reviewer processes | Round transitions and fail-closed evidence policy are unit tested; the P0 adaptation also ran one real editor round per case on the host GPU path. `qwen3_gate.py` still abstains from Forge milestones beyond M0 because MegaBench's synthetic fixture is not the paper's checkpoint oracle |

The CPU `status: pass` means only that this **toy functional gate** passed. `mbu: null` and the abstentions in `details` are intentional. The schedule's `paper_M6_static_targets` and `paper_M7_counter_count_target` report static shape properties, not paper milestone completion. Its `shared_slot` is an offline reuse descriptor; it is not CUDA shared memory.

The real-checkpoint command reads the pinned local Qwen3-0.6B safetensors and optionally runs Hugging Face bf16 on the same teacher-forced prefixes. With tokens `100 200`, the local run found candidate KV error C = `1.19e-5` against fp32 and B = `1.61e-5` for fp32 against fp64, with matching greedy token IDs across candidate, fp64, and Hugging Face. The checkpoint SHA-256 was `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. These are CPU diagnostics over two taps, not a long rollout or a GPU gate pass; no SGLang baseline was available.

## Campaign driver contract

`python -m reproductions.ForgeMegakernel.campaign --spec /path/to/spec.json --campaign-dir /path/to/new-campaign --rounds 1` starts or resumes a campaign. The spec pins a seed candidate directory, model dimensions, batch, context, `checkpoint_path` and its SHA-256, peak GPU bandwidth, timing scope, cell-specific M6–M8 MBU floors, rules, and three argv arrays named `agent_command`, `gate_command`, and `review_command`. The driver verifies checkpoint bytes before every invocation and detects edits during each round. The MBU floors are inputs because the paper reports 28%, 35%, and 38% for its Qwen3-0.6B b1×128 ablation cell, not universal floors for all models and GPUs.

The agent receives a JSON prompt on stdin and edits only `FORGE_CANDIDATE_DIR`. It writes `{"hypothesis": "..."}` to `FORGE_AGENT_REPORT`. The independent gate receives the milestone, run ID, nominal bytes, candidate digest, pinned checkpoint path/digest, and timing scope on stdin. It prints the paper-style `{status, suite, summary, metrics, details}` JSON envelope plus the same run ID and candidate digest. The reviewer receives the run ID, complete diff, gate envelope, and host-derived metrics on stdin, and prints a JSON `status` and matching run ID. Each command is a new process. The driver stores all rounds, diff patches, gate/review reports, and a JSONL ledger in a new campaign directory; failed rounds keep their artifacts but the next round starts from the last accepted candidate. An unchanged or missing metric never advances a milestone. The exact required metric keys are in `milestones.py`.

For `gpt-6-sol` coding rounds, set `agent_command` to an argv array ending in `"-m", "reproductions.ForgeMegakernel.codex_adapter", "edit"`, and `review_command` to the same module with `"review"`; use an absolute path to this repository's `.venv/bin/python` as the first element. The adapter invokes [Codex CLI `exec`](https://learn.chatgpt.com/docs/developer-commands?surface=cli) with `--model gpt-6-sol`, a fresh ephemeral session, workspace-write access for edits, and read-only access for review. The P0 campaign exercised these real Codex paths. An agent cannot establish gate status by writing its own report.

Each editor prompt includes the current milestone's implementation guidance alongside the milestone matrix and the last trusted gate summary. It asks the agent to identify the failed correctness or structural check first, or use measured latency, MBU, and schedule evidence to choose one performance change after those checks pass. The M5–M8 guidance names the required launch, per-SM stream, counter, and buffer-pool properties and their diagnostics. The P0 prompt adds explicit advice to examine launch geometry, serial work, redundant reads, and idle SMs when a correct one-launch candidate is slow. The reviewer prompt asks for source-path, compiled-branch, gate-input-special-casing, and metric checks, with abstention when evidence is insufficient. These prompts guide edits; they do not substitute for the missing checkpoint and profiler-backed gate.

For the Qwen3 b1×128 CUDA source seed, set `gate_command` to an argv array ending in `"-m", "reproductions.ForgeMegakernel.qwen3_gate"`. The gate pins the checkpoint digest, builds/imports the candidate at M0 on a CUDA host, and runs MegaBench's independent dense-step check for later milestones. A MegaBench pass yields development feedback but **abstains** from M1 onward until the checkpoint-matched Forge oracle and structural/profiler gates are implemented. It abstains immediately in the default command sandbox, which hides CUDA devices.

The driver hashes its own evaluator package, the spec, optional `protected_paths`, and the accepted candidate. Hash checks detect changes, but they are **not an operating-system sandbox**: run coding agents with an external filesystem restriction that allows writes only to the candidate and agent report. A gate must independently inspect source/launches and collect real checkpoint, CUDA-event, `%globaltimer`, Nsight Compute, and oracle evidence. A JSON envelope fabricated by an agent is not valid evidence. The CPU toy gate cannot serve as that gate.

## Oracle details

- The golden pass upcasts exactly the same bf16 weight bits to fp64. The fp32 reference and candidate upcast those bits to fp32. The relative KV error compares the key and value row written at each layer and lane, and reports both `B = δ(fp32, fp64)` and `C = δ(candidate, fp32)`.
- The candidate must produce finite logits, survive poisoning of cache positions after each lane's write cursor with `100` noise, and give each lane the same result when decoded alone at its own position. The poison comparison is bit exact; the independent-lane comparison uses `1e-5` tolerance because CPU batch and scalar matrix reductions have different rounding orders.
- The top-logprob diagnostic uses the fp64 golden's top 64 tokens. `production_numerical_evidence` computes the paper's `5D`, SGLang mean, and SGLang maximum accuracy bars when supplied with checkpoint-matched teacher-forced baselines. This toy command does not supply those baselines and does not claim those bars pass.
- The paper's precision contract requires operator-specific paired probes. `precision_alpha` implements Eq. 6, with an abstention when exact and narrowed references coincide. The local gate does not yet generate probes for every operator, so precision-contract completion is not claimed.
- `model_bandwidth_utilization` and `slope_us` implement the paper's equations, but a credible MBU also needs a real GPU step time, a declared timing scope, measured HBM traffic, and the correct device bandwidth. The CLI deliberately reports no MBU.

## What a full reproduction still requires

1. Load a canonical bf16 checkpoint, prepare teacher-forced taps and independent HF bf16/float64 and SGLang outputs, and freeze those oracle inputs before an agent edits kernel source.
2. Have the coding agents generate and measure a **single CUDA launch** covering embedding, every decoder layer, the LM head, and token selection. Audit the actual launch count, not just the host call count.
3. Run agent campaigns through the M6–M8 ladder on a GPU: typed per-SM interpreter, fine-grained counter edges without grid-wide barriers, and a true shared-memory buffer pool with measured lifetime safety. Audit source and schedule, `%globaltimer` spans, CUDA-event latency, and Nsight Compute traffic.
4. Connect a trusted GPU evaluator to the campaign driver. Compute Eq. 2 for every measured `(model, batch, context)` cell; use the paper's 16/80-step two-point slope and compare matching timing scopes and checkpoints against SGLang 0.5.18 and MPK. Record GPU model, clocks, profiler counters, and each agent round's hypothesis, diff, gate result, and keep/revert decision.
5. Run the 1000-token drift check and a free 4000-token rollout before claiming M9. Deploy inside SGLang before claiming serving-level reproduction.

The paper used one H100 80GB HBM3 at 3350 GB/s and reported 50.5–85.9% MBU over 14 cells, with a 1.21× geometric mean speedup over SGLang. Those are **paper results**, not results from this directory. This host has eight B200 GPUs; the default command sandbox has no `/dev/nvidia*` nodes and reports CUDA unavailable, while an unrestricted process sees all eight GPUs. The P0 campaign's independent gates ran outside the default sandbox; its nested Codex editors could not run local CUDA checks.

## MegaBench Codex trial (2026-10-01)

A fresh Codex CLI session attempted MegaBench's `dense-step-qwen3-06b-b1-s128` case in the isolated [candidate workspace](../../megabench/experiments/2026-10-01/17-20-30-forge-codex/dense-codex). Its [report](../../megabench/experiments/2026-10-01/17-20-30-forge-codex/dense-codex/IMPLEMENTATION_REPORT.md) and source are kept there under MegaBench's ignored experiment directory. The agent built a full 28-layer, one-CTA CUDA decode candidate with one explicit launch. `nvcc` compilation and PyTorch extension import succeeded. The official `./check_candidate` returned `unavailable: CUDA not visible`; therefore no correctness, launch-audit, latency, or MegaBench score is established. This is an M5-shaped source prototype only: it has no per-SM instruction streams, cross-SM dependency counters, or device shared-memory pool.
