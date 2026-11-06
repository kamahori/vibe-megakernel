# One case per agent session

Each MegaBench case is a separate agent task. Assign its case ID before the
agent starts, using a fresh minimal workspace and a fresh agent conversation.
Give the agent only that case's implementation objective. Shared read-only
benchmark contracts and the same tool/model budget are allowed. Do not carry
candidate source, progress notes, or agent chat history from another case.

Create a workspace with
`python -m megabench.make_agent_workspace --case CASE_ID --agent METHOD --output NEW_PATH`,
where `METHOD` is `plain`, `vibesys`, or `kernelagent`. Give the agent its
`TASK.md` and start it in `NEW_PATH`. The workspace contains only the selected
case metadata, its PyTorch oracle and necessary imports, a submission adapter,
and local check and profiler commands. VibeSys also gets its task TOML and
evaluator entry point; KernelAgent gets `problem.txt` and `test.py`. These
helpers call the trusted benchmark in the main repository. No whole-repository
copy is needed. Use a new timestamped path for each run; the generator refuses
to overwrite an existing workspace.

An agent may inspect the assigned case's PyTorch oracle and use its own local
checks. The official evaluator uses fresh inputs, compares every output and
input mutation, then audits GPU launches. It writes exactly one case result,
with a method and session ID. A failed case keeps its failure result; it does
not borrow another session's candidate.

For one candidate:

```bash
.venv/bin/python -m megabench evaluate \
  --case CASE_ID --submission /unique/session/workspace/submission.py \
  --method METHOD --session-id UNIQUE_ID --device cuda:0 \
  --output /unique/session/result.jsonl
```

After separate sessions finish, `aggregate` accepts one result JSONL per
ready case for one method. It rejects duplicate cases or session IDs, shared
candidate directories, and mismatched method labels. A suite speedup exists
only if all cases pass. The score is the geometric mean of each case's
provisional speedup against its local reference baseline. Every result also
records a digest of the benchmark contract; aggregation rejects results from
another contract revision.

```bash
.venv/bin/python -m megabench aggregate --suite p0 --method METHOD \
  --input /session-1/result.jsonl --input /session-2/result.jsonl \
  --input /session-3/result.jsonl --input /session-4/result.jsonl \
  --input /session-5/result.jsonl --output /path/to/summary.json
```

Plain Codex and VibeSys should each receive five independent P0 agent
sessions, one per case, with equivalent session limits. MPK is evaluated as a
per-case implementation too; a native launcher probe alone is not a case
submission. Keep all candidate code and raw logs outside tracked source.

## Optional Nsight Compute feedback

For an agent method without a built-in profiler, the [syfi ncu MCP server](https://github.com/kamahori/mlsys-contest-syfi-fully-agent/tree/main/full-agent-pipeline/mcps/ncu-mcp)
can provide performance counters. Install it outside the candidate workspace
and give each session its own report directory. Its `profile` tool accepts one
executable path; give it the workspace's `profile_candidate` script. Run
`check_candidate` from the workspace for local correctness and launch feedback.
Set `NCU_PATH` to the main benchmark repository's
`megabench/integrations/ncu_capture.sh` so only the CUDA profiler
start/stop window is captured. Pass `kernel_filter="regex:<model kernel name>"`,
`set="basic"`, `launch_count=1`, and the assigned physical GPU to `profile`,
then use `read_report_details` for feedback. Obtain the model kernel name from
the candidate's source or the evaluator's launch audit. NCU may omit C++
signature text from that name: if `regex:kernel\(P\)` captures nothing and NCU
reports `kernel`, retry once with `regex:kernel` while retaining the profiler
markers and one-launch limit. Without a filter, setup kernels may be captured
instead. NCU measurements are diagnostic and
do not replace MegaBench correctness, launch audit, or latency scoring.

For VibeSys, use `--agent vibesys` when creating the workspace. This creates a
case-specific objective and evaluator entry point under `.vibesys/tasks/CASE_ID/`.
Run VibeSys with that single task name. Its accuracy and benchmark commands
only call the assigned case, using the main repository's Python environment.
Create a new workspace for the next case. For local coding sandboxes, expose
the main benchmark repository, shared Python environment, and CUDA toolkit as
read-only resources with `VIBESYS_AGENT_SANDBOX_ALLOW`, using the toolkit's
real directory rather than a symlink. Otherwise the workspace check and
profile commands cannot import the trusted harness.

VibeSys has a built-in NCU report MCP server; it does not need the external
server or its `ncu_capture.sh` wrapper. The task must declare
`domain = "kernel-writing"` and the run must use `--profiler ncu`. Set `NCU_PATH`
to the real CUDA toolkit's `ncu` executable and add the real toolkit directory
to `PATH` and `VIBESYS_AGENT_SANDBOX_ALLOW`. The agent profiles the warmed
`./profile_candidate` entry point with `--profile-from-start off`, then reads
the saved report through the built-in MCP tools.

When the candidate workspace is inside the main benchmark's `megabench/`
directory, a read-only sandbox grant for that ancestor can be masked by the
workspace mount. Create a small trusted benchmark snapshot in a sibling
directory, omitting experiments, runs, archives, and caches, and pass it as
`--benchmark-root` when creating workspaces. Link the snapshot's `.venv` to
the shared environment. Expose the snapshot and shared environment through
`VIBESYS_AGENT_SANDBOX_ALLOW`. Prefix each NCU capture command with
`TMPDIR=/tmp TMP=/tmp TEMP=/tmp`; the agent sandbox may reset inherited temp
variables to `/raid/tmp`, which is absent inside it. Check
`./check_candidate --help` from a fresh workspace before starting the agent.

## Optional per-block phase timeline

NCU averages counters over the whole grid, so it cannot say which phase of a
persistent or grid-synchronized kernel, or which block, sets the critical path.
Every workspace also has `./timeline_candidate`, a `megakernel_skills/` copy of `megabench/skills/`, and a copy of
`megabench/integrations/timeline_probe.cuh` for that question. The candidate
opts in by exporting `build_timeline(case)` next to `build(case)`, returning
`(run, event_names)`. `run(inputs)` launches a build with the header's probes
compiled in (`-DMEGABENCH_TIMELINE`) and returns that call's zeroed-then-filled
`uint64` buffer. Without the define, the probes compile to nothing, so the
scored build is unchanged. FlashInfer's `include/flashinfer/profiler.cuh`
writes the same buffer layout and may be used instead.

The command warms the build up once, records the second call, and writes
`timeline.u64`, `trace.json` (open it in `ui.perfetto.dev`, one track per
block and group) and `summary.json` under `reports/timeline-*/`. It also prints
one row per phase, with times summed over occurrences:

- `wall`: earliest begin to latest end;
- `slowest` and `mean`: per-block duration;
- `begin_skew`: spread of block start times;
- `imbalance`: slowest minus mean;
- the block that most often finished last.

`python -m megabench.integrations.timeline PATH --events a,b --event a`
re-reads a saved buffer or trace without a GPU and prints one phase's
per-occurrence skew. Probes cost a timer read and a store each and can shift
where cost lands. Timestamps are `%globaltimer` nanoseconds, too coarse for
phases under about 0.1 µs, and one capture must span less than about 2.1 s.
Timelines locate a bottleneck. Correctness and latency still come only from
`./check_candidate` and the final grading.

## AMD ROCm sessions

On an AMD GPU, create the workspace with `--backend rocm` (it is also chosen
automatically when `/opt/rocm` exists and `nvidia-smi` does not; `--platform` is an
alias). The workspace then targets the ROCm build of PyTorch, where `torch.cuda` and
`cuda:0` address the HIP device. Skills and skill scripts are filtered by backend
(`megabench/skills/.vibesys.toml` and `backend_files.toml`), so a ROCm workspace gets
no Blackwell tensor-core skill and no cuobjdump gate. Its tools change as follows:

- `./profile_candidate` runs `rocprofv3` instead of Nsight Compute. It always
  records a kernel dispatch trace with stats (grid, workgroup, VGPR/SGPR/LDS,
  scratch, duration). `--kernel REGEX --pmc COUNTER ...` collects hardware
  counters for the warmed call only, and `--att` requests a thread trace.
  `rocprofv3-avail` lists the counters. A counter set that does not fit in one
  hardware pass fails; on gfx942 that includes `FETCH_SIZE WRITE_SIZE
  GRBM_GUI_ACTIVE` together.
- `./trace_report` combines the phase timeline, rocprofv3 counters and
  PC sampling (host_trap) into `reports/trace-*/report.md`. Counters are
  discovered on the device, and a set that does not fit is retried one counter
  per pass. PC samples are kept for the candidate's dispatches only and grouped
  by instruction class: sleep or barrier, `s_waitcnt`, VMEM, LDS, VALU, SALU.
  The report ranks findings such as achieved HBM bandwidth, L2 hit rate, the
  share of samples spent waiting, and the most imbalanced phase. rocprofv3 can
  hang in its own error handler after a fatal error while holding GPU memory,
  so each pass runs in its own process group and is killed on a fatal log line
  or a timeout.
- `timeline_probe.cuh` compiles under HIP. It stamps raw `wall_clock64()`
  ticks, and `./timeline_candidate` scales them by
  `hipDeviceAttributeWallClockRate` (25 MHz on MI210) and writes the scale to
  `clock.json`. `python -m megabench.integrations.timeline` reads that file
  when it re-reads a saved buffer; `--tick-ns` overrides it. The rate query uses
  the ROCm 7.x enum value and rejects an implausible answer; set
  `MEGABENCH_TICK_NS` (nanoseconds per tick) where it fails.
- `megakernel_skills/megakernel-phase-profiling/scripts/codegen_gate_amd.py`
  replaces the cuobjdump-based gate. It extracts code objects with
  `clang-offload-bundler`, compares ISA from `llvm-objdump`, and reads VGPR,
  AGPR, SGPR, LDS, scratch and spill counts from `llvm-readelf --notes`.
- `--gpu {mi300x,mi210,mi350x}` (default `mi300x`) sets the hardware section of
  TASK.md, the offload architecture (gfx942, gfx90a, gfx950) and the bandwidth
  `./sol_info` assumes (5.3, 1.6 and 8 TB/s).
- The PipeLink-style `megakernel-stage-linking` skill is optional
  (`--optional-skill megakernel-stage-linking`). It did not improve performance on
  MI300X: a separately compiled and linked seam cost 1.6-2.4%, and stages swapped
  between independently written kernels were slower or neutral.

The ATT thread-trace decoder is a separate ROCm package; where it is not
installed, `--att` produces no instruction timeline and PC sampling is the
instruction-level signal. A persistent kernel that deadlocks keeps its process
on the GPU, and every later launch queues behind it: wrap grid-size and
synchronization experiments in `timeout`, and kill any leftover process before
the next check.
