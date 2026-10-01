# One case per agent session

Each MegaBench case is a separate agent task. Assign its case ID before the
agent starts, using a fresh candidate checkout and a fresh agent conversation.
Give the agent only that case's implementation objective. Shared read-only
benchmark contracts and the same tool/model budget are allowed. Do not carry
candidate source, progress notes, or agent chat history from another case.

An agent may inspect the assigned case's PyTorch oracle and use its own local
checks. The official evaluator uses fresh inputs, compares every output and
input mutation, then audits GPU launches. It writes exactly one case result,
with a method and session ID. A failed case keeps its failure result; it does
not borrow another session's candidate.

For one candidate:

```bash
.venv/bin/python -m megabench evaluate \
  --case CASE_ID --submission /unique/session/checkout/submission.py \
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

For VibeSys, run `python -m megabench.make_vibesys_case_task --case CASE_ID`
inside a fresh checkout. This creates a case-specific objective and a
protected evaluator under `.vibesys/tasks/CASE_ID/`. Run VibeSys with that
single task name. Its accuracy and benchmark commands only call the assigned
case. Generate a new task in a new checkout for the next case. For local
coding sandboxes, expose the shared Python environment and CUDA toolkit as
read-only resources with `VIBESYS_AGENT_SANDBOX_ALLOW`, using the toolkit's
real directory rather than a symlink. Otherwise the trusted evaluator may
have Torch while the coding agent cannot import it.
