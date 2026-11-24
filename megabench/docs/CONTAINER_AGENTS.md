# Claude Code, Codex and opencode in containers

`megabench.integrations.container_run` runs a stock agent CLI (Claude Code,
Codex or opencode, no VibeSys) on one MegaBench case. The agent runs in a rootless Podman
container inside a Slurm GPU job. Its submission is then graded by the trusted
evaluator in a second container in the same job. It is built for
`/goal`-style runs: the agent gets a condition and keeps working until the
condition holds or the time runs out.

## What the agent can and cannot reach

| | Agent container |
|---|---|
| Workspace (`RUN/<case>/`) | read-write, at its host path |
| Trusted snapshot (`RUN/trusted/`: `megabench/` at `--ref`, without past-run docs, submissions, examples, experiments) | read-only |
| `HOME` | `RUN/home/` at `/home/agent`: only the agent's settings and its own session files |
| GPUs | the job's GPUs (Slurm's cgroup) |
| Network | an internal Podman network. The only way out is an HTTP CONNECT proxy on port 443 that admits the model API hosts (`api.anthropic.com` for Claude; `api.openai.com`, `chatgpt.com`, `auth.openai.com` for Codex; the provider's API host, such as `inference.coralbricks.ai`, for opencode) and refuses the rest. Every decision is logged in `RUN/logs/egress-JOB.jsonl` |
| Web search / fetch | off: Claude's `WebSearch` and `WebFetch` are denied in `settings.json`; Codex has `web_search = "disabled"`. These tools run on the provider's side, so the proxy alone would not stop them |
| Repository, host `HOME`, other runs, `/scratch/hf`, `RUN/state`, `RUN/results`, `RUN/grade`, the credential file | not visible (`/scratch/hf` is masked with an empty directory) |

The credential is read on the host at the start of each round. It goes into
the container as an environment variable passed by name, so it never appears
on a command line. The agent can still read it inside its own container,
because the CLI needs it.

Nothing the agent wrote runs on the host. The end-of-round git commit runs in a
container without network, because the agent controls `.git/config` and the
hooks. Grading also runs in a container without network, on a copy of the
workspace with build caches removed. It uses a baseline cache the agent never
saw, so local timings the agent edited cannot reach the grade.

## Slurm shape

The cluster refuses CPU-only jobs, and containers run only inside jobs. So one
round is one GPU job with `case.gpus` GPUs:

1. Start the proxy and the agent container.
2. Run the agent until it exits, its idle screen stops changing (tmux mode), or
   `--agent-minutes` runs out.
3. Commit the workspace in a container, then grade a copy.

The round's `--time` is the agent minutes, plus the grade minutes, plus 5
minutes of setup. The default is 90 + 20 + 5 = 1:55, which fits `main`.
Longer sessions use `--rounds N`: a chain of jobs with `afterany`
dependencies. Each later round resumes the session (`claude --continue`,
`codex resume --last`) and repeats the `/goal`. A round in which the agent
ended on its own writes `RUN/state/finished`, and the rounds after it exit at
once. For many cases or many rounds, queue them in `background`, not `main`
(see the `run-gpu-job` skill).

## Steps

### 1. Build the image (once per CLI or dependency change)

```bash
.venv/bin/python -m megabench.integrations.container_run build-image \
  --context /scratch/$USER/megabench-agent-image
slurm-windows                      # pick partition and --time (the build takes ~20 min)
sbatch ... /scratch/$USER/megabench-agent-image/build.sbatch   # the printed command, or pass --submit
```

The image is `nvidia/cuda:13.1.2-devel-ubuntu24.04` plus the following:

- `/opt/venv`, built from `megabench/docker/requirements.lock` (the repo `.venv`, frozen).
- `ncu`, tmux, git and ripgrep.
- The host's own `claude`, `codex` and `opencode` binaries, at the same versions as the host.
  For opencode, install it on the host first: `npm install -g --prefix ~/.local opencode-ai`.

The tag is a hash of the build context, for example
`localhost/megabench-agent:ab9169a1bf00`. After a CLI update, run
`build-image` again to get a new tag.

### 2. Prepare a run

Use a new directory for each run:

```bash
C=megabench/experiments/$(date +%F)/$(date +%H-%M-%S)-goal-claude
.venv/bin/python -m megabench.integrations.container_run prepare \
  --agent claude --model claude-opus-5-5 \
  --case dense-step-qwen3-06b-b1-s128 --run-dir $C/run \
  --image localhost/megabench-agent:TAG \
  --credential-kind claude-oauth-token --credential-file ~/.config/megabench/claude-token
```

Each agent takes one of these credential kinds:

- **Claude:** `claude-oauth-token` (from `claude setup-token`) or `anthropic-api-key`.
- **Codex:** `openai-api-key`, or `codex-auth-json` (the `auth.json` of a ChatGPT login,
  copied to the run's `HOME`).
- **opencode:** `coralbricks-api-key` (Coral Bricks' OpenAI-compatible API,
  `https://inference.coralbricks.ai/v1`). For example:
  `(umask 077; printenv CORALBRICKS_KEY > ~/.config/megabench/coralbricks-key)`.

The credential file must have mode 600 and must sit outside the run directory.

Options:

- `--mode headless|tmux`
  - **Headless** runs `claude -p` with stream-json input: `/goal <goal>`, then a
    kickoff message. For Codex it runs `codex exec "/goal <goal> <kickoff>"`.
  - **tmux** starts the interactive TUI in tmux and types the same lines into it
    the way a user would.
  - The default is headless for Claude and tmux for Codex. Claude's `/goal` is
    marked as supported in non-interactive mode. Codex's goal feature is
    confirmed only in its TUI; in `codex exec` the `/goal` line may reach the
    model as plain text.
- `--goal "<condition>"`: one line. The default asks for a passing
  `./check_candidate`, then either a time within 1.5x of `./sol_info` or three
  optimizations in a row that each gain less than 2%.
- `--goal-style prompt`: sends the same goal as plain text, without `/goal`.
  Use it as a control run.
- **opencode** runs headless only (`opencode run --format json`). Its config is
  `RUN/home/.config/opencode/opencode.json`: one provider, with `webfetch` and
  `websearch` denied. `--model` defaults to `deepseek-v4.1-flash`; the other
  choices are `deepseek-v4.1-flash-fast` and `deepseek-v4.1-flash-value`.
  `--effort` sets `reasoningEffort`: `low`, `high` (the default) or `max`.
  `none` turns reasoning off, which is the API's own default. opencode has no
  `/goal`, so with `--goal-style command` the harness stands in for it: the goal
  goes into the first prompt, and whenever `opencode run` stops with time left,
  the harness resumes the session (`--continue`) with `control/goal_check.txt`.
  It stops when the agent's last reply says `GOAL MET`, when time runs out, or
  after three runs in a row that each last under a minute. Each re-prompt is a
  `harness_goal_reprompt` line in the round's output, and `report` counts it.
  `--goal-style prompt` sends the goal once, with no re-prompts.
- `--rounds`, `--agent-minutes`, `--grade-minutes`, `--idle-minutes` (tmux),
  `--partition`, `--allow-host`, `--ref`, `--effort`.

### 3. Smoke check (about 5 minutes, no credential used)

```bash
.venv/bin/python -m megabench.integrations.container_run smoke --run-dir $C/run
cat $C/run/logs/smoke-JOB.txt       # ends with SMOKE PASS or SMOKE FAIL
```

The smoke job starts the agent container exactly as a round does. It checks
the following:

- The forbidden paths are absent.
- The trusted snapshot is read-only and the workspace is writable.
- The GPU count matches the job's, and torch runs on it.
- Direct egress is blocked, and the proxy refuses `github.com` and plain HTTP
  but reaches the allowed hosts.
- The CLIs, `nvcc` and `ncu` run.

### 4. Submit

```bash
slurm-windows
.venv/bin/python -m megabench.integrations.container_run submit --run-dir $C/run [--partition background]
```

To watch a tmux-mode round live, read-only:

```bash
srun --jobid=JOB --overlap --pty podman exec -it mb-agent-JOB tmux attach -r -t agent
```

### 5. Results

| Path | What |
|---|---|
| `RUN/results/round-N-JOB.jsonl` | the trusted `megabench evaluate` record of round N (method `container-AGENT-MODEL`, session `...-rN`) |
| `RUN/logs/agent-round-N.out` | headless: the CLI's JSON stream; tmux: the sent lines and the last screen |
| `RUN/home/.claude/projects/`, `RUN/home/.codex/sessions/`, `RUN/home/.local/share/opencode/` (SQLite) | full transcripts |
| `RUN/logs/egress-JOB.jsonl` | every connection the agent attempted through the proxy |
| `RUN/logs/slurm-JOB.out`, `RUN/logs/grade-round-N.err` | job and grading logs |
| `RUN/state/round-N.commit` | the workspace commit that was graded |
| `RUN/config.json` | all settings, the trusted ref and its commit |

`megabench aggregate` takes the result files as in the
[session protocol](SESSION_PROTOCOL.md).
