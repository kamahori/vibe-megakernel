#!/bin/bash
# Runs inside the agent container (see container_round.sh).
#   entry.sh round   run the agent for one round (headless or in tmux)
#   entry.sh smoke   check what the container can see and reach
# Exit 124 means the round's time ran out; a later round resumes the session.
set -uo pipefail
ACTION=${1:-round}
ROUND=${ROUND:-1}
cd "$WS"
NET_MODE=${NET_MODE:-proxy}
if [ -n "${EGRESS_SOCKET:-}" ]; then
    # The proxy's socket (outside this container) as 127.0.0.1:3128, this container's only way out.
    python3.12 "$CONTROL/egress_proxy.py" forward --unix "$EGRESS_SOCKET" --port 3128 &
    for _ in $(seq 50); do
        (exec 3<>/dev/tcp/127.0.0.1/3128) 2>/dev/null && break
        sleep 0.1
    done
fi

# The VibeSys plugins and vibesys-cli, installed into $HOME (RUN/home) from GitHub; prints
# what it installed. Each step is skipped when already done, so a later round reuses them.
vibesys_setup() {
    export PATH="$HOME/.local/bin:$PATH" VIBESYS_CLI="$HOME/.local/bin/vibesys-cli"
    if [ ! -d "$HOME/.claude/plugins/marketplaces/vibesys" ]; then
        claude plugin marketplace add "$VIBESYS_PLUGIN_SOURCE" || return 1
    fi
    claude plugin install vibesys@vibesys || return 1
    claude plugin install "${DOMAIN_PLUGIN:-vibesys-kernels-cuda}@vibesys" || return 1
    if [ ! -x "$VIBESYS_CLI" ]; then
        curl -fsSL https://raw.githubusercontent.com/uw-syfi/vibesys/main/cli/install.sh \
            | VIBESYS_CLI_VERSION="$VIBESYS_CLI_VERSION" VIBESYS_CLI_INSTALL_DIR="$HOME/.local/bin" sh \
            || return 1
    fi
    echo "vibesys plugin $(git -C "$HOME/.claude/plugins/marketplaces/vibesys" rev-parse HEAD) ($VIBESYS_PLUGIN_SOURCE); $("$VIBESYS_CLI" --version)"
}

if [ "$ACTION" = smoke ]; then
    fail=0
    ok() { echo "ok    $*"; }
    bad() { echo "FAIL  $*"; fail=1; }
    for path in $FORBID; do
        if [ -e "$path" ]; then bad "visible: $path"; else ok "absent: $path"; fi
    done
    for path in "$WS/TASK.md" "$WS/submission.py" "$TRUSTED/megabench/cases.py" \
                "$TRUSTED/.venv/bin/python" /home/agent; do
        if [ -e "$path" ]; then ok "present: $path"; else bad "missing: $path"; fi
    done
    echo "visible in $(dirname "$RUN"): $(ls -A "$(dirname "$RUN")" | tr '\n' ' ')"
    echo "visible in $RUN: $(ls -A "$RUN" | tr '\n' ' ')"
    if touch "$TRUSTED/.smoke" 2>/dev/null; then rm -f "$TRUSTED/.smoke"; bad "trusted snapshot is writable"
    else ok "trusted snapshot is read-only"; fi
    if touch "$WS/.smoke" 2>/dev/null; then rm -f "$WS/.smoke"; ok "workspace is writable"
    else bad "workspace is not writable"; fi
    if [ -n "$(ls -A /scratch/hf 2>/dev/null)" ]; then bad "/scratch/hf is not empty"
    else ok "/scratch/hf is empty"; fi
    if [ "${PLATFORM:-cuda}" = neuron ]; then
        if neuron-ls >/dev/null 2>&1; then ok "neuron-ls: $(neuron-ls --json-output 2>/dev/null | jq -c '[.[] | {nc_count, memory_size}]')"
        else bad "neuron-ls does not see the Neuron device"; fi
        cores=$(python -c 'import torch, torch_neuronx; print(torch.neuron.device_count())' 2>/dev/null | tail -n 1)
        if [ "$cores" = "$EXPECT_GPUS" ]; then ok "torch-neuronx sees $cores NeuronCore(s) (NEURON_RT_VISIBLE_CORES=$NEURON_RT_VISIBLE_CORES)"
        else bad "torch-neuronx sees ${cores:-no} NeuronCore(s), the run has $EXPECT_GPUS"; fi
        if python -c 'import torch; x = torch.arange(4096., device="neuron"); assert float(x.sum().cpu()) == 4096 * 4095 / 2; print(torch.__version__)' >/dev/null 2>&1
        then ok "torch runs on the NeuronCore"; else bad "torch cannot use the NeuronCore"; fi
    else
    gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$gpus" = "$EXPECT_GPUS" ]; then ok "nvidia-smi sees $gpus GPU(s), as the job"
    else bad "nvidia-smi sees $gpus GPU(s), the job has $EXPECT_GPUS"; fi
    torch_gpus=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
    if [ "$torch_gpus" = "$EXPECT_GPUS" ]; then ok "torch sees $torch_gpus GPU(s)"
    else bad "torch sees ${torch_gpus:-no} GPU(s), the job has $EXPECT_GPUS"; fi
    if python -c 'import torch; x = torch.arange(4096., device="cuda"); assert float(x.sum()) == 4096 * 4095 / 2; print(torch.__version__, torch.cuda.get_device_name())'
    then ok "torch runs on the GPU"; else bad "torch cannot use the GPU"; fi
    fi
    if (cd "$WS" && python -c 'import megabench.cases' 2>/dev/null); then
        bad "megabench imports without PYTHONPATH (should come only from the trusted snapshot)"
    else ok "megabench is not importable from the workspace by default"; fi
    if PYTHONPATH="$TRUSTED" python -m megabench sol --case "$CASE" >/dev/null 2>&1; then
        ok "trusted megabench runs (sol)"; else bad "trusted megabench does not run"; fi
    if [ "$NET_MODE" = open ]; then
        for url in https://github.com https://api.anthropic.com https://pypi.org http://example.com; do
            code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$url")
            if [ "$code" != "000" ]; then ok "open network reaches $url (HTTP $code)"
            else bad "open network cannot reach $url"; fi
        done
    else
        if curl -sS -m 10 --noproxy '*' -o /dev/null https://github.com 2>/dev/null; then
            bad "direct egress reaches github.com"; else ok "direct egress is blocked"; fi
        if curl -fsS -m 15 -o /dev/null https://github.com 2>/dev/null; then
            bad "the proxy lets github.com through"; else ok "the proxy refuses github.com"; fi
        if curl -fsS -m 15 -o /dev/null http://example.com 2>/dev/null; then
            bad "plain HTTP gets out"; else ok "plain HTTP is refused"; fi
        for host in $ALLOW_HOSTS; do
            case "$host" in .*) continue ;; esac
            code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "https://$host/")
            if [ "$code" != "000" ]; then ok "the proxy reaches $host (HTTP $code)"
            else bad "the proxy cannot reach $host"; fi
        done
    fi
    if [ "${PLATFORM:-cuda}" = neuron ]; then
        # The plugin's profiling skills capture NEFF/NTFF pairs with neuron-profile.
        if out=$(neuron-profile --help 2>&1 | head -n 3) && [ -n "$out" ]; then ok "neuron-profile runs"
        else bad "neuron-profile does not run"; fi
        if python -c 'import nki, nki.language, nki.isa; print(nki.__version__)' >/dev/null 2>&1; then ok "nki imports"
        else bad "nki does not import"; fi
    else
        # Nsight Compute must be able to read the GPU's counters, not only start.
        if out=$(cd /tmp && ncu --target-processes all -c 1 --metrics gpu__time_duration.sum \
                python -c 'import torch; torch.ones(1 << 20, device="cuda").sum().item()' 2>&1) \
                && grep -q gpu__time_duration <<< "$out"; then
            ok "ncu profiles a kernel: $(grep -m1 gpu__time_duration <<< "$out" | tr -s ' ')"
        else bad "ncu cannot profile: $(grep -m2 -E 'ERR|==ERROR==|==WARNING==' <<< "$out" | tr '\n' ' ')"; fi
    fi
    if [ -n "${VIBESYS_PLUGIN_SOURCE:-}" ]; then
        # In a scratch HOME, so the run's own HOME starts clean for round 1.
        real_home=$HOME
        HOME=$(mktemp -d)
        cp -a "$real_home/.claude" "$real_home/.claude.json" "$real_home/.config" "$HOME/" 2>/dev/null
        # vibesys_setup runs in a subshell here, so its exports do not reach this shell.
        export PATH="$HOME/.local/bin:$PATH" VIBESYS_CLI="$HOME/.local/bin/vibesys-cli"
        if line=$(vibesys_setup 2>&1 | tail -n 1) && [ -x "$VIBESYS_CLI" ]; then ok "setup: $line"
        else bad "vibesys setup failed: $line"; fi
        skill=$(find "$HOME/.claude/plugins/cache" -path '*vibesys/*/optimize/references/megakernel-divide-and-conquer.md' | head -n 1)
        if [ -n "$skill" ]; then ok "plugin has the megakernel divide-and-conquer mode"
        else bad "the installed plugin has no megakernel divide-and-conquer mode"; fi
        if claude plugin list 2>/dev/null | grep -q "${DOMAIN_PLUGIN:-vibesys-kernels-cuda}"; then ok "plugins listed: $(claude plugin list 2>/dev/null | grep -c '@vibesys') from vibesys"
        else bad "claude plugin list does not show the plugins"; fi
        if out=$("$VIBESYS_CLI" daemon ensure 2>&1); then ok "vibesys daemon: $(head -n 1 <<< "$out")"
        else bad "vibesys daemon does not start: $out"; fi
        if out=$("$VIBESYS_CLI" task card "$CASE" --repo "$WS" 2>&1); then ok "task card: $(tr '\n' ' ' <<< "$out" | cut -c1-200)"
        else bad "vibesys-cli refuses the task: $(tr '\n' ' ' <<< "$out" | cut -c1-400)"; fi
        # The task's own check on a submission that calls the reference: correct, but over the
        # launch budget, so it must run to the end and fail on launches.
        cat > /tmp/reference_submission.py <<PY
import sys
sys.path.insert(0, "$TRUSTED")
from dataclasses import replace
from megabench.cases import select_cases
from megabench.tasks.workloads import reference
def build(case):
    c = replace(select_cases("all", [case["id"]])[0], gpus=1, tp=1, backend="cuda")
    return lambda inputs: reference(c, inputs)
PY
        out=$(cd "$WS" && timeout 1500 python ".vibesys/tasks/$CASE/evaluate.py" accuracy \
              --submission /tmp/reference_submission.py 2>&1 | tail -n 1)
        status=$(jq -r .status <<< "$out" 2>/dev/null)
        if [ -n "$status" ] && [ "$status" != null ] && [ "$status" != ok_provisional ]; then
            ok "task check runs: reference submission -> $status"
        else bad "task check did not run as expected: $(cut -c1-400 <<< "$out")"; fi
        "$VIBESYS_CLI" daemon stop >/dev/null 2>&1
        HOME=$real_home
    fi
    if tmux -V >/dev/null 2>&1; then ok "tmux: $(tmux -V)"; else bad "tmux does not run"; fi
    if [ "${PLATFORM:-cuda}" = neuron ]; then tools="claude git neuron-ls neuronx-cc"
    else tools="claude codex opencode nvcc ncu git"; fi
    for tool in $tools; do
        if version=$("$tool" --version 2>&1 | tail -n 1); then ok "$tool: $version"
        else bad "$tool does not run"; fi
    done
    echo "user: $(id)"
    if [ "$fail" = 0 ]; then echo "SMOKE PASS"; else echo "SMOKE FAIL"; fi
    exit 0
fi

if [ -n "${VIBESYS_PLUGIN_SOURCE:-}" ]; then
    # Before the agent's clock starts; its output goes to stderr, not the agent's JSON stream.
    vibesys_setup >&2 || { echo "vibesys setup failed" >&2; exit 5; }
fi

goal=$(<"$CONTROL/goal.txt")
kickoff=$(<"$CONTROL/kickoff.txt")
resume=$(<"$CONTROL/resume.txt")
if [ "$ROUND" -gt 1 ]; then second=$resume; else second=$kickoff; fi
if [ "$AGENT" = opencode ]; then
    # No /goal in opencode: the goal goes in the prompt; with GOAL_STYLE=command the loop
    # below re-prompts until the agent says GOAL MET.
    first="$second Keep working until this holds: $goal"
    second=""
elif [ "$GOAL_STYLE" = command ]; then
    first="/goal $goal"
else  # the same goal as plain text, for a run without the goal feature
    first="$second Keep working until this holds: $goal"
    second=""
fi

if [ "$AGENT" = codex ] && [ -n "${OPENAI_API_KEY:-}" ] && [ ! -e "$HOME/.codex/auth.json" ]; then
    printenv OPENAI_API_KEY | codex login --with-api-key >/dev/null || exit 3
fi

json_message() {
    jq -cn --arg t "$1" '{type: "user", message: {role: "user", content: $t}}'
}

if [ "$AGENT" = opencode ]; then
    # `opencode run` until the agent stops; then (GOAL_STYLE=command) resume the session with
    # goal_check.txt until its last reply ends with GOAL MET or the time runs out. stdin must
    # be closed: `opencode run` otherwise waits on it. Its model, provider and permissions are
    # in $HOME/.config/opencode/opencode.json.
    export OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_AUTOUPDATE=1 \
        OPENCODE_DISABLE_LSP_DOWNLOAD=1
    deadline=$(( $(date +%s) + AGENT_SECONDS ))
    args=(run --format json)
    [ "$ROUND" -gt 1 ] && args+=(--continue)
    prompt=$first
    last=/tmp/opencode-last.jsonl
    quick=0  # consecutive runs under a minute: the agent is not working; stop re-prompting
    while :; do
        began=$(date +%s)
        left=$(( deadline - began ))
        [ "$left" -le 30 ] && exit 124
        timeout --signal=INT --kill-after=120 "$left" opencode "${args[@]}" "$prompt" \
            < /dev/null | tee "$last"
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 124 ] && exit 124
        [ "$GOAL_STYLE" = command ] && [ "$rc" -eq 0 ] || exit "$rc"
        if jq -r 'select(.type == "text") | .part.text // empty' "$last" 2>/dev/null \
                | tail -n 5 | grep -q 'GOAL MET'; then
            exit 0
        fi
        if [ $(( $(date +%s) - began )) -lt 60 ]; then quick=$((quick + 1)); else quick=0; fi
        [ "$quick" -ge 3 ] && exit 0
        jq -cn --arg t "$(date -Is)" '{type: "harness_goal_reprompt", time: $t}'
        args=(run --format json --continue)
        prompt=$(<"$CONTROL/goal_check.txt")
    done
fi

if [ "$MODE" = headless ]; then
    if [ "$AGENT" = claude ]; then
        args=(-p --dangerously-skip-permissions --verbose
              --input-format stream-json --output-format stream-json)
        [ -n "$MODEL" ] && args+=(--model "$MODEL")
        [ -n "$EFFORT" ] && args+=(--effort "$EFFORT")
        [ "$ROUND" -gt 1 ] && args+=(--continue)
        {
            json_message "$first"
            [ -n "$second" ] && json_message "$second"
        } | timeout --signal=INT --kill-after=120 "$AGENT_SECONDS" claude "${args[@]}"
        exit $?
    fi
    prompt=$first
    [ -n "$second" ] && prompt="$first $second"
    args=(--dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --json)
    [ -n "$MODEL" ] && args+=(-m "$MODEL")
    if [ "$ROUND" -gt 1 ]; then
        timeout --signal=INT --kill-after=120 "$AGENT_SECONDS" \
            codex exec "${args[@]}" resume --last "$prompt"
    else
        timeout --signal=INT --kill-after=120 "$AGENT_SECONDS" codex exec "${args[@]}" "$prompt"
    fi
    exit $?
fi

# tmux: the interactive TUI, typed into like a user. The round ends when the agent exits,
# when the screen has not changed for IDLE_SECONDS (the goal is met or the agent waits for
# input), or when the round's time runs out.
if [ "$AGENT" = claude ]; then
    command=(claude --dangerously-skip-permissions)
    [ -n "$MODEL" ] && command+=(--model "$MODEL")
    [ -n "$EFFORT" ] && command+=(--effort "$EFFORT")
    [ "$ROUND" -gt 1 ] && command+=(--continue)
else
    command=(codex --dangerously-bypass-approvals-and-sandbox)
    [ -n "$MODEL" ] && command+=(-m "$MODEL")
    [ "$ROUND" -gt 1 ] && command=(codex resume --last --dangerously-bypass-approvals-and-sandbox)
fi
start=$(date +%s)
tmux new-session -d -s agent -x 200 -y 50 "$(printf '%q ' "${command[@]}")"
send() {
    tmux send-keys -t agent -l -- "$1"
    sleep 2
    tmux send-keys -t agent Enter
}
sleep 20
send "$first"
if [ -n "$second" ]; then sleep 15; send "$second"; fi
echo "sent at $(date -Is): $first"
[ -n "$second" ] && echo "sent: $second"

deadline=$((start + AGENT_SECONDS))
last=""
changed=$(date +%s)
reason=exited
while tmux has-session -t agent 2>/dev/null; do
    now=$(date +%s)
    if [ "$now" -ge "$deadline" ]; then reason=deadline; break; fi
    screen=$(tmux capture-pane -p -t agent 2>/dev/null | md5sum)
    if [ "$screen" != "$last" ]; then
        last=$screen
        changed=$now
    elif [ $((now - changed)) -ge "$IDLE_SECONDS" ]; then
        reason=idle
        break
    fi
    sleep 30
done
echo "round ends at $(date -Is): $reason"
echo "--- last screen"
tmux capture-pane -p -S -200 -t agent 2>/dev/null
tmux send-keys -t agent C-c 2>/dev/null
sleep 3
tmux kill-server 2>/dev/null
[ "$reason" = deadline ] && exit 124
exit 0
