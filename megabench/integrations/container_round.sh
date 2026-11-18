#!/bin/bash
# One round of a container_run run, or its smoke check, inside a Slurm job.
#   round.sh RUN_DIR round N
#   round.sh RUN_DIR smoke
#   round.sh RUN_DIR rescore      re-check the best passing snapshots with the trusted evaluator
# container_run.py copies this file to RUN_DIR/round.sh and submits it with sbatch.
# Nothing that the agent wrote runs here on the host: the agent, the snapshot commit
# and the grading each run in a container; this script only copies files.
set -euo pipefail
RUN=$1
ACTION=$2
NUMBER=${3:-0}
set -a
# shellcheck source=/dev/null
source "$RUN/run.env"
set +a
JOB=${SLURM_JOB_ID:?run this inside a Slurm job (container_run submit / smoke)}
AGENT_CT=mb-agent-$JOB
LOGS=$RUN/logs
EMPTY=$RUN/.empty
mkdir -p "$EMPTY" "$RUN/state/baseline-cache"
# /usr/bin before /usr/local/bin: the host's /usr/local/bin/nvidia-smi (mounted into every
# container) is a wrapper that needs sudo outside a job's cgroup; the real binary is mounted below.
CONTAINER_PATH=/opt/venv/bin:/usr/local/cuda/bin:/opt/codex/codex-path:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin
# The proxy's socket directory: short (Unix socket paths are limited to 108 bytes), private.
NETDIR=$(mktemp -d "/tmp/mb-net-$JOB.XXXXXX")
chmod 700 "$NETDIR"
PROXY_PID=
HELPER_PIDS=()
HARVEST="/usr/bin/python3 $RUN/control/harvest.py"

cleanup() {
    podman rm -f -t 10 "$AGENT_CT" "mb-grade-$JOB" "mb-commit-$JOB" "mb-unpack-$JOB" >/dev/null 2>&1 || true
    [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null
    for pid in "${HELPER_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    rm -rf "$NETDIR"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

echo "job $JOB on $(hostname): $ACTION $NUMBER, image $IMAGE, GPUs ${CUDA_VISIBLE_DEVICES:-?}"
/usr/bin/nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader || true

# The trusted evaluator in a container without network, on DIR; writes OUT (jsonl).
grade_dir() {  # grade_dir DIR OUT SESSION REPS
    local regimes=()
    [ -n "${REGIMES:-}" ] && regimes=(-v "$RUN/state/regimes.json:/regimes.json:ro")
    podman run --rm --name "mb-grade-$JOB" --network none --userns=keep-id --entrypoint= --shm-size "$SHM_SIZE" \
        --security-opt no-new-privileges \
        -v "$TRUSTED:$TRUSTED:ro" -v "$1:$1" -v "$EMPTY:/scratch/hf:ro" \
        -v "$RUN/state/baseline-cache:/baseline-cache" "${regimes[@]}" \
        -e HOME=/tmp -e PATH="$CONTAINER_PATH" -e PYTHONPATH="$TRUSTED" \
        -e MEGABENCH_BASELINE_CACHE=/baseline-cache -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -w "$1" "$IMAGE" \
        bash -c 'python -m megabench evaluate --case "$1" --submission "$2/submission.py" \
            --method "$3" --session-id "$4" --reps "$5" --timeout "$6" --output /tmp/result.jsonl \
            $([ -e /regimes.json ] && echo --regimes /regimes.json) >&2;
            status=$?; cat /tmp/result.jsonl 2>/dev/null; exit $status' \
        grade "$CASE" "$1" "$METHOD" "$3" "$4" "$GRADE_TIMEOUT" > "$2"
}

if [ "$ACTION" = rescore ]; then
    # The best passing snapshots by the agent's own p50; each is unpacked in a container
    # without network and graded RESCORE_REPS times with fresh seeds; the first that passes
    # every check and the static review is the run's result (see `container_run report`).
    mkdir -p "$RUN/rescore"
    n=0
    for tarball in $($HARVEST candidates "$RUN"); do
        n=$((n + 1))
        name=$(basename "$tarball" .tgz)
        dir=$RUN/rescore/$name-$JOB
        mkdir -p "$dir/src"
        podman run --rm --name "mb-unpack-$JOB" --network none --userns=keep-id --entrypoint= \
            -v "$tarball:/snapshot.tgz:ro" -v "$dir/src:/out" "$IMAGE" \
            tar --no-same-owner -xzf /snapshot.tgz -C /out
        if ! $HARVEST review "$RUN" "$dir/src" > "$dir/review.json"; then
            echo "$name: static review failed: $(cat "$dir/review.json")"
            continue
        fi
        for k in $(seq "${RESCORE_REPS:-3}"); do
            set +e
            grade_dir "$dir/src" "$dir/check-$k.jsonl" "$SESSION_ID-rescore-$name-$k" "$GRADE_REPS" \
                2>> "$dir/grade.err"
            echo "$name check $k: exit $?"
            set -e
            rm -rf "$dir/src/.torch_extensions" "$dir/src/.cache"
        done
    done
    echo "rescored $n snapshot(s)"
    exit 0
fi

# The agent's only way out: containers get --network none; the allowlist proxy runs here in
# the job (host network and DNS) on a Unix socket that the agent container sees, and the
# container's entry script exposes it as 127.0.0.1:3128. No Podman network is used: rootless
# Podman's uplink is one process per user shared by all jobs, and it dies with the job that
# started it, cutting every other job off.
# shellcheck disable=SC2086
/usr/bin/python3 "$RUN/control/egress_proxy.py" serve --unix "$NETDIR/egress.sock" \
    --allow $ALLOW_HOSTS --log "$LOGS/egress-$JOB.jsonl" &
PROXY_PID=$!
for _ in $(seq 50); do
    [ -S "$NETDIR/egress.sock" ] && break
    sleep 0.2
done
[ -S "$NETDIR/egress.sock" ] || { echo "the egress proxy did not start" >&2; exit 1; }
proxy_url=http://127.0.0.1:3128

# Every agent container: the workspace (rw), the trusted snapshot (ro), its HOME and the
# control files (ro) at fixed paths; the shared HF cache is masked by an empty directory.
common=(--network none --userns=keep-id --entrypoint= --hostname megabench --shm-size "$SHM_SIZE"
    --security-opt no-new-privileges
    -v "$WS:$WS" -v "$TRUSTED:$TRUSTED:ro" -v "$RUN/home:/home/agent"
    -v "$RUN/control:$RUN/control:ro" -v "$EMPTY:/scratch/hf:ro"
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro -v "$NETDIR:/run/mb-net"
    -e HOME=/home/agent -e PATH="$CONTAINER_PATH" -e TMPDIR=/tmp -e TERM=xterm-256color
    -e HTTPS_PROXY="$proxy_url" -e https_proxy="$proxy_url"
    -e HTTP_PROXY="$proxy_url" -e http_proxy="$proxy_url"
    -e NO_PROXY=localhost,127.0.0.1 -e no_proxy=localhost,127.0.0.1
    -e HF_HUB_CACHE=/home/agent/.cache/huggingface/hub -e HF_XET_CACHE=/home/agent/.cache/huggingface/xet
    -e HF_DATASETS_CACHE=/home/agent/.cache/huggingface/datasets -e HF_HUB_OFFLINE=1
    -e DISABLE_AUTOUPDATER=1 -e IS_SANDBOX=1 -e CONTROL="$RUN/control"
    -e EGRESS_SOCKET=/run/mb-net/egress.sock
    -e WS -e TRUSTED -e CASE -e AGENT -e MODEL -e EFFORT -e MODE -e GOAL_STYLE
    -e AGENT_SECONDS -e IDLE_SECONDS -e ALLOW_HOSTS -e FORBID -e RUN
    -w "$WS")

if [ "$ACTION" = smoke ]; then
    out=$LOGS/smoke-$JOB.txt
    podman run --rm --name "$AGENT_CT" "${common[@]}" -e EXPECT_GPUS="$(/usr/bin/nvidia-smi -L | wc -l)" \
        "$IMAGE" "$RUN/control/entry.sh" smoke > "$out" 2>&1 || true
    echo "--- egress log" >> "$out"
    cat "$LOGS/egress-$JOB.jsonl" >> "$out" 2>/dev/null || true
    cat "$out"
    tail -n 50 "$out" | grep -q '^SMOKE PASS' || exit 1
    exit 0
fi

[ "$ACTION" = round ] || { echo "unknown action $ACTION" >&2; exit 2; }
if [ -e "$RUN/state/finished" ]; then
    echo "the agent ended its session in an earlier round ($(cat "$RUN/state/finished")); nothing to do"
    exit 0
fi
seconds=$($HARVEST budget "$RUN" "$NUMBER")
if [ "$seconds" -le 0 ]; then
    echo "budget spent before round $NUMBER" > "$RUN/state/finished"
    echo "the agent budget is spent; nothing to do"
    exit 0
fi
AGENT_SECONDS=$seconds
touch "$RUN/state/round-$NUMBER.started"
# Host-side helpers: keep the sources of every passing check, sample the GPUs once a second.
( while sleep 5; do $HARVEST harvest "$RUN" >/dev/null 2>> "$LOGS/harvest.err"; done ) &
HELPER_PIDS+=($!)
/usr/bin/nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader \
    -l 1 > "$LOGS/gpu-round-$NUMBER-$JOB.csv" 2>/dev/null &
HELPER_PIDS+=($!)

# The credential: read now, passed by name (never on a command line), dropped after the agent.
credential=()
if [ -n "$CREDENTIAL_ENV" ]; then
    export "$CREDENTIAL_ENV"="$(<"$CREDENTIAL_FILE")"
    credential=(-e "$CREDENTIAL_ENV")
elif [ "$CREDENTIAL_KIND" = codex-auth-json ] && [ ! -e "$RUN/home/.codex/auth.json" ]; then
    install -m 600 "$CREDENTIAL_FILE" "$RUN/home/.codex/auth.json"
fi

start=$(date +%s)
set +e
# Each output line's arrival time goes to .times (line i of .out <-> line i of .times).
podman run --rm --name "$AGENT_CT" "${common[@]}" "${credential[@]}" -e ROUND="$NUMBER" \
    "$IMAGE" "$RUN/control/entry.sh" round 2> "$LOGS/agent-round-$NUMBER.err" \
    | /usr/bin/python3 -u -c 'import sys, time
out, times = open(sys.argv[1], "a"), open(sys.argv[2], "a")
for line in sys.stdin:
    out.write(line); out.flush(); times.write(f"{time.time():.3f}\n"); times.flush()' \
    "$LOGS/agent-round-$NUMBER.out" "$LOGS/agent-round-$NUMBER.times"
rc=${PIPESTATUS[0]}
set -e
[ -n "$CREDENTIAL_ENV" ] && unset "$CREDENTIAL_ENV"
elapsed=$(( $(date +%s) - start ))
echo "agent exit $rc after ${elapsed}s"
echo "$elapsed $rc" > "$RUN/state/round-$NUMBER.elapsed"
$HARVEST harvest "$RUN" >/dev/null 2>> "$LOGS/harvest.err" || true
# 124: the round's time ran out, so a later round resumes. Anything else ends the session.
if [ "$rc" -ne 124 ]; then
    echo "round $NUMBER, exit $rc after ${elapsed}s" > "$RUN/state/finished"
fi

# Snapshot the workspace in git (in a container: hooks and config are the agent's).
podman run --rm --name "mb-commit-$JOB" --network none --userns=keep-id --entrypoint= -v "$WS:$WS" -w "$WS" \
    -e HOME=/tmp "$IMAGE" bash -c 'git add -A && git -c core.hooksPath=/dev/null \
    -c user.name=MegaBench -c user.email=megabench@example.invalid commit -q --no-verify \
    -m "container_run: end of round $1" >/dev/null; git rev-parse HEAD' snapshot "$NUMBER" \
    > "$RUN/state/round-$NUMBER.commit" 2>> "$LOGS/grade-round-$NUMBER.err" || true

# Grade a copy: the trusted evaluator, no network, a baseline cache the agent never saw.
grade=$RUN/grade/round-$NUMBER-$JOB
mkdir -p "$grade"
cp -a "$WS/." "$grade/"
rm -rf "$grade/.git" "$grade/.torch_extensions" "$grade/.cache" "$grade/reports"
result=$RUN/results/round-$NUMBER-$JOB.jsonl
set -o noclobber
set +e
grade_dir "$grade" "$result" "$SESSION_ID-r$NUMBER" "$GRADE_REPS" 2>> "$LOGS/grade-round-$NUMBER.err"
grc=$?
set -e
echo "grading exit $grc; result $result"
head -c 2000 "$result" || true
echo
