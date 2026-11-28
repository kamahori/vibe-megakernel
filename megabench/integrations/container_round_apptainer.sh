#!/bin/bash
# One round of a container_run run, or its smoke check, inside a Slurm job: the Apptainer and
# AMD ROCm counterpart of container_round.sh, for clusters without Docker or rootless Podman.
#   round.sh RUN_DIR round N
#   round.sh RUN_DIR smoke
#   round.sh RUN_DIR rescore      re-check the best passing snapshots with the trusted evaluator
# container_run.py copies this file to RUN_DIR/round.sh and submits it with sbatch.
# Nothing that the agent wrote runs here on the host: the agent, the snapshot commit and the
# grading each run in a container; this script only copies files.
#
# Each container: `apptainer exec --containall --no-mount hostfs,cwd`, so no host directory is
# visible but the ones bound here; a fresh /tmp on node-local disk; the GPUs of the (exclusive)
# node through --rocm. The agent shares the node's network (NET_MODE=open); grading, commits
# and exports run with `--net --network none`.
set -euo pipefail
RUN=$1
ACTION=$2
NUMBER=${3:-0}
set -a
# shellcheck source=/dev/null
source "$RUN/run.env"
set +a
JOB=${SLURM_JOB_ID:?run this inside a Slurm job (container_run submit / smoke)}
LOGS=$RUN/logs
EMPTY=$RUN/.empty
mkdir -p "$EMPTY" "$RUN/state/baseline-cache"
CONTAINER_PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Node-local scratch for each container's /tmp (the job's TMPDIR, erased with the job).
SCRATCH=${TMPDIR:-/tmp}/mb-$JOB
mkdir -p "$SCRATCH"
chmod 700 "$SCRATCH"
HELPER_PIDS=()
HARVEST="/usr/bin/python3 $RUN/control/harvest.py"
GPU_FLAGS=(${APPTAINER_GPU_FLAGS:---rocm})
CRED_COPY=$RUN/home/.claude/.credentials.json

cleanup() {
    for pid in "${HELPER_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    rm -f "$CRED_COPY"
    rm -rf "$SCRATCH"
    # A campaign (container_campaign.py) submits its next job when this one ends, however.
    if [ "$ACTION" = round ] && [ -n "${CAMPAIGN:-}" ] && [ -e "$CAMPAIGN" ]; then
        python3.12 "$CAMPAIGN_DRIVER" next "$CAMPAIGN" --from-job "$JOB" \
            >> "${CAMPAIGN%.*}.log" 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# ct NAME [apptainer options...] -- COMMAND...: one container with a fresh /tmp of its own.
ct() {
    local name=$1; shift
    local opts=() home=/tmp
    # Apptainer ignores HOME from --env/--env-file (it sets HOME to the host user's home path,
    # which is not mounted here), so `--env HOME=X` becomes `--home X`: the path only, since
    # --no-mount home keeps the host home out and any directory there comes from a --bind.
    while [ "$1" != -- ]; do
        if [ "$1" = --env ] && [[ "$2" == HOME=* ]]; then home=${2#HOME=}; shift 2; continue; fi
        opts+=("$1"); shift
    done
    shift
    local tmp=$SCRATCH/$name
    rm -rf "$tmp"; mkdir -p "$tmp"
    apptainer exec --containall --cleanenv --no-mount hostfs,cwd,home --home "$home" \
        --bind "$tmp:/tmp" "${opts[@]}" "$IMAGE" "$@"
}

gpu_count() {  # the node's AMD GPUs: KFD topology nodes with a GPU id
    local n=0 f
    for f in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do
        [ -r "$f" ] && [ "$(cat "$f")" != 0 ] && n=$((n + 1))
    done
    echo "$n"
}
SMI=$(command -v amd-smi || ls /opt/rocm*/bin/amd-smi 2>/dev/null | head -n 1 || true)

echo "job $JOB on $(hostname): $ACTION $NUMBER, image $IMAGE, $(gpu_count) AMD GPU(s)"
echo "container /tmp on $SCRATCH: $(df -h --output=avail "$SCRATCH" | tail -n 1 | tr -d " ") free"
[ -n "$SMI" ] && "$SMI" list 2>/dev/null | grep -E 'GPU|BDF|UUID' || true

# The trusted evaluator in a container without network, on DIR; writes OUT (jsonl).
grade_dir() {  # grade_dir DIR OUT SESSION REPS
    local regimes=()
    [ -n "${REGIMES:-}" ] && regimes=(--bind "$RUN/state/regimes.json:/regimes.json:ro")
    ct grade --net --network none "${GPU_FLAGS[@]}" \
        --bind "$TRUSTED:$TRUSTED:ro" --bind "$1:$1" \
        --bind "$RUN/state/baseline-cache:/baseline-cache" "${regimes[@]}" \
        --env HOME=/tmp --env PATH="$CONTAINER_PATH" --env PYTHONPATH="$TRUSTED" \
        --env MEGABENCH_BASELINE_CACHE=/baseline-cache \
        --env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
        --pwd "$1" -- \
        bash -c 'python -m megabench evaluate --case "$1" --submission "$2/submission.py" \
            --method "$3" --session-id "$4" --reps "$5" --timeout "$6" --output /tmp/result.jsonl \
            $([ -e /regimes.json ] && echo --regimes /regimes.json) >&2;
            status=$?; cat /tmp/result.jsonl 2>/dev/null; exit $status' \
        grade "$CASE" "$1" "$METHOD" "$3" "$4" "$GRADE_TIMEOUT" > "$2"
}

if [ "$ACTION" = rescore ]; then
    mkdir -p "$RUN/rescore"
    n=0
    for tarball in $($HARVEST candidates "$RUN"); do
        n=$((n + 1))
        name=$(basename "$tarball" .tgz)
        dir=$RUN/rescore/$name-$JOB
        mkdir -p "$dir/src"
        ct unpack --net --network none --bind "$tarball:/snapshot.tgz:ro" --bind "$dir/src:/out" -- \
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

NET_MODE=${NET_MODE:-open}
[ "$NET_MODE" = open ] || { echo "the apptainer runtime supports --network open only" >&2; exit 2; }
WORK=${WORK:-$WS}

# Every agent container: the workspace (rw; WORK is its parent for a bare VibeSys workspace,
# whose worktrees sit beside it), the trusted snapshot (ro), its HOME and the control files
# (ro) at fixed paths. Environment: only what is named here (--cleanenv).
agent_env() {  # the agent container's environment, as an --env-file
    local file=$SCRATCH/agent.env
    {
        printf '%s=%q\n' HOME /home/agent PATH "$CONTAINER_PATH" TMPDIR /tmp TERM xterm-256color \
            CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS 0 CLAUDE_CODE_PLUGIN_GIT_TIMEOUT_MS 600000 \
            HF_HUB_CACHE /home/agent/.cache/huggingface/hub \
            HF_XET_CACHE /home/agent/.cache/huggingface/xet \
            HF_DATASETS_CACHE /home/agent/.cache/huggingface/datasets HF_HUB_OFFLINE 1 \
            DISABLE_AUTOUPDATER 1 IS_SANDBOX 1 CONTROL "$RUN/control" ROCM_PATH /opt/rocm \
            GPU_VENDOR amd
        for name in WS TRUSTED CASE AGENT MODEL EFFORT MODE GOAL_STYLE AGENT_SECONDS IDLE_SECONDS \
                    ALLOW_HOSTS FORBID RUN NET_MODE VIBESYS_PLUGIN_SOURCE VIBESYS_CLI_VERSION \
                    VIBESYS_KERNELS_PLUGIN RESULT_BRANCH "$@"; do
            printf '%s=%q\n' "$name" "${!name-}"
        done
    } > "$file"
    echo "$file"
}
# vibesys-cli keeps the machine's leases in /run/lock/vibesys-leases. The image's /run/lock is
# mode 1777 but read-only (squashfs), so every lease failed ("leased by another daemon") and no
# measure job ever started. Each agent container gets a private, writable one on local disk.
LEASES=$SCRATCH/run-lock
mkdir -p "$LEASES" && chmod 1777 "$LEASES"
common=("${GPU_FLAGS[@]}" --bind "$LEASES:/run/lock"
    --bind "$WORK:$WORK" --bind "${AGENT_TRUSTED:-$TRUSTED}:$TRUSTED:ro"
    --bind "$RUN/home:/home/agent" --env HOME=/home/agent --bind "$RUN/control:$RUN/control:ro"
    --pwd "$WS")

if [ "$ACTION" = smoke ]; then
    out=$LOGS/smoke-$JOB.txt
    EXPECT_GPUS=$(gpu_count)
    ct agent "${common[@]}" --env-file "$(agent_env EXPECT_GPUS)" -- \
        "$RUN/control/entry.sh" smoke > "$out" 2>&1 || true
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
if [ -n "$SMI" ]; then
    ( while sleep 1; do
          printf '%s,' "$(date -Is)"
          "$SMI" metric -u -m --csv 2>/dev/null | tail -n +2 | tr '\n' ';'
          echo
      done > "$LOGS/gpu-round-$NUMBER-$JOB.csv" ) &
    HELPER_PIDS+=($!)
fi

# The credential: Claude Code's credentials file, copied into the run's HOME for the agent
# only, without its refresh token (so no run can rotate the token that the host and the other
# runs hold), and removed when the agent exits.
case "$CREDENTIAL_KIND" in
    claude-credentials-json)
        umask 077
        mkdir -p "$(dirname "$CRED_COPY")"
        left=$(/usr/bin/python3 - "$CREDENTIAL_FILE" "$CRED_COPY" <<'PY'
import json, sys, time
data = json.load(open(sys.argv[1]))
oauth = data.get("claudeAiOauth", {})
oauth.pop("refreshToken", None)
oauth.pop("refreshTokenExpiresAt", None)
with open(sys.argv[2], "w") as out:
    json.dump(data, out)
print(int(oauth.get("expiresAt", 0) / 1000 - time.time()))
PY
)
        umask 022
        echo "credential: access token valid for $((left / 60)) more minutes (round needs $((AGENT_SECONDS / 60)))"
        if [ "$left" -lt "$AGENT_SECONDS" ]; then
            echo "WARNING: the access token expires before the round ends; the agent will lose access then" >&2
        fi
        ;;
    *)
        echo "the apptainer runtime takes --credential-kind claude-credentials-json" >&2
        exit 2 ;;
esac

start=$(date +%s)
set +e
# Each output line's arrival time goes to .times (line i of .out <-> line i of .times).
ct agent "${common[@]}" --env-file "$(agent_env)" --env ROUND="$NUMBER" -- \
    "$RUN/control/entry.sh" round 2> "$LOGS/agent-round-$NUMBER.err" \
    | /usr/bin/python3 -u -c 'import sys, time
out, times = open(sys.argv[1], "a"), open(sys.argv[2], "a")
for line in sys.stdin:
    out.write(line); out.flush(); times.write(f"{time.time():.3f}\n"); times.flush()' \
    "$LOGS/agent-round-$NUMBER.out" "$LOGS/agent-round-$NUMBER.times"
rc=${PIPESTATUS[0]}
set -e
rm -f "$CRED_COPY"
elapsed=$(( $(date +%s) - start ))
echo "agent exit $rc after ${elapsed}s"
# A usage limit of the agent's account stopped it: only the time until the first refusal counts,
# and a later round resumes the session once the limit lifts (container_campaign.py waits).
limited=$($HARVEST limit "$RUN" "$NUMBER" 2>> "$LOGS/harvest.err" || true)
if [ -n "$limited" ]; then
    read -r resets kind hit <<< "$limited"
    counted=$(( hit - start ))
    [ "$counted" -lt 0 ] && counted=0
    [ "$counted" -lt "$elapsed" ] && elapsed=$counted
    echo "$limited" > "$RUN/state/round-$NUMBER.limited"
    echo "usage limit ($kind) after ${elapsed}s of agent time; it lifts at $(date -d "@$resets" -Is)"
fi
echo "$elapsed $rc" > "$RUN/state/round-$NUMBER.elapsed"
$HARVEST harvest "$RUN" >/dev/null 2>> "$LOGS/harvest.err" || true
# 124: the round's time ran out, so a later round resumes; so does a usage limit, and a failed
# setup (exit 5, e.g. a GitHub clone that stalled). Anything else ends the session.
if [ "$rc" -eq 5 ] && [ -z "$limited" ]; then
    touch "$RUN/state/round-$NUMBER.setup-failed"
elif [ "$rc" -ne 124 ] && [ -z "$limited" ]; then
    echo "round $NUMBER, exit $rc after ${elapsed}s" > "$RUN/state/finished"
fi
if [ "$elapsed" -lt 600 ] && { [ -n "$limited" ] || [ "$rc" -eq 5 ]; }; then
    echo "round $NUMBER did under 10 minutes of work; not committing or grading it"
    exit 0
fi

# Snapshot the workspace in git (in a container: hooks and config are the agent's).
ct commit --net --network none --bind "$WS:$WS" --pwd "$WS" --env HOME=/tmp -- \
    bash -c 'git add -A && git -c core.hooksPath=/dev/null \
    -c user.name=MegaBench -c user.email=megabench@example.invalid commit -q --no-verify \
    -m "container_run: end of round $1" >/dev/null; git rev-parse HEAD' snapshot "$NUMBER" \
    > "$RUN/state/round-$NUMBER.commit" 2>> "$LOGS/grade-round-$NUMBER.err" || true

# Grade a copy: the trusted evaluator, no network, a baseline cache the agent never saw.
grade=$RUN/grade/round-$NUMBER-$JOB
mkdir -p "$grade"
if [ -n "${RESULT_BRANCH:-}" ]; then
    # A VibeSys run's result is the head of its result branch, not the checkout. With no
    # result branch (the bootstrap never passed) the checkout is graded.
    ct export --net --network none --bind "$WS:$WS:ro" --bind "$grade:/out" --pwd "$WS" \
        --env HOME=/tmp --env GIT_CONFIG_NOSYSTEM=1 -- \
        bash -c 'c=$(git -c core.fsmonitor= -c safe.directory="*" rev-parse --verify -q "refs/heads/$1^{commit}");
                 if [ -n "$c" ]; then git -c core.fsmonitor= -c safe.directory="*" archive "$c" | tar -x -C /out && echo "$1 $c";
                 else cp -a . /out/ && echo "checkout (no $1)"; fi' export "$RESULT_BRANCH" \
        > "$RUN/state/round-$NUMBER.graded" 2>> "$LOGS/grade-round-$NUMBER.err" || true
    echo "graded tree: $(cat "$RUN/state/round-$NUMBER.graded" 2>/dev/null)"
else
    cp -a "$WS/." "$grade/"
fi
rm -rf "$grade/.git" "$grade/.torch_extensions" "$grade/.cache" "$grade/reports" "$grade/.vibesys"
result=$RUN/results/round-$NUMBER-$JOB.jsonl
set -o noclobber
set +e
grade_dir "$grade" "$result" "$SESSION_ID-r$NUMBER" "$GRADE_REPS" 2>> "$LOGS/grade-round-$NUMBER.err"
grc=$?
set -e
echo "grading exit $grc; result $result"
head -c 2000 "$result" || true
echo
