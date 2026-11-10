# Run scripts

Scripts that prepare, start, watch and continue isolated VibeSys runs of megabench cases.
They take every machine path from `env.sh` (each setting can be set in the environment).

| Script | What it does |
|---|---|
| `prep.sh NAME CASE REF TOKEN_FILE [ROUNDS]` | Prepares `$VS_LOOP_HOME/runs/NAME` with `clean_vibesys_run` and prints the launch command. `SEED_FROM`, `SEED_REF` and `EXTRA_ARGS` (for example `--phase-agents --restart`) are read from the environment. |
| `failover.py RUN CASE REF TOTAL_ROUNDS` | Watches a run. It moves the run to another account when its account reaches a usage limit, and continues a run that ended early from its last closed round. |
| `launch_when_free.sh RUN CASE REF TOTAL_ROUNDS` | Starts a prepared run and its failover watcher when Slurm can start the run's GPU jobs at once. |
| `watch.sh RUN...` | Prints one line for each new round file, error line or run exit. |

VibeSys with the megabench-loop changes is the submodule `third_party/vibesys` (branch `megabench-loop` of uw-syfi/vibesys).
Get it with `git submodule update --init third_party/vibesys`, then install it (as its README says) into the environment that `VS_VIBESYS` points to.
`env.sh` takes the VibeSys skills from the submodule when it is checked out.

Example:

```bash
export VS_LOOP_HOME=/path/to/loop-home        # holds runs/, venv/ and the VibeSys install
export VS_FORBID=/path/one:/path/two          # more paths the agents must not see
export VS_NEVER_ACCOUNTS=me@example.com       # accounts that failover.py never uses
EXTRA_ARGS="--phase-agents" megabench/scripts/prep.sh my-run CASE REF ~/.secrets/vibesys_oauth_token_x 5
setsid nohup "$VS_LOOP_HOME/runs/my-run/launch.sh" > "$VS_LOOP_HOME/runs/my-run/vibesys.log" 2>&1 < /dev/null &
setsid nohup python3 megabench/scripts/failover.py my-run CASE REF 5 >> "$VS_LOOP_HOME/runs/failover.log" 2>&1 &
```
