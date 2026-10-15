"""Build the prefilled prompt caches of the real-weight DeepSeek-V3.2 cases.

Run once on the cases' GPU count, inside a Slurm allocation:

    torch.distributed.run --standalone --nproc-per-node=8 \\
        -m megabench.prepare_deepseek_v32 --case deepseek-v32-step \\
        --case deepseek-v32-step-ctx4k --case deepseek-v32-step-ctx32k

All missing prompt windows of the given cases are prefilled together with the
checkpoint (TP over all ranks, each layer read once), and
rank 0 writes ``$MEGABENCH_CACHE_DIR/deepseek-v32/<revision>/ctx<N>-prompt<i>.pt``
with the BF16 latent, rope and index-key caches, the window's tokens and the
prefill's teacher-forced next-token statistics (a check that the weights
loaded correctly). Existing caches are kept.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import torch
import torch.distributed as dist

from .cases import select_cases
from .tasks import deepseek_checkpoint as checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', action='append', required=True, help='case id (repeatable)')
    parser.add_argument('--prompt', type=int, action='append', help='prompt index (default: all)')
    args = parser.parse_args()
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", rank))}'
    torch.cuda.set_device(device)
    log = lambda message: print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}',
                                flush=True) if rank == 0 else None
    jobs = []
    for case in select_cases('all', args.case):
        if not case.checkpoint or case.family != 'deepseek_v32_step':
            raise SystemExit(f'{case.id}: not a real-weight DeepSeek-V3.2 case')
        if dist.get_world_size() != case.tp:
            raise SystemExit(f'{case.id}: run with {case.tp} ranks')
        for prompt in args.prompt or range(case.params['prompts']):
            path = checkpoint.cache_path(case, prompt)
            if path.exists():
                log(f'{path} exists; kept')
            else:
                jobs.append((case, prompt))
    if jobs:
        log(f'prefilling {len(jobs)} prompt windows')
        records = checkpoint.build_prompt_caches(jobs, device, rank=rank, keep_caches=rank == 0, log=log)
        if rank == 0:
            for (case, prompt), record in zip(jobs, records):
                path = checkpoint.save_prompt_cache(case, prompt, record)
                log(f'{path}: teacher-forced top-1 {record["top1_accuracy"]:.3f}, '
                    f'mean NLL {record["mean_nll"]:.3f}')
            log(f'prefill took {records[0]["prefill_s"]:.0f}s')
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
