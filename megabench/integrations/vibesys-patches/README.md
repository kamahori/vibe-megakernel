# VibeSys changes of the megabench loop

These patches are the VibeSys branch `megabench-loop`.
The megabench loop (`megabench/scripts/`, `clean_vibesys_run.py`) needs these patches.
Apply them to upstream uw-syfi/vibesys at commit `effaee9`:

```bash
git clone https://github.com/uw-syfi/vibesys.git && cd vibesys
git checkout -b megabench-loop effaee9
git am /path/to/vibe-megakernel/megabench/integrations/vibesys-patches/*.patch
```

Patch 0001 is the state of the installed VibeSys copy that the loop started from (session resume, multi-agent plugin options, write-megakernel skill).
Patches 0002 to 0009 are the loop changes: the microbench phase and its time limit, the ledger of open questions, crash resume, GPU queue time, and round numbers after a restart.

When the branch is on a VibeSys remote, `third_party/vibesys` will become a submodule at the branch tip, and this folder will be removed.
