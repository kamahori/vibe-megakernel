#include "abi.cuh"
SEAM void stage_scale(const Params& p, Regs& r, int i) { r.v = p.in[i] * 2.0f; }
