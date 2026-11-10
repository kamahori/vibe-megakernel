#include "abi.cuh"
SEAM void stage_bias(const Params& p, Regs& r, int i) { p.out[i] = r.v + 1.0f; }
