#pragma once
#include <hip/hip_runtime.h>
struct Params { const float* in; float* out; float* mid; int n; };
struct Regs { float v; };
#define SEAM extern "C" __device__
SEAM void stage_scale(const Params& p, Regs& r, int i);
SEAM void stage_bias(const Params& p, Regs& r, int i);
