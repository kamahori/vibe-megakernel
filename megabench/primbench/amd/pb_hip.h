// Shared helpers for the AMD (HIP) primbench probes: gfx90a, gfx942, gfx950.
// Every probe prints one flat JSON line per measured point (keys p.*, m.*, c.*, s.*, plus
// family, primitive, verified, n_reps, cv), the same format as the CUDA probes, so
// `python -m megabench.primdb add FILE` turns them into rows.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// XCD (XCC) of the calling wave; 0 on parts without XCDs (gfx90a).
__device__ __forceinline__ unsigned xcc_id() {
#if defined(__gfx942__) || defined(__gfx950__)
  unsigned x; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(x)); return x & 0xf;
#else
  return 0;
#endif
}
// Raw HW_ID register (CU, SIMD, SE fields; layout per ISA manual).
__device__ __forceinline__ unsigned hw_id() {
  unsigned x; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(x)); return x;
}

static inline int num_cus() {
  int n = 0; CK(hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, 0)); return n;
}
static inline std::string arch_name() {
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
  std::string a = p.gcnArchName; return a.substr(0, a.find(':'));
}
static inline double wall_mhz() {  // wall_clock64() rate
  int khz = 0; CK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0)); return khz / 1e3;
}

static inline double median(std::vector<double> v) {
  std::sort(v.begin(), v.end()); return v.empty() ? NAN : v[v.size() / 2];
}
static inline double cv(const std::vector<double>& v) {
  if (v.size() < 2) return 0;
  double m = 0, s = 0; for (double x : v) m += x; m /= v.size();
  for (double x : v) s += (x - m) * (x - m);
  return m ? std::sqrt(s / v.size()) / m : 0;
}

struct Json {
  std::string b = "{";
  Json& key(const char* k) { if (b.size() > 1) b += ","; b += "\""; b += k; b += "\":"; return *this; }
  Json& s(const char* k, const std::string& v) { key(k); b += "\"" + v + "\""; return *this; }
  Json& i(const char* k, long long v) { key(k); b += std::to_string(v); return *this; }
  Json& f(const char* k, double v) {
    key(k); char t[64];
    if (std::isfinite(v)) snprintf(t, sizeof t, "%.6g", v); else snprintf(t, sizeof t, "null");
    b += t; return *this;
  }
  Json& t(const char* k, bool v) { key(k); b += v ? "true" : "false"; return *this; }
  Json& null(const char* k) { key(k); b += "null"; return *this; }
  // arch, CU count and source file on every row
  Json& dev(const char* bench) {
    return s("c.arch", arch_name()).i("c.cus", num_cus()).s("s.bench", bench);
  }
  void print() { printf("%s}\n", b.c_str()); fflush(stdout); }
};

// Time `fn` (which enqueues work) with HIP events; returns ms per call, median of reps.
template <class F> static std::vector<double> time_ms(F fn, int reps, int warm = 2) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  for (int i = 0; i < warm; ++i) fn();
  CK(hipDeviceSynchronize());
  std::vector<double> out;
  for (int r = 0; r < reps; ++r) {
    CK(hipEventRecord(a)); fn(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); out.push_back(ms);
  }
  CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
  return out;
}
