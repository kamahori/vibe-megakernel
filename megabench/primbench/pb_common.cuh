// Shared helpers for primbench microbenchmarks (sm_100a).
// Every bench prints one flat JSON line per measured point: keys p.* (params),
// m.* (metrics), c.* (conditions), plus family, primitive, verified, n_reps, cv.
// `python -m megabench.primdb add FILE` turns those lines into primdb rows.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Per-CTA timing slot written by thread 0: clock64 delta and globaltimer delta (ns).
struct Stamp { long long cyc; unsigned long long ns; };

static inline int num_sms() {
  int n = 0; CK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0)); return n;
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

// Minimal JSON line builder: Json().s("family","x").i("p.M",128).f("m.cyc",54.4).print();
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
  void print() { printf("%s}\n", b.c_str()); fflush(stdout); }
};

// Clock in MHz from a set of stamps (median over CTAs).
static inline double clock_mhz(const std::vector<Stamp>& st) {
  std::vector<double> r;
  for (auto& s : st) if (s.ns) r.push_back((double)s.cyc / (double)s.ns * 1e3);
  return median(r);
}
