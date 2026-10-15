// Static, byte-balanced partitioning helpers (device side). The Python
// twin is loadstream/plan.py; use it to inspect balance before writing code.
#pragma once

namespace ls {

struct Range {
    int begin, end;
    __device__ __host__ int size() const { return end - begin; }
};

// Split `units` (e.g. rows, or row-granules of `granule` rows) as evenly as
// possible over `parts`; part p gets a contiguous range. The first
// (units % parts) parts get one extra unit.
__device__ __host__ inline Range band(int units, int parts, int p) {
    const int base = units / parts, extra = units % parts;
    const int begin = p * base + (p < extra ? p : extra);
    return {begin, begin + base + (p < extra ? 1 : 0)};
}

// Rows in granules of `granule` (e.g. 16 for mma tiles, or a head's rows).
__device__ __host__ inline Range band_rows(int rows, int granule, int parts, int p) {
    Range g = band((rows + granule - 1) / granule, parts, p);
    int e = g.end * granule;
    return {g.begin * granule, e < rows ? e : rows};
}

// Rows that fit one ring slot.
__device__ __host__ inline int rows_per_slot(int slot_bytes, int row_bytes) {
    int r = slot_bytes / row_bytes;
    return r > 0 ? r : 1;
}

}  // namespace ls
