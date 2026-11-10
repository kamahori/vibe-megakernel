#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static PFN_encodeTiled get_encode() {
  static PFN_encodeTiled fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (!p) throw std::runtime_error("cuTensorMapEncodeTiled not found");
    fn = (PFN_encodeTiled)p;
  }
  return fn;
}
// 2D bf16 tensor [rows, inner], row-major, box {box_inner, box_rows}, 128B swizzle
static CUtensorMap make_map(const void* ptr, uint64_t inner, uint64_t rows, uint32_t box_inner, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, rows};
  cuuint64_t strides[1] = {inner * 2};
  cuuint32_t box[2] = {box_inner, box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("cuTensorMapEncodeTiled failed " + std::to_string((int)r));
  return m;
}
