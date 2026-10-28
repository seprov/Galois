#pragma once
// Minimal runtime for IrGL's fine-grained nested ForAll scheduler. This avoids
// graph/worklist/ModernGPU dependencies for applications with their own data.
// Use a CUDA toolkit providing CUB (CCCL) and compile as C++17 or newer.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cassert>
// Matches csr_graph.h; repeated typedefs to the same type are permitted.
typedef unsigned index_type;
#include "internal.h"
#include "ggcuda.h"
inline __device__ __host__ int irgl_roundup(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}
